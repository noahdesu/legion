#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>

#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  MAIN_TASK_ID,
  INIT_TASK_ID,
};

enum FieldIDs {
  FID_POINT,
};

struct Fpoint {
  double x;
  double y;
};

#define INIT_NUM_CLUSTERS 2

static double euclid_dist_2(Fpoint p, Fpoint c)
{
  return ((p.x - c.x) * (p.x - c.x)) + ((p.y - c.y) * (p.y - c.y));
}

static unsigned find_nearest_cluster(Fpoint p, Fpoint *clusters)
{
  unsigned index = 0;
  double min_dist = euclid_dist_2(p, clusters[0]);
  for (unsigned i = 1; i < INIT_NUM_CLUSTERS; i++) {
    double dist = euclid_dist_2(p, clusters[i]);
    if (dist < min_dist) {
      min_dist = dist;
      index = i;
    }
  }
  return index;
}

void main_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  int num_elements = 8192;
  int num_subregions = 4;

  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, 
      Domain::from_rect<1>(elem_rect));

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(Fpoint), FID_POINT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  Rect<1> color_bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  assert((num_elements % num_subregions) == 0);
  Blockify<1> coloring(num_elements/num_subregions);
  IndexPartition ip = runtime->create_index_partition(ctx, is, coloring);

  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  Fpoint init_cluster_points[INIT_NUM_CLUSTERS];
  init_cluster_points[0].x = 0.25;
  init_cluster_points[0].y = 0.25;
  init_cluster_points[1].x = 0.75;
  init_cluster_points[1].y = 0.75;
#if 0
  for (unsigned i = 0; i < INIT_NUM_CLUSTERS; i++) {
    init_cluster_points[i].x = drand48();
    init_cluster_points[i].y = drand48();
  }
#endif

  IndexLauncher init_launcher(INIT_TASK_ID, color_domain,
      TaskArgument(init_cluster_points, sizeof(init_cluster_points)),
      ArgumentMap());

  init_launcher.add_region_requirement(
      RegionRequirement(lp, 0,
        WRITE_DISCARD, EXCLUSIVE, lr));
  init_launcher.region_requirements[0].add_field(FID_POINT);
  runtime->execute_index_space(ctx, init_launcher);

  InlineLauncher pts_launcher(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
  pts_launcher.add_field(FID_POINT);
  PhysicalRegion pr = runtime->map_region(ctx, pts_launcher);
  pr.wait_until_valid();

  RegionAccessor<AccessorType::Generic, Fpoint> pt_acc = 
    pr.get_field_accessor(FID_POINT).typeify<Fpoint>();

  std::stringstream filename;
  filename << "init_cluster_pts." << task->get_unique_task_id() << ".dat";

  std::ofstream init_pts_file;
  init_pts_file.open(filename.str().c_str(), std::ios::out | std::ios::trunc);
  assert(init_pts_file.is_open());

  Fpoint clusters[INIT_NUM_CLUSTERS];
  for (unsigned i = 0; i < INIT_NUM_CLUSTERS; i++) {
    clusters[i].x = drand48();
    clusters[i].y = drand48();
    init_pts_file << clusters[i].x << " " << clusters[i].y << std::endl;
  }

  long clusterSizes[INIT_NUM_CLUSTERS];
  memset(clusterSizes, 0, sizeof(clusterSizes));

  Fpoint newClusters[INIT_NUM_CLUSTERS];
  memset(newClusters, 0, sizeof(clusterSizes));

  for (int xx = 0; xx < 5; xx++) {

    for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++) {
      Fpoint p = pt_acc.read(DomainPoint::from_point<1>(pir.p));
      unsigned cluster_index = find_nearest_cluster(p, clusters);
      clusterSizes[cluster_index]++;
      newClusters[cluster_index].x += p.x;
      newClusters[cluster_index].y += p.y;
    }

    for (unsigned i = 0; i < INIT_NUM_CLUSTERS; i++) {
      if (clusterSizes[i] > 1) {
        clusters[i].x = newClusters[i].x / clusterSizes[i];
        clusters[i].y = newClusters[i].y / clusterSizes[i];
      }
      newClusters[i].x = 0;
      newClusters[i].y = 0;
      clusterSizes[i] = 0;
    }

  }

  filename.str("");
  filename << "final_cluster_pts." << task->get_unique_task_id() << ".dat";

  std::ofstream final_pts_file;
  final_pts_file.open(filename.str().c_str(), std::ios::out | std::ios::trunc);
  assert(final_pts_file.is_open());

  for (unsigned i = 0; i < INIT_NUM_CLUSTERS; i++) {
    final_pts_file << clusters[i].x << " " << clusters[i].y << std::endl;
  }

  final_pts_file.close();
}

void init_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  Fpoint *init_cluster_points = (Fpoint*)task->args;

  RegionAccessor<AccessorType::Generic, Fpoint> pt_acc = 
    regions[0].get_field_accessor(FID_POINT).typeify<Fpoint>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    double u = drand48();
    double v = drand48();
    double w = 0.5 * std::sqrt(u);
    double t = 2.0 * M_PI * v;
    double x = w * std::cos(t);
    double y = w * std::sin(t);

    unsigned cluster = std::rand() % INIT_NUM_CLUSTERS + 1;

    Fpoint pt;
    pt.x = init_cluster_points[cluster].x + x;
    pt.y = init_cluster_points[cluster].y + y;

    pt_acc.write(DomainPoint::from_point<1>(pir.p), pt);
  }

  std::stringstream filename;
  filename << "pts." << task->get_unique_task_id() << ".dat";

  std::ofstream pts_file;
  pts_file.open(filename.str().c_str(), std::ios::out | std::ios::trunc);
  assert(pts_file.is_open());

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    Fpoint pt = pt_acc.read(DomainPoint::from_point<1>(pir.p));
    pts_file << pt.x << " " << pt.y << std::endl;
  }

  pts_file.close();
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(MAIN_TASK_ID);

  HighLevelRuntime::register_legion_task<main_task>(MAIN_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "main");

  HighLevelRuntime::register_legion_task<init_task>(INIT_TASK_ID,
      Processor::LOC_PROC, true/*single*/, true/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "init");

  return HighLevelRuntime::start(argc, argv);
}

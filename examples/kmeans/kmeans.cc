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

struct DoublePoint {
  DoublePoint() {}
  DoublePoint(double x, double y) :
    x(x), y(y)
  {}

  double euclid_dist_2(const DoublePoint& o) const {
    double xd = x - o.x;
    double yd = y - o.y;
    return xd*xd + yd*yd;
  }

  double x;
  double y;
};

static std::ostream& operator<<(std::ostream &os, const DoublePoint& pt)
{
  os << pt.x << " " << pt.y;
  return os;
}

static int find_nearest_cluster(DoublePoint pt,
    const DoublePoint *clusters, int num_clusters)
{
  int index = 0;

  double min_dist = pt.euclid_dist_2(clusters[0]);
  for (int i = 1; i < num_clusters; i++) {
    double dist = pt.euclid_dist_2(clusters[i]);
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
  int num_clusters = 2;

  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, 
      Domain::from_rect<1>(elem_rect));

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator =
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DoublePoint), FID_POINT);
  }

  LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

  Rect<1> color_bounds(Point<1>(0),Point<1>(num_subregions-1));
  Domain color_domain = Domain::from_rect<1>(color_bounds);

  assert((num_elements % num_subregions) == 0);
  Blockify<1> coloring(num_elements/num_subregions);
  IndexPartition ip = runtime->create_index_partition(ctx, is, coloring);

  LogicalPartition lp = runtime->get_logical_partition(ctx, lr, ip);

  IndexLauncher init_launcher(INIT_TASK_ID, color_domain,
      TaskArgument(NULL, 0), ArgumentMap());

  init_launcher.add_region_requirement(
      RegionRequirement(lp, 0,
        WRITE_DISCARD, EXCLUSIVE, lr));
  init_launcher.region_requirements[0].add_field(FID_POINT);
  runtime->execute_index_space(ctx, init_launcher);

  InlineLauncher pts_launcher(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
  pts_launcher.add_field(FID_POINT);
  PhysicalRegion pr = runtime->map_region(ctx, pts_launcher);
  pr.wait_until_valid();

  RegionAccessor<AccessorType::Generic, DoublePoint> pt_acc = 
    pr.get_field_accessor(FID_POINT).typeify<DoublePoint>();

  std::stringstream filename;
  filename << "init_cluster_pts." << task->get_unique_task_id() << ".dat";

  std::ofstream init_pts_file;
  init_pts_file.open(filename.str().c_str(), std::ios::out | std::ios::trunc);
  assert(init_pts_file.is_open());

  DoublePoint clusters[num_clusters];
  for (int i = 0; i < num_clusters; i++) {
    clusters[i].x = drand48();
    clusters[i].y = drand48();
    init_pts_file << clusters[i].x << " " << clusters[i].y << std::endl;
  }

  long clusterSizes[num_clusters];
  memset(clusterSizes, 0, sizeof(clusterSizes));

  DoublePoint newClusters[num_clusters];
  memset(newClusters, 0, sizeof(clusterSizes));

  for (int xx = 0; xx < 5; xx++) {

    for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++) {
      DoublePoint p = pt_acc.read(DomainPoint::from_point<1>(pir.p));
      int cluster_index = find_nearest_cluster(p, clusters, num_clusters);
      clusterSizes[cluster_index]++;
      newClusters[cluster_index].x += p.x;
      newClusters[cluster_index].y += p.y;
    }

    for (int i = 0; i < num_clusters; i++) {
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

  for (int i = 0; i < num_clusters; i++) {
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

  RegionAccessor<AccessorType::Generic, DoublePoint> pt_acc = 
    regions[0].get_field_accessor(FID_POINT).typeify<DoublePoint>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  /*
   * Create some clusters evenly spaced in [0,1].
   */
  int num_init_clusters = 2;
  DoublePoint init_clusters[num_init_clusters];
  double increment = 1.0 / (double)(num_init_clusters+1);
  double position = increment;
  for (int i = 0; i < num_init_clusters; i++) {
    init_clusters[i].x = position;
    init_clusters[i].y = position;
    position += increment;
  }

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    double u = drand48();
    double v = drand48();
    double w = 0.5 * std::sqrt(u);
    double t = 2.0 * M_PI * v;
    double x = w * std::cos(t);
    double y = w * std::sin(t);

    int cluster = std::rand() % num_init_clusters;

    DoublePoint pt(
        init_clusters[cluster].x + x,
        init_clusters[cluster].y + y);

    pt_acc.write(DomainPoint::from_point<1>(pir.p), pt);
  }

  std::stringstream filename;
  filename << "pts." << task->get_unique_task_id() << ".dat";

  std::ofstream pts_file;
  pts_file.open(filename.str().c_str(), std::ios::out | std::ios::trunc);
  assert(pts_file.is_open());

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    DoublePoint pt = pt_acc.read(DomainPoint::from_point<1>(pir.p));
    pts_file << pt << std::endl;
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

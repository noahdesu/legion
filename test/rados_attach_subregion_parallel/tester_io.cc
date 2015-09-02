/* Copyright 2015 Stanford University
 * Copyright 2015 Los Alamos National Laboratory 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include <rados/librados.hpp>
#include "legion.h"
#include "legion_io.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

void top_level_task(const Task *task,
		    const std::vector<PhysicalRegion> &regions,
		    Context ctx, HighLevelRuntime *runtime)
{
  uint64_t num_elements = 1024;
  int sub_regions = 64;
  int ndim = 2;
  volatile int debug_flag = 0;

  const InputArgs &command_args = HighLevelRuntime::get_input_args();
  for (int i = 1; i < command_args.argc; i++)
  {
    if (!strcmp(command_args.argv[i],"-n"))
      num_elements = atoll(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i],"-s"))
      sub_regions = atoi(command_args.argv[++i]);
    if (!strcmp(command_args.argv[i],"-d"))
      debug_flag = 1;
    if (!strcmp(command_args.argv[i], "-r"))
      ndim = atoi(command_args.argv[++i]);
  }

  while (debug_flag == 1) {}

  assert(ndim == 2 || ndim == 3);

  int elem_rect_hi_val;
  int color_hi_val;
  int patch_val;

  switch (ndim) {
    case 2:
      elem_rect_hi_val = sqrt(num_elements) - 1;
      color_hi_val = sqrt(sub_regions)-1;
      patch_val = sqrt(num_elements / sub_regions); 
      break;

    case 3:
      elem_rect_hi_val = std::ceil(std::pow(num_elements, 1/3.0)) - 1;
      color_hi_val = std::ceil(std::pow(sub_regions, 1/3.0)) - 1;
      patch_val = std::ceil(std::pow(num_elements / sub_regions, 1/3.0));
      assert(num_elements == (std::pow(patch_val, 3) * std::pow(color_hi_val+1, 3)));
      assert(num_elements == std::pow(elem_rect_hi_val+1, 3));
      break;

    default:
      assert(0);
  }

  std::cout << "Running legion IO tester with "
            << num_elements << " elements and "
            << sub_regions << " subregions" << std::endl;

  /*
   *
   */
  librados::Rados cluster;
  cluster.init(NULL);
  cluster.conf_read_file(NULL);
  cluster.conf_read_file("/home/nwatkins/ceph/src/ceph.conf");
  cluster.conf_read_file("/users/nwatkins/ceph/src/ceph.conf");
  cluster.conf_parse_env(NULL);
  int ret = cluster.connect();
  assert(ret == 0);

  librados::IoCtx ioctx;
  ret = cluster.ioctx_create("legion", ioctx);
  assert(ret == 0);

  /* give me color points to address my decomposition */ 
  Domain color_domain;
  switch (ndim) {
    case 2:
      {
        Point<2> color_lo; color_lo.x[0] = 0; color_lo.x[1] = 0;
        Point<2> color_hi; color_hi.x[0] = color_hi.x[1] = color_hi_val;
        Rect<2> color_bounds(color_lo, color_hi); 
        color_domain = Domain::from_rect<2>(color_bounds);
      }
      break;

    case 3:
      color_domain = Domain::from_rect<3>(Rect<3>(
            make_point(0, 0, 0),
            make_point(
              color_hi_val,
              color_hi_val,
              color_hi_val)));
      break;

    default:
      assert(0);
  }

  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
    allocator.allocate_field(sizeof(double),FID_SAL);
    allocator.allocate_field(sizeof(double),FID_KE);
    allocator.allocate_field(sizeof(double),FID_VOR);
  }

  FieldSpace persistent_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, persistent_fs);
    allocator.allocate_field(sizeof(double),FID_TEMP);
  }
 
  Domain elem_domain;
  switch (ndim) {
    case 2:
      {
        Point<2> elem_rect_lo; elem_rect_lo.x[0] = 0; elem_rect_lo.x[1]=0;
        Point<2> elem_rect_hi; elem_rect_hi.x[0] = elem_rect_hi.x[1] = elem_rect_hi_val;
        Rect<2> elem_rect( elem_rect_lo, elem_rect_hi );
        elem_domain = Domain::from_rect<2>(elem_rect);
      }
      break;

    case 3:
      elem_domain = Domain::from_rect<3>(Rect<3>(
            make_point(0, 0, 0),
            make_point(
              elem_rect_hi_val,
              elem_rect_hi_val,
              elem_rect_hi_val)));
      break;

    default:
      assert(0);
  }
 
  
  IndexSpace is = runtime->create_index_space(ctx, elem_domain);

  LogicalRegion ocean_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion persistent_lr = runtime->create_logical_region(ctx, is, persistent_fs);

  IndexPartition ip;
  switch (ndim) {
    case 2:
      {
        Point<2> patch_color; patch_color.x[0] = patch_color.x[1] = patch_val;
        Blockify<2> coloring(patch_color); 
        ip  = runtime->create_index_partition(ctx, is, coloring);
      }
      break;

    case 3:
      {
        Blockify<3> coloring(make_point(patch_val, patch_val, patch_val));
        ip  = runtime->create_index_partition(ctx, is, coloring);
      }
      break;

    default:
      assert(0);
  }
  runtime->attach_name(ip, "ip");

  LogicalPartition ocean_lp = runtime->get_logical_partition(ctx, ocean_lr, ip);
  LogicalPartition persistent_lp = runtime->get_logical_partition(ctx, persistent_lr, ip);
  runtime->attach_name(ocean_lp, "ocean_lp");
  
  //First initialize fields with some data
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_domain,
      TaskArgument(&patch_val, sizeof(patch_val)), ArgumentMap());

  // Use data parallel and task parallel 
  init_launcher.add_region_requirement(
      RegionRequirement(ocean_lp, 0/*projection ID*/,
        WRITE_DISCARD, EXCLUSIVE, ocean_lr));
  init_launcher.add_field(0, FID_TEMP);
  runtime->execute_index_space(ctx, init_launcher);

  /*
   *
   */
  PersistentRegion ocean_pr = PersistentRegion(runtime, &ioctx);

  std::map<FieldID, const char*> field_map;
  field_map.insert(std::make_pair(FID_TEMP, "bam/baz"));
  ocean_pr.create_persistent_subregions(ctx, "ocean_pr.hdf5",
      persistent_lr, persistent_lp, color_domain, field_map);

  ocean_pr.write_persistent_subregions(ctx, ocean_lr, ocean_lp);
  
  LogicalRegion ocean_check_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalPartition ocean_check_lp = runtime->get_logical_partition(ctx, ocean_check_lr, ip);
  
  ocean_pr.read_persistent_subregions(ctx, ocean_check_lr, ocean_check_lp);

  /*
   *
   */
  IndexLauncher check_launcher(CHECK_TASK_ID, color_domain,
      TaskArgument(NULL, 0), ArgumentMap());
  
  check_launcher.add_region_requirement(RegionRequirement(
        ocean_check_lp, 0, READ_ONLY, EXCLUSIVE, ocean_check_lr));

  check_launcher.add_region_requirement(RegionRequirement(
        ocean_lp, 0, READ_ONLY, EXCLUSIVE, ocean_lr));

  check_launcher.region_requirements[0].add_field(FID_TEMP);
  check_launcher.region_requirements[1].add_field(FID_TEMP);
  
  runtime->execute_index_space(ctx, check_launcher);
  
  runtime->destroy_logical_region(ctx, ocean_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

void init_field_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);
  int extent = *(const int*) task->args;
 
  std::cout << "init_field_task extent is: " << extent << " domain point is:[" << task->index_point[0] << "," <<
    task->index_point[1] << "]" << " linearization is: " << task->index_point[0]*extent+task->index_point[1]*extent <<  std::endl;

  FieldID fid = *(task->regions[0].privilege_fields.begin());
  RegionAccessor<AccessorType::Generic, double> acc_temp = 
    regions[0].get_field_accessor(fid).typeify<double>();
  
  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());

  switch (dom.get_dim()) {
    case 2:
      {
        Rect<2> rect = dom.get_rect<2>();
        for (GenericPointInRectIterator<2> pir(rect); pir; pir++) {
          acc_temp.write(DomainPoint::from_point<2>(pir.p),
              task->index_point[0]*extent +
              task->index_point[1]*extent + drand48());
        }
      }
      break;

    case 3:
      {
        Rect<3> rect = dom.get_rect<3>();
        for (GenericPointInRectIterator<3> pir(rect); pir; pir++) {
          acc_temp.write(DomainPoint::from_point<3>(pir.p),
              task->index_point[0]*extent +
              task->index_point[1]*extent +
              task->index_point[2]*extent + drand48());
        }
      }
      break;

    default:
      assert(0);
  }
}

void check_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(task->regions.size() == 2);
  assert(task->regions[0].instance_fields.size() ==
         task->regions[1].instance_fields.size());

  bool all_passed = true;
  int values_checked = 0;

  Domain dom = runtime->get_index_space_domain(ctx, 
    task->regions[0].region.get_index_space());

  switch (dom.get_dim()) {
    case 2:
      {
        Rect<2> rect = dom.get_rect<2>();
        for (unsigned i = 0; i < task->regions[0].instance_fields.size(); i++) {
          RegionAccessor<AccessorType::Generic, double> acc_src = 
            regions[0].get_field_accessor(i).typeify<double>();
          RegionAccessor<AccessorType::Generic, double> acc_dst = 
            regions[1].get_field_accessor(i).typeify<double>();
          for (GenericPointInRectIterator<2> pir(rect); pir; pir++) {
            if (acc_src.read(DomainPoint::from_point<2>(pir.p)) !=
                acc_dst.read(DomainPoint::from_point<2>(pir.p))) 
              all_passed = false;
            values_checked++;
          }
        }
      }
      break;

    case 3:
      {
        Rect<3> rect = dom.get_rect<3>();
        for (unsigned i = 0; i < task->regions[0].instance_fields.size(); i++) {
          RegionAccessor<AccessorType::Generic, double> acc_src = 
            regions[0].get_field_accessor(i).typeify<double>();
          RegionAccessor<AccessorType::Generic, double> acc_dst = 
            regions[1].get_field_accessor(i).typeify<double>();
          for (GenericPointInRectIterator<3> pir(rect); pir; pir++) {
            if (acc_src.read(DomainPoint::from_point<3>(pir.p)) !=
                acc_dst.read(DomainPoint::from_point<3>(pir.p))) 
              all_passed = false;
            values_checked++;
          }
        }
      }
      break;

    default:
      assert(0);
  }
  
  if (all_passed)
    printf("SUCCESS! checked %d values\n", values_checked);
  else
    printf("FAILURE!\n");
}
  
int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
    Processor::LOC_PROC, true/*single*/, false/*index*/);
  HighLevelRuntime::register_legion_task<init_field_task>(INIT_FIELD_TASK_ID,
    Processor::LOC_PROC, true/*single*/, true/*index*/);
  HighLevelRuntime::register_legion_task<check_task>(CHECK_TASK_ID,
    Processor::LOC_PROC, true/*single*/, true/*index*/);
  PersistentRegion_init();
  
  return HighLevelRuntime::start(argc, argv);
}

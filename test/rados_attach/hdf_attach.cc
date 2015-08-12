/* Copyright 2015 Stanford University
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
#include "legion.h"
#include <rados/librados.hpp>
using namespace LegionRuntime::HighLevel;

/*
 * In this section we use a sequential
 * implementation of daxpy to show how
 * to create physical instances of logical
 * reigons.  In later sections we will
 * show how to extend this daxpy example
 * so that it will run with sub-tasks
 * and also run in parallel.
 */

// Note since we are now accessing data inside
// of logical regions we need the accessor namespace.
using namespace LegionRuntime::Accessor;


enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

void generate_rados_objects(const char *pool, const char* file_name, std::vector<const char*> path_names, int num_elements)
{
  librados::Rados cluster;
  cluster.init(NULL);
  cluster.conf_read_file("/home/nwatkins/ceph/src/ceph.conf");
  cluster.conf_parse_env(NULL);
  int ret = cluster.connect();
  assert(ret == 0);

  librados::IoCtx ioctx;
  ret = cluster.ioctx_create(pool, ioctx);
  assert(ret == 0);

  int *arr = (int*)calloc(num_elements, sizeof(int));
  for (int i = 0; i < num_elements; i++) {
    arr[i] = i;
  }

  for (std::vector<const char*>::iterator it = path_names.begin();
       it != path_names.end(); it++) {
    std::stringstream ss;
    ss << file_name << "." << *it;
    ceph::bufferlist bl;
    bl.append((const char*)arr, num_elements * sizeof(int));
    int ret = ioctx.write_full(ss.str(), bl);
    assert(ret == 0);
  }
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_elements = 1024;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }

  std::string input_file = "input.h5";
  std::string output_file = "output.h5";
  std::string x_name = "/dset_x";
  std::string y_name = "/dset_y";
  std::string z_name = "/dset_z";

  std::vector<const char*> input_paths;
  input_paths.push_back(x_name.c_str());
  input_paths.push_back(y_name.c_str());

  std::vector<const char*> output_paths;
  output_paths.push_back(z_name.c_str());

  printf("Generating Rados objects...\n");
  generate_rados_objects("legion", input_file.c_str(), input_paths, num_elements);
  generate_rados_objects("legion", output_file.c_str(), output_paths, num_elements);

  printf("Running daxpy for %d elements...\n", num_elements);

  // We'll create two logical regions with a common index space
  // for storing our inputs and outputs.  The input region will
  // have two fields for storing the 'x' and 'y' fields of the
  // daxpy computation, and the output region will have a single
  // field 'z' for storing the result.
  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, 
      Domain::from_rect<1>(elem_rect));

  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(int),FID_X);
    allocator.allocate_field(sizeof(int),FID_Y);
  }

  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(int),FID_Z);
  }

  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);

  std::map<FieldID, const char*> input_map;
  input_map[FID_X] = x_name.c_str();
  input_map[FID_Y] = y_name.c_str();
  PhysicalRegion input_region = runtime->attach_rados(ctx, input_file.c_str(),
      input_lr, input_lr, input_map, LEGION_FILE_READ_WRITE);
  runtime->remap_region(ctx, input_region);
  input_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, int> acc_x =
    input_region.get_field_accessor(FID_X).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> acc_y =
    input_region.get_field_accessor(FID_Y).typeify<int>();

  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    acc_x.write(DomainPoint::from_point<1>(pir.p), rand());
    acc_y.write(DomainPoint::from_point<1>(pir.p), rand());
  }

  std::map<FieldID, const char*> output_map;
  output_map[FID_Z] = z_name.c_str();

  PhysicalRegion output_region = runtime->attach_rados(ctx, output_file.c_str(),
      output_lr, output_lr, output_map, LEGION_FILE_READ_WRITE);
  runtime->remap_region(ctx, output_region);
  output_region.wait_until_valid();

  RegionAccessor<AccessorType::Generic, int> acc_z =
    output_region.get_field_accessor(FID_Z).typeify<int>();

  // Iterate over our points and perform the daxpy computation.  Note
  // we can use the same iterator because both the input and output
  // regions were created using the same index space.
  const int alpha = rand() % 100;
  printf("Running daxpy computation with alpha %d...", alpha);
  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    int value = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p)) +
                           acc_y.read(DomainPoint::from_point<1>(pir.p));
    acc_z.write(DomainPoint::from_point<1>(pir.p), value);
  }
  printf("Done!\n");

  printf("Checking results...");
  bool all_passed = true;
  // Check our results are the same
  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    int expected = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p)) + 
                           acc_y.read(DomainPoint::from_point<1>(pir.p));
    int received = acc_z.read(DomainPoint::from_point<1>(pir.p));
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");

  runtime->detach_rados(ctx, input_region);
  printf("Detached input file...\n");
  runtime->detach_rados(ctx, output_region);
  printf("Detached output file...\n");

  // Clean up all our data structures.
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
  printf("Finish top-level task...\n");
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/);

  return HighLevelRuntime::start(argc, argv);
}

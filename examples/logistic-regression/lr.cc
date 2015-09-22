/*
 * points = spark.textFile(...).map(parsePoint).cache()
 * w = numpy.random.ranf(size = D) # current separating plane
 * for i in range(ITERATIONS):
 *     gradient = points.map(
 *         lambda p: (1 / (1 + exp(-p.y*(w.dot(p.x)))) - 1) * p.y * p.x
 *     ).reduce(lambda a, b: a + b)
 *     w -= gradient
 * print "Final separating plane: %s" % w
 */
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;
using namespace LegionRuntime::Arrays;

enum TaskIDs {
  MAIN_TASK_ID,
  INIT_TASK_ID,
  GRADIENT_TASK_ID,
};

enum FieldIDs {
  FID_POINT,
};

enum RedopIDs {
  ADD_REDOP_ID = 1,
};

// point with float-point coordinates
struct Fpoint {
  double x;
  double y;
};

/*
 * Addition reduction operator for Fpoint.
 *
 * TODO: this should work because the update to the point components don't
 * have to be consistent with respect to each other.
 *
 * TODO: do Fpoint::x and Fpoint::y need a forced alignemnt for the compare
 * and swap?
 */
struct AddRedop {
  typedef Fpoint LHS;
  typedef Fpoint RHS;
  static const Fpoint identity;

  template <bool EXCLUSIVE>
  static void apply(LHS& lhs, RHS rhs);

  template <bool EXCLUSIVE>
  static void fold(RHS& rhs1, RHS rhs2);
};

const Fpoint AddRedop::identity = {0.0, 0.0};

template<>
void AddRedop::apply<true>(LHS& lhs, RHS rhs)
{
  lhs.x += rhs.x;
  lhs.y += rhs.y;
}

template<>
void AddRedop::apply<false>(LHS& lhs, RHS rhs)
{
  {
    int64_t *target = (int64_t *)&lhs.x;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs.x;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }

  {
    int64_t *target = (int64_t *)&lhs.y;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs.y;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
}

template<>
void AddRedop::fold<true>(RHS& rhs1, RHS rhs2)
{
  rhs1.x += rhs2.x;
  rhs1.y += rhs2.y;
}

template<>
void AddRedop::fold<false>(RHS &rhs1, RHS rhs2)
{
  {
    int64_t *target = (int64_t *)&rhs1.x;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs2.x;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }

  {
    int64_t *target = (int64_t *)&rhs1.y;
    union { int64_t as_int; double as_T; } oldval, newval;
    do {
      oldval.as_int = *target;
      newval.as_T = oldval.as_T + rhs2.y;
    } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
  }
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

  IndexLauncher init_launcher(INIT_TASK_ID, color_domain,
      TaskArgument(NULL, 0), ArgumentMap());

  init_launcher.add_region_requirement(
      RegionRequirement(lp, 0,
        WRITE_DISCARD, EXCLUSIVE, lr));
  init_launcher.region_requirements[0].add_field(FID_POINT);
  runtime->execute_index_space(ctx, init_launcher);

#if 0
  /*
   * This si a very simple, easy to understand example. We can demonstrate
   * this with all in memory, tiering, and remote reductions in storage.
   */

  >>> (1 / (1 + exp(-0.5*(w.dot(0.5)))) - 1) * 0.5 * 0.5
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        NameError: name 'exp' is not defined
        >>> (1 / (1 + numpy.exp(-0.5*(w.dot(0.5)))) - 1) * 0.5 * 0.5
        array([-0.11214672, -0.1129544 ])

  w is 2d
  the lambda func produces a 2d output
  the reduction produces gradient which is 2d
  ranf provides float, so everything is double
#endif

  Fpoint w;
  w.x = drand48();
  w.y = drand48();

  for (int i = 0; i < 100; i++) {

    IndexLauncher gradient_launcher(GRADIENT_TASK_ID, color_domain,
        TaskArgument(&w, sizeof(w)), ArgumentMap());

    gradient_launcher.add_region_requirement(
        RegionRequirement(lp, 0,
          READ_ONLY, EXCLUSIVE, lr));
    gradient_launcher.region_requirements[0].add_field(FID_POINT);

    Future gradient_future = runtime->execute_index_space(ctx,
        gradient_launcher, ADD_REDOP_ID);

    Fpoint gradient = gradient_future.get_result<Fpoint>();

    w.x -= gradient.x;
    w.y -= gradient.y;

    std::cout << i << ": " << w.x <<  " " << w.y << std::endl;
  }

  runtime->destroy_logical_region(ctx, lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}


void init_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  RegionAccessor<AccessorType::Generic, Fpoint> pt_acc = 
    regions[0].get_field_accessor(FID_POINT).typeify<Fpoint>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    Fpoint pt;
    pt.x = drand48();
    pt.y = drand48();
    pt_acc.write(DomainPoint::from_point<1>(pir.p), pt);
  }
}

Fpoint gradient_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1); 
  assert(task->regions.size() == 1);
  assert(task->regions[0].privilege_fields.size() == 1);

  Fpoint w = *((Fpoint*)task->args);
  Fpoint ret = {0.0, 0.0};

  RegionAccessor<AccessorType::Generic, Fpoint> pt_acc = 
    regions[0].get_field_accessor(FID_POINT).typeify<Fpoint>();

  Domain dom = runtime->get_index_space_domain(ctx, 
      task->regions[0].region.get_index_space());
  Rect<1> rect = dom.get_rect<1>();

  for (GenericPointInRectIterator<1> pir(rect); pir; pir++) {
    Fpoint p = pt_acc.read(DomainPoint::from_point<1>(pir.p));

    // lambda p: (1 / (1 + exp(-p.y*(w.dot(p.x)))) - 1) * p.y * p.x
    double x = (1.0 / (1.0 + std::exp(-p.y * w.x * p.x)) - 1) * p.y * p.x;
    double y = (1.0 / (1.0 + std::exp(-p.y * w.y * p.x)) - 1) * p.y * p.x;

    // reduce(lambda a, b: a + b)
    ret.x += x;
    ret.y += y;
  }

  return ret;
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

  HighLevelRuntime::register_legion_task<Fpoint, gradient_task>(
      GRADIENT_TASK_ID, Processor::LOC_PROC, true, true,
      AUTO_GENERATE_ID, TaskConfigOptions(true), "gradient");

  HighLevelRuntime::register_reduction_op<AddRedop>(ADD_REDOP_ID);

  return HighLevelRuntime::start(argc, argv);
}

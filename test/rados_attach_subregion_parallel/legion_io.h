#ifndef __LEGION_IO_H__
#define __LEGION_IO_H__

#include <rados/librados.hpp>
#include "legion.h"

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

static inline void current_utc_time(struct timespec *ts) {
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  int ret = clock_gettime(CLOCK_REALTIME, ts);
  assert(!ret);
#endif
}

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  STENCIL_TASK_ID,
  CHECK_TASK_ID,
  COPY_VALUES_TASK_ID,
};

enum FieldIDs {
  FID_TEMP,
  FID_SAL,
  FID_KE,
  FID_VOR,
  FID_PERS,
};

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

struct Piece {
  LogicalRegion parent_lr;
  LogicalRegion child_lr;
  PhysicalRegion pr;
  DomainPoint dp;
  char shard_name[40];
  FieldID field_ids[10];
  char field_names [2][10];
}; 

extern void PersistentRegion_init();

class PersistentRegion {
 public:
  PersistentRegion(HighLevelRuntime *runtime, librados::IoCtx *ioctx) :
    runtime(runtime), ioctx(ioctx)
  {}

  void create_persistent_subregions(Context ctx, const char *name,
      LogicalRegion parent_lr, LogicalPartition lp,
      Domain dom, std::map<FieldID, const char*>& field_map); 

  void write_persistent_subregions(Context ctx,
      LogicalRegion src_lr, LogicalPartition src_lp);

  void read_persistent_subregions(Context ctx,
      LogicalRegion src_lr, LogicalPartition src_lp);

  LogicalRegion get_logical_region() {
    return parent_lr;
  }

  LogicalPartition get_logical_partition() {
    return lp;
  }

 private:
  HighLevelRuntime *runtime;
  librados::IoCtx *ioctx;
  std::vector<Piece> pieces; 
  LogicalPartition  lp;
  LogicalRegion parent_lr;
  Domain dom;
  std::map<FieldID, const char*> field_map; 
};

#endif

/* Copyright 2015 Stanford University, NVIDIA Corporation
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

// Runtime implementation for Realm

#include "runtime_impl.h"

#include "proc_impl.h"
#include "mem_impl.h"
#include "inst_impl.h"

#include "activemsg.h"

#ifndef USE_GASNET
/*extern*/ void *fake_gasnet_mem_base = 0;
/*extern*/ size_t fake_gasnet_mem_size = 0;
#endif

// remote copy active messages from from lowlevel_dma.h for now
#include "lowlevel_dma.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::RemoteCopyMessage RemoteCopyMessage;
  typedef LegionRuntime::LowLevel::RemoteFillMessage RemoteFillMessage;
};

#ifdef USE_CUDA
#include "lowlevel_gpu.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::GPUProcessor GPUProcessor;
  typedef LegionRuntime::LowLevel::GPUFBMemory GPUFBMemory;
  typedef LegionRuntime::LowLevel::GPUZCMemory GPUZCMemory;
  typedef LegionRuntime::LowLevel::GPUWorker GPUWorker;
};
#endif

// create xd message and update bytes read/write messages
#include "channel.h"
namespace Realm {
  typedef LegionRuntime::LowLevel::XferDesCreateMessage XferDesCreateMessage;
  typedef LegionRuntime::LowLevel::XferDesDestroyMessage XferDesDestroyMessage;
  typedef LegionRuntime::LowLevel::NotifyXferDesCompleteMessage NotifyXferDesCompleteMessage;
  typedef LegionRuntime::LowLevel::UpdateBytesWriteMessage UpdateBytesWriteMessage;
  typedef LegionRuntime::LowLevel::UpdateBytesReadMessage UpdateBytesReadMessage;
}

#include <unistd.h>
#include <signal.h>

#define CHECK_PTHREAD(cmd) do { \
  int ret = (cmd); \
  if(ret != 0) { \
    fprintf(stderr, "PTHREAD: %s = %d (%s)\n", #cmd, ret, strerror(ret)); \
    exit(1); \
  } \
} while(0)

namespace Realm {

  Logger log_runtime("realm");
  
  ////////////////////////////////////////////////////////////////////////
  //
  // signal handlers
  //

    static void realm_freeze(int signal)
    {
      assert((signal == SIGINT) || (signal == SIGABRT) ||
             (signal == SIGSEGV) || (signal == SIGFPE) ||
             (signal == SIGBUS));
      int process_id = getpid();
      char hostname[128];
      gethostname(hostname, 127);
      fprintf(stderr,"Legion process received signal %d: %s\n",
                      signal, strsignal(signal));
      fprintf(stderr,"Process %d on node %s is frozen!\n", 
                      process_id, hostname);
      fflush(stderr);
      while (true)
        sleep(1);
    }

  ////////////////////////////////////////////////////////////////////////
  //
  // class Runtime
  //

    Runtime::Runtime(void)
      : impl(0)
    {
      // ok to construct extra ones - we will make sure only one calls init() though
    }

    /*static*/ Runtime Runtime::get_runtime(void)
    {
      Runtime r;
      // explicit namespace qualifier here due to name collision
      r.impl = Realm::get_runtime();
      return r;
    }

    bool Runtime::init(int *argc, char ***argv)
    {
      if(runtime_singleton != 0) {
	fprintf(stderr, "ERROR: cannot initialize more than one runtime at a time!\n");
	return false;
      }

      impl = new RuntimeImpl;
      runtime_singleton = ((RuntimeImpl *)impl);
      return ((RuntimeImpl *)impl)->init(argc, argv);
    }
    
    bool Runtime::register_task(Processor::TaskFuncID taskid, Processor::TaskFuncPtr taskptr)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->task_table.count(taskid) > 0)
	return false;

      ((RuntimeImpl *)impl)->task_table[taskid] = taskptr;
      return true;
    }

    bool Runtime::register_reduction(ReductionOpID redop_id, const ReductionOpUntyped *redop)
    {
      assert(impl != 0);

      if(((RuntimeImpl *)impl)->reduce_op_table.count(redop_id) > 0)
	return false;

      ((RuntimeImpl *)impl)->reduce_op_table[redop_id] = redop;
      return true;
    }

    void Runtime::run(Processor::TaskFuncID task_id /*= 0*/,
		      RunStyle style /*= ONE_TASK_ONLY*/,
		      const void *args /*= 0*/, size_t arglen /*= 0*/,
                      bool background /*= false*/)
    {
      ((RuntimeImpl *)impl)->run(task_id, style, args, arglen, background);
    }

    void Runtime::shutdown(void)
    {
      ((RuntimeImpl *)impl)->shutdown(true); // local request
    }

    void Runtime::wait_for_shutdown(void)
    {
      ((RuntimeImpl *)impl)->wait_for_shutdown();

      // after the shutdown, we nuke the RuntimeImpl
      delete ((RuntimeImpl *)impl);
      impl = 0;
      runtime_singleton = 0;
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RuntimeImpl
  //

    RuntimeImpl *runtime_singleton = 0;

  // these should probably be member variables of RuntimeImpl?
    static std::vector<ProcessorImpl*> local_cpus;
    static std::vector<ProcessorImpl*> local_util_procs;
    static std::vector<ProcessorImpl*> local_io_procs;
    static size_t stack_size_in_mb;
#ifdef USE_CUDA
    static std::vector<GPUProcessor *> local_gpus;
    static std::map<GPUProcessor *, GPUFBMemory *> gpu_fbmems;
    static std::map<GPUProcessor *, GPUZCMemory *> gpu_zcmems;
#endif
  
    RuntimeImpl::RuntimeImpl(void)
      : machine(0), nodes(0), global_memory(0),
	local_event_free_list(0), local_barrier_free_list(0),
	local_reservation_free_list(0), local_index_space_free_list(0),
	local_proc_group_free_list(0), background_pthread(0)
    {
      machine = new MachineImpl;
    }

    RuntimeImpl::~RuntimeImpl(void)
    {
      delete machine;
    }

    bool RuntimeImpl::init(int *argc, char ***argv)
    {
      // have to register domain mappings too
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<1> >();
      LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<2> >();
      LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Arrays::CArrayLinearization<3> >();
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<1> >();
      LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<2> >();
      LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Arrays::FortranArrayLinearization<3> >();
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Arrays::Translation<1> >();
      // we also register split dim linearization
      LegionRuntime::Arrays::Mapping<1,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<1> >();
      LegionRuntime::Arrays::Mapping<2,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<2> >();
      LegionRuntime::Arrays::Mapping<3,1>::register_mapping<LegionRuntime::Layouts::SplitDimLinearization<3> >();


      DetailedTimer::init_timers();
      
      // low-level runtime parameters
#ifdef USE_GASNET
      size_t gasnet_mem_size_in_mb = 256;
#else
      size_t gasnet_mem_size_in_mb = 0;
#endif
      size_t cpu_mem_size_in_mb = 512;
      size_t reg_mem_size_in_mb = 0;
      size_t disk_mem_size_in_mb = 0;
      // Static variable for stack size since we need to 
      // remember it when we launch threads in run 
      stack_size_in_mb = 2;
      unsigned init_stack_count = 1;
      unsigned num_local_cpus = 1;
      unsigned num_util_procs = 1;
      unsigned num_io_procs = 0;
      //unsigned cpu_worker_threads = 1;
      unsigned dma_worker_threads = 1;
      unsigned active_msg_worker_threads = 1;
      unsigned active_msg_handler_threads = 1;
      bool     active_msg_sender_threads = false;
#ifdef USE_CUDA
      size_t zc_mem_size_in_mb = 64;
      size_t fb_mem_size_in_mb = 256;
      unsigned num_local_gpus = 0;
      unsigned num_gpu_streams = 12;
      bool     gpu_worker_thread = true;
      bool pin_sysmem_for_gpu = true;
#endif
#ifdef EVENT_TRACING
      size_t   event_trace_block_size = 1 << 20;
      double   event_trace_exp_arrv_rate = 1e3;
#endif
#ifdef LOCK_TRACING
      size_t   lock_trace_block_size = 1 << 20;
      double   lock_trace_exp_arrv_rate = 1e2;
#endif
      // should local proc threads get dedicated cores?
      bool bind_localproc_threads = true;
      bool use_greenlet_procs = true;
      bool disable_greenlets = false;

      for(int i = 1; i < *argc; i++) {
#define INT_ARG(argname, varname)                       \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = atoi((*argv)[++i]);		\
	    continue;					\
	  }

#define BOOL_ARG(argname, varname)                      \
	  if(!strcmp((*argv)[i], argname)) {		\
	    varname = true;				\
	    continue;					\
	  }

	INT_ARG("-ll:gsize", gasnet_mem_size_in_mb);
	INT_ARG("-ll:csize", cpu_mem_size_in_mb);
	INT_ARG("-ll:rsize", reg_mem_size_in_mb);
        INT_ARG("-ll:dsize", disk_mem_size_in_mb);
        INT_ARG("-ll:stacksize", stack_size_in_mb);
        INT_ARG("-ll:stacks", init_stack_count);
	INT_ARG("-ll:cpu", num_local_cpus);
	INT_ARG("-ll:util", num_util_procs);
        INT_ARG("-ll:io", num_io_procs);
	//INT_ARG("-ll:workers", cpu_worker_threads);
	INT_ARG("-ll:dma", dma_worker_threads);
	INT_ARG("-ll:amsg", active_msg_worker_threads);
	INT_ARG("-ll:ahandlers", active_msg_handler_threads);
        BOOL_ARG("-ll:senders", active_msg_sender_threads);
	INT_ARG("-ll:bind", bind_localproc_threads);
        BOOL_ARG("-ll:greenlet", use_greenlet_procs);
        BOOL_ARG("-ll:gdb", disable_greenlets);
#ifdef USE_CUDA
	INT_ARG("-ll:fsize", fb_mem_size_in_mb);
	INT_ARG("-ll:zsize", zc_mem_size_in_mb);
	INT_ARG("-ll:gpu", num_local_gpus);
        INT_ARG("-ll:streams", num_gpu_streams);
        BOOL_ARG("-ll:gpuworker", gpu_worker_thread);
        INT_ARG("-ll:pin", pin_sysmem_for_gpu);
#endif

	if(!strcmp((*argv)[i], "-ll:eventtrace")) {
#ifdef EVENT_TRACING
	  event_trace_file = strdup((*argv)[++i]);
#else
	  fprintf(stderr, "WARNING: event tracing requested, but not enabled at compile time!\n");
#endif
	  continue;
	}

        if (!strcmp((*argv)[i], "-ll:locktrace"))
        {
#ifdef LOCK_TRACING
          lock_trace_file = strdup((*argv)[++i]);
#else
          fprintf(stderr, "WARNING: lock tracing requested, but not enabled at compile time!\n");
#endif
          continue;
        }

        if (!strcmp((*argv)[i], "-ll:prefix"))
        {
#ifdef NODE_LOGGING
          RuntimeImpl::prefix = strdup((*argv)[++i]);
#else
          fprintf(stderr,"WARNING: prefix set, but NODE_LOGGING not enabled at compile time!\n");
#endif
          continue;
        }

        // Skip arguments that parsed in activemsg.cc
        if (!strcmp((*argv)[i], "-ll:numlmbs") || !strcmp((*argv)[i],"-ll:lmbsize") ||
            !strcmp((*argv)[i], "-ll:forcelong") || !strcmp((*argv)[i],"-ll:sdpsize"))
        {
          i++;
          continue;
        }

        if (strncmp((*argv)[i], "-ll:", 4) == 0)
        {
	  fprintf(stderr, "ERROR: unrecognized lowlevel option: %s\n", (*argv)[i]);
          assert(0);
	}
      }

      if(bind_localproc_threads) {
	// this has to preceed all spawning of threads, including the ones done by things like gasnet_init()
	proc_assignment = new ProcessorAssignment(num_local_cpus);

	// now move ourselves off the reserved cores
	proc_assignment->bind_thread(-1, 0, "machine thread");
      }

      if (disable_greenlets)
        use_greenlet_procs = false;
      if (use_greenlet_procs)
        greenlet::init_greenlet_library();

      //GASNetNode::my_node = new GASNetNode(argc, argv, this);
      // SJT: WAR for issue on Titan with duplicate cookies on Gemini
      //  communication domains
      char *orig_pmi_gni_cookie = getenv("PMI_GNI_COOKIE");
      if(orig_pmi_gni_cookie) {
        char *new_pmi_gni_cookie = (char *)malloc(256);
        sprintf(new_pmi_gni_cookie, "PMI_GNI_COOKIE=%d", 1+atoi(orig_pmi_gni_cookie));
        //printf("changing PMI cookie to: '%s'\n", new_pmi_gni_cookie);
        putenv(new_pmi_gni_cookie);  // libc now owns the memory
      }
      // SJT: another GASNET workaround - if we don't have GASNET_IB_SPAWNER set, assume it was MPI
      if(!getenv("GASNET_IB_SPAWNER"))
	putenv(strdup("GASNET_IB_SPAWNER=mpi"));
#ifdef DEBUG_REALM_STARTUP
      { // we don't have rank IDs yet, so everybody gets to spew
        char s[80];
        gethostname(s, 79);
        strcat(s, " enter gasnet_init");
        TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif
      CHECK_GASNET( gasnet_init(argc, argv) );
#ifdef DEBUG_REALM_STARTUP
      { // once we're convinced there isn't skew here, reduce this to rank 0
        char s[80];
        gethostname(s, 79);
        strcat(s, " exit gasnet_init");
        TimeStamp ts(s, false);
        fflush(stdout);
      }
#endif

      // Check that we have enough resources for the number of nodes we are using
      if (gasnet_nodes() > MAX_NUM_NODES)
      {
        fprintf(stderr,"ERROR: Launched %d nodes, but runtime is configured "
                       "for at most %d nodes. Update the 'MAX_NUM_NODES' macro "
                       "in legion_types.h", gasnet_nodes(), MAX_NUM_NODES);
        gasnet_exit(1);
      }
      if (gasnet_nodes() > (1 << ID::NODE_BITS))
      {
#ifdef LEGION_IDS_ARE_64BIT
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes. Update the allocation "
                       "of bits in ID", gasnet_nodes(), (1 << ID::NODE_BITS));
#else
        fprintf(stderr,"ERROR: Launched %d nodes, but low-level IDs are only "
                       "configured for at most %d nodes.  Update the allocation "
                       "of bits in ID or switch to 64-bit IDs with the "
                       "-DLEGION_IDS_ARE_64BIT compile-time flag",
                       gasnet_nodes(), (1 << ID::NODE_BITS));
#endif
        gasnet_exit(1);
      }

      // initialize barrier timestamp
      BarrierImpl::barrier_adjustment_timestamp = (((Barrier::timestamp_t)(gasnet_mynode())) << BarrierImpl::BARRIER_TIMESTAMP_NODEID_SHIFT) + 1;

      Logger::configure_from_cmdline(*argc, (const char **)*argv);

      gasnet_handlerentry_t handlers[128];
      int hcount = 0;
      hcount += NodeAnnounceMessage::Message::add_handler_entries(&handlers[hcount], "Node Announce AM");
      hcount += SpawnTaskMessage::Message::add_handler_entries(&handlers[hcount], "Spawn Task AM");
      hcount += LockRequestMessage::Message::add_handler_entries(&handlers[hcount], "Lock Request AM");
      hcount += LockReleaseMessage::Message::add_handler_entries(&handlers[hcount], "Lock Release AM");
      hcount += LockGrantMessage::Message::add_handler_entries(&handlers[hcount], "Lock Grant AM");
      hcount += EventSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Event Subscribe AM");
      hcount += EventTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Event Trigger AM");
      hcount += RemoteMemAllocRequest::Request::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Request AM");
      hcount += RemoteMemAllocRequest::Response::add_handler_entries(&handlers[hcount], "Remote Memory Allocation Response AM");
      hcount += CreateInstanceRequest::Request::add_handler_entries(&handlers[hcount], "Create Instance Request AM");
      hcount += CreateInstanceRequest::Response::add_handler_entries(&handlers[hcount], "Create Instance Response AM");
      hcount += RemoteCopyMessage::add_handler_entries(&handlers[hcount], "Remote Copy AM");
      hcount += RemoteFillMessage::add_handler_entries(&handlers[hcount], "Remote Fill AM");
      hcount += ValidMaskRequestMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Request AM");
      hcount += ValidMaskDataMessage::Message::add_handler_entries(&handlers[hcount], "Valid Mask Data AM");
#ifdef DETAILED_TIMING
      hcount += TimerDataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Request AM");
      hcount += TimerDataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Roll-up Data AM");
      hcount += ClearTimersMessage::Message::add_handler_entries(&handlers[hcount], "Clear Timer Request AM");
#endif
      hcount += DestroyInstanceMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Instance AM");
      hcount += RemoteWriteMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write AM");
      hcount += RemoteReduceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduce AM");
      hcount += RemoteWriteFenceMessage::Message::add_handler_entries(&handlers[hcount], "Remote Write Fence AM");
      hcount += DestroyLockMessage::Message::add_handler_entries(&handlers[hcount], "Destroy Lock AM");
      hcount += RemoteReduceListMessage::Message::add_handler_entries(&handlers[hcount], "Remote Reduction List AM");
      hcount += RuntimeShutdownMessage::Message::add_handler_entries(&handlers[hcount], "Machine Shutdown AM");
      hcount += BarrierAdjustMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Adjust AM");
      hcount += BarrierSubscribeMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Subscribe AM");
      hcount += BarrierTriggerMessage::Message::add_handler_entries(&handlers[hcount], "Barrier Trigger AM");
      hcount += MetadataRequestMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Request AM");
      hcount += MetadataResponseMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Response AM");
      hcount += MetadataInvalidateMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Invalidate AM");
      hcount += MetadataInvalidateAckMessage::Message::add_handler_entries(&handlers[hcount], "Metadata Inval Ack AM");
      hcount += XferDesCreateMessage::Message::add_handler_entries(&handlers[hcount], "Create XferDes Request AM");
      hcount += XferDesDestroyMessage::Message::add_handler_entries(&handlers[hcount], "Destroy XferDes Request AM");
      hcount += NotifyXferDesCompleteMessage::Message::add_handler_entries(&handlers[hcount], "Notify XferDes Completion Request AM");
      hcount += UpdateBytesWriteMessage::Message::add_handler_entries(&handlers[hcount], "Update Bytes Write AM");
      hcount += UpdateBytesReadMessage::Message::add_handler_entries(&handlers[hcount], "Update Bytes Read AM");
      //hcount += TestMessage::add_handler_entries(&handlers[hcount], "Test AM");
      //hcount += TestMessage2::add_handler_entries(&handlers[hcount], "Test 2 AM");

      init_endpoints(handlers, hcount, 
		     gasnet_mem_size_in_mb, reg_mem_size_in_mb,
		     *argc, (const char **)*argv);
#ifndef USE_GASNET
      // network initialization is also responsible for setting the "zero_time"
      //  for relative timing - no synchronization necessary in non-gasnet case
      Realm::Clock::set_zero_time();
#endif

      // Put this here so that it complies with the GASNet specification and
      // doesn't make any calls between gasnet_init and gasnet_attach
      gasnet_set_waitmode(GASNET_WAIT_BLOCK);

      nodes = new Node[gasnet_nodes()];

      // create allocators for local node events/locks/index spaces
      {
	Node& n = nodes[gasnet_mynode()];
	local_event_free_list = new EventTableAllocator::FreeList(n.events, gasnet_mynode());
	local_barrier_free_list = new BarrierTableAllocator::FreeList(n.barriers, gasnet_mynode());
	local_reservation_free_list = new ReservationTableAllocator::FreeList(n.reservations, gasnet_mynode());
	local_index_space_free_list = new IndexSpaceTableAllocator::FreeList(n.index_spaces, gasnet_mynode());
	local_proc_group_free_list = new ProcessorGroupTableAllocator::FreeList(n.proc_groups, gasnet_mynode());
      }

#ifdef DEADLOCK_TRACE
      next_thread = 0;
      signaled_threads = 0;
      signal(SIGTERM, deadlock_catch);
      signal(SIGINT, deadlock_catch);
#endif
#if defined(REALM_BACKTRACE) || defined(LEGION_BACKTRACE)
      signal(SIGSEGV, realm_backtrace);
      signal(SIGABRT, realm_backtrace);
      signal(SIGFPE,  realm_backtrace);
      signal(SIGILL,  realm_backtrace);
      signal(SIGBUS,  realm_backtrace);
#endif
      if ((getenv("LEGION_FREEZE_ON_ERROR") != NULL) ||
          (getenv("REALM_FREEZE_ON_ERROR") != NULL))
      {
        signal(SIGSEGV, realm_freeze);
        signal(SIGABRT, realm_freeze);
        signal(SIGFPE,  realm_freeze);
        signal(SIGILL,  realm_freeze);
        signal(SIGBUS,  realm_freeze);
      }
      
      start_polling_threads(active_msg_worker_threads);

      start_handler_threads(active_msg_handler_threads, stack_size_in_mb << 20);

      //LegionRuntime::LowLevel::start_dma_worker_threads(dma_worker_threads);

      if (active_msg_sender_threads)
        start_sending_threads();

#ifdef EVENT_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save event info
      Tracer<EventTraceItem>::init_trace(event_trace_block_size,
                                         event_trace_exp_arrv_rate);
#endif
#ifdef LOCK_TRACING
      // Always initialize even if we won't dump to file, otherwise segfaults happen
      // when we try to save lock info
      Tracer<LockTraceItem>::init_trace(lock_trace_block_size,
                                        lock_trace_exp_arrv_rate);
#endif
	
      //gasnet_seginfo_t seginfos = new gasnet_seginfo_t[num_nodes];
      //CHECK_GASNET( gasnet_getSegmentInfo(seginfos, num_nodes) );

      if(gasnet_mem_size_in_mb > 0)
	global_memory = new GASNetMemory(ID(ID::ID_MEMORY, 0, ID::ID_GLOBAL_MEM, 0).convert<Memory>(), gasnet_mem_size_in_mb << 20);
      else
	global_memory = 0;

      Node *n = &nodes[gasnet_mynode()];

      // create utility processors (if any)
      if (num_util_procs > 0)
      {
        for(unsigned i = 0; i < num_util_procs; i++) {
          ProcessorImpl *up;
          if (use_greenlet_procs)
            up = new GreenletProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, stack_size_in_mb << 20, 
                                    init_stack_count, "utility worker");
          else
            up = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(), 
                                    n->processors.size()).convert<Processor>(),
                                    Processor::UTIL_PROC, 
                                    stack_size_in_mb << 20, "utility worker");
          n->processors.push_back(up);
          local_util_procs.push_back(up);
        }
      }
      // create i/o processors (if any)
      if (num_io_procs > 0)
      {
        for (unsigned i = 0; i < num_io_procs; i++) {
          LocalProcessor *io = new LocalProcessor(ID(ID::ID_PROCESSOR, gasnet_mynode(),
                                            n->processors.size()).convert<Processor>(),
                                            Processor::IO_PROC,
                                            stack_size_in_mb << 20, "io worker");
          n->processors.push_back(io);
          local_io_procs.push_back(io);
        }
      }

#ifdef USE_CUDA
      // Initialize the driver API
      CHECK_CU( cuInit(0) );
      // Keep track of the local system memories so we can pin them
      // after we've initialized the GPU
      std::vector<LocalCPUMemory*> local_mems;
      // Figure out which GPUs support peer access (if any)
      // and prioritize them so they are used first
      std::vector<int> peer_gpus;
      std::vector<int> dumb_gpus;
      {
        int num_devices;
        CHECK_CU( cuDeviceGetCount(&num_devices) );
        for (int i = 0; i < num_devices; i++)
        {
          CUdevice device;
          CHECK_CU( cuDeviceGet(&device, i) );
          bool has_peer = false;
          // Go through all the other devices and see
          // if we have peer access to them
          for (int j = 0; j < num_devices; j++)
          {
            if (i == j) continue;
            CUdevice peer;
            CHECK_CU( cuDeviceGet(&peer, j) );
            int can_access;
            CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
            if (can_access)
            {
              has_peer = true;
              break;
            }
          }
          if (has_peer)
            peer_gpus.push_back(i);
          else
            dumb_gpus.push_back(i);
        }
      }
#endif
      // create local processors
      for(unsigned i = 0; i < num_local_cpus; i++) {
	Processor p = ID(ID::ID_PROCESSOR, 
			 gasnet_mynode(), 
			 n->processors.size()).convert<Processor>();
        ProcessorImpl *lp;
        if (use_greenlet_procs)
          lp = new GreenletProcessor(p, Processor::LOC_PROC,
                                     stack_size_in_mb << 20, init_stack_count,
                                     "local worker", i);
        else
	  lp = new LocalProcessor(p, Processor::LOC_PROC,
                                  stack_size_in_mb << 20,
                                  "local worker", i);
	n->processors.push_back(lp);
	local_cpus.push_back(lp);
      }

      // create local memory
      LocalCPUMemory *cpumem;
      if(cpu_mem_size_in_mb > 0) {
	cpumem = new LocalCPUMemory(ID(ID::ID_MEMORY, 
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    cpu_mem_size_in_mb << 20);
	n->memories.push_back(cpumem);
#ifdef USE_CUDA
        local_mems.push_back(cpumem);
#endif
      } else
	cpumem = 0;

      LocalCPUMemory *regmem;
      if(reg_mem_size_in_mb > 0) {
	gasnet_seginfo_t *seginfos = new gasnet_seginfo_t[gasnet_nodes()];
	CHECK_GASNET( gasnet_getSegmentInfo(seginfos, gasnet_nodes()) );
	char *regmem_base = ((char *)(seginfos[gasnet_mynode()].addr)) + (gasnet_mem_size_in_mb << 20);
	delete[] seginfos;
	regmem = new LocalCPUMemory(ID(ID::ID_MEMORY,
				       gasnet_mynode(),
				       n->memories.size(), 0).convert<Memory>(),
				    reg_mem_size_in_mb << 20,
				    regmem_base,
				    true);
	n->memories.push_back(regmem);
#ifdef USE_CUDA
        local_mems.push_back(regmem);
#endif
      } else
	regmem = 0;

      // create local disk memory
      DiskMemory *diskmem;
      if(disk_mem_size_in_mb > 0) {
        char file_name[30];
        sprintf(file_name, "disk_file%d.tmp", gasnet_mynode());
        diskmem = new DiskMemory(ID(ID::ID_MEMORY,
                                    gasnet_mynode(),
                                    n->memories.size(), 0).convert<Memory>(),
                                 disk_mem_size_in_mb << 20,
                                 std::string(file_name));
        n->memories.push_back(diskmem);
      } else
        diskmem = 0;

#ifdef USE_RADOS
      // FIXME: pass pool name in at runtime
      RadosMemory *rados_mem = new RadosMemory(ID(ID::ID_MEMORY, gasnet_mynode(),
            n->memories.size(), 0).convert<Memory>(), "legion");
      n->memories.push_back(rados_mem);
#endif

#ifdef USE_HDF
      // create HDF memory
      HDFMemory *hdfmem;
      hdfmem = new HDFMemory(ID(ID::ID_MEMORY,
                                gasnet_mynode(),
                                n->memories.size(), 0).convert<Memory>());
      n->memories.push_back(hdfmem);
#endif



#ifdef USE_CUDA
      if(num_local_gpus > 0) {
        if (num_local_gpus > (peer_gpus.size() + dumb_gpus.size()))
        {
          printf("Requested %d GPUs, but only %ld GPUs exist on node %d\n",
            num_local_gpus, peer_gpus.size()+dumb_gpus.size(), gasnet_mynode());
          assert(false);
        }
        GPUWorker *gpu_worker = 0;
        if (gpu_worker_thread) {
          gpu_worker = GPUWorker::start_gpu_worker_thread(stack_size_in_mb << 20);
        }
	for(unsigned i = 0; i < num_local_gpus; i++) {
	  Processor p = ID(ID::ID_PROCESSOR, 
			   gasnet_mynode(), 
			   n->processors.size()).convert<Processor>();
	  //printf("GPU's ID is " IDFMT "\n", p.id);
 	  GPUProcessor *gp = new GPUProcessor(p, Processor::TOC_PROC, "gpu worker",
                                              (i < peer_gpus.size() ?
                                                peer_gpus[i] : 
                                                dumb_gpus[i-peer_gpus.size()]), 
                                              zc_mem_size_in_mb << 20,
                                              fb_mem_size_in_mb << 20,
                                              stack_size_in_mb << 20,
                                              gpu_worker, num_gpu_streams);
	  n->processors.push_back(gp);
	  local_gpus.push_back(gp);

	  Memory m = ID(ID::ID_MEMORY,
			gasnet_mynode(),
			n->memories.size(), 0).convert<Memory>();
	  GPUFBMemory *fbm = new GPUFBMemory(m, gp);
	  n->memories.push_back(fbm);

	  gpu_fbmems[gp] = fbm;

	  Memory m2 = ID(ID::ID_MEMORY,
			 gasnet_mynode(),
			 n->memories.size(), 0).convert<Memory>();
	  GPUZCMemory *zcm = new GPUZCMemory(m2, gp);
	  n->memories.push_back(zcm);

	  gpu_zcmems[gp] = zcm;
	}
        // Now pin any CPU memories
        if(pin_sysmem_for_gpu)
          for (unsigned idx = 0; idx < local_mems.size(); idx++)
            local_mems[idx]->pin_memory(local_gpus[0]);

        // Register peer access for any GPUs which support it
        if ((num_local_gpus > 1) && (peer_gpus.size() > 1))
        {
          unsigned peer_count = (num_local_gpus < peer_gpus.size()) ? 
                                  num_local_gpus : peer_gpus.size();
          // Needs to go both ways so register in all directions
          for (unsigned i = 0; i < peer_count; i++)
          {
            CUdevice device;
            CHECK_CU( cuDeviceGet(&device, peer_gpus[i]) );
            for (unsigned j = 0; j < peer_count; j++)
            {
              if (i == j) continue;
              CUdevice peer;
              CHECK_CU( cuDeviceGet(&peer, peer_gpus[j]) );
              int can_access;
              CHECK_CU( cuDeviceCanAccessPeer(&can_access, device, peer) );
              if (can_access)
                local_gpus[i]->enable_peer_access(local_gpus[j]);
            }
          }
        }
      }
#endif

      {
	const unsigned ADATA_SIZE = 4096;
	size_t adata[ADATA_SIZE];
	unsigned apos = 0;

	unsigned num_procs = 0;
	unsigned num_memories = 0;

	for(std::vector<ProcessorImpl *>::const_iterator it = local_util_procs.begin();
	    it != local_util_procs.end();
	    it++) {
	  num_procs++;
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::UTIL_PROC;
	}

	for(std::vector<ProcessorImpl *>::const_iterator it = local_io_procs.begin();
	    it != local_io_procs.end();
	    it++) {
	  num_procs++;
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::IO_PROC;
	}

	for(std::vector<ProcessorImpl *>::const_iterator it = local_cpus.begin();
	    it != local_cpus.end();
	    it++) {
	  num_procs++;
          adata[apos++] = NODE_ANNOUNCE_PROC;
          adata[apos++] = (*it)->me.id;
          adata[apos++] = Processor::LOC_PROC;
	}

	// memories
	if(cpumem) {
	  num_memories++;
	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = Memory::SYSTEM_MEM;
	  adata[apos++] = cpumem->size;
	  adata[apos++] = 0; // not registered
	}

	if(regmem) {
	  num_memories++;
	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = regmem->me.id;
	  adata[apos++] = Memory::REGDMA_MEM;
	  adata[apos++] = regmem->size;
	  adata[apos++] = (size_t)(regmem->base);
	}

	if(diskmem) {
	  num_memories++;
	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = diskmem->me.id;
	  adata[apos++] = Memory::DISK_MEM;
	  adata[apos++] = diskmem->size;
	  adata[apos++] = 0;
	}

#ifdef USE_RADOS
	if (rados_mem) {
	  num_memories++;
	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = rados_mem->me.id;
	  adata[apos++] = Memory::RADOS_MEM;
	  adata[apos++] = rados_mem->size;
	  adata[apos++] = 0;
	}
#endif

#ifdef USE_HDF
	if(hdfmem) {
	  num_memories++;
	  adata[apos++] = NODE_ANNOUNCE_MEM;
	  adata[apos++] = hdfmem->me.id;
	  adata[apos++] = Memory::HDF_MEM;
	  adata[apos++] = hdfmem->size;
	  adata[apos++] = 0;
	}
#endif

	// list affinities between local CPUs / memories
	std::vector<ProcessorImpl *> all_local_procs;
	all_local_procs.insert(all_local_procs.end(),
			       local_util_procs.begin(), local_util_procs.end());
	all_local_procs.insert(all_local_procs.end(),
			       local_io_procs.begin(), local_io_procs.end());
	all_local_procs.insert(all_local_procs.end(),
			       local_cpus.begin(), local_cpus.end());
	for(std::vector<ProcessorImpl*>::iterator it = all_local_procs.begin();
	    it != all_local_procs.end();
	    it++) {
	  if(cpumem) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = cpumem->me.id;
	    adata[apos++] = 100;  // "large" bandwidth
	    adata[apos++] = 1;    // "small" latency
	  }

	  if(regmem) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = regmem->me.id;
	    adata[apos++] = 80;  // "large" bandwidth
	    adata[apos++] = 5;    // "small" latency
	  }

	  if(diskmem) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = diskmem->me.id;
	    adata[apos++] = 5;  // "low" bandwidth
	    adata[apos++] = 100;  // "high" latency
	  }

#ifdef USE_RADOS
	  if (rados_mem) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = rados_mem->me.id;
	    adata[apos++] = 5; // "low" bandwidth
	    adata[apos++] = 100; // "high" latency
	  } 
#endif

#ifdef USE_HDF
	  if(hdfmem) {
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = hdfmem->me.id;
	    adata[apos++] = 5; // "low" bandwidth
	    adata[apos++] = 100; // "high" latency
	  } 
#endif

	  if(global_memory) {
  	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = global_memory->me.id;
	    adata[apos++] = 10;  // "lower" bandwidth
	    adata[apos++] = 50;    // "higher" latency
	  }
	}

	if(cpumem && global_memory) {
	  adata[apos++] = NODE_ANNOUNCE_MMA;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = global_memory->me.id;
	  adata[apos++] = 30;  // "lower" bandwidth
	  adata[apos++] = 25;    // "higher" latency
	}

	if(cpumem && diskmem) {
	  adata[apos++] = NODE_ANNOUNCE_MMA;
	  adata[apos++] = cpumem->me.id;
	  adata[apos++] = diskmem->me.id;
	  adata[apos++] = 15;    // "low" bandwidth
	  adata[apos++] = 50;    // "high" latency
	}

#ifdef USE_CUDA
	for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	    it != local_gpus.end();
	    it++)
	{
	  num_procs++;
	  adata[apos++] = NODE_ANNOUNCE_PROC;
	  adata[apos++] = (*it)->me.id;
	  adata[apos++] = Processor::TOC_PROC;

	  GPUFBMemory *fbm = gpu_fbmems[*it];
	  if(fbm) {
	    num_memories++;

	    adata[apos++] = NODE_ANNOUNCE_MEM;
	    adata[apos++] = fbm->me.id;
	    adata[apos++] = Memory::GPU_FB_MEM;
	    adata[apos++] = fbm->size;
	    adata[apos++] = 0; // not registered

	    // FB has very good bandwidth and ok latency to GPU
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = fbm->me.id;
	    adata[apos++] = 200; // "big" bandwidth
	    adata[apos++] = 5;   // "ok" latency
	  }

	  GPUZCMemory *zcm = gpu_zcmems[*it];
	  if(zcm) {
	    num_memories++;

	    adata[apos++] = NODE_ANNOUNCE_MEM;
	    adata[apos++] = zcm->me.id;
	    adata[apos++] = Memory::Z_COPY_MEM;
	    adata[apos++] = zcm->size;
	    adata[apos++] = 0; // not registered

	    // ZC has medium bandwidth and bad latency to GPU
	    adata[apos++] = NODE_ANNOUNCE_PMA;
	    adata[apos++] = (*it)->me.id;
	    adata[apos++] = zcm->me.id;
	    adata[apos++] = 20;
	    adata[apos++] = 200;

	    // ZC also accessible to all the local CPUs
	    for(std::vector<ProcessorImpl*>::iterator it2 = local_cpus.begin();
		it2 != local_cpus.end();
		it2++) {
	      adata[apos++] = NODE_ANNOUNCE_PMA;
	      adata[apos++] = (*it2)->me.id;
	      adata[apos++] = zcm->me.id;
	      adata[apos++] = 40;
	      adata[apos++] = 3;
	    }
	  }
	}
#endif

	adata[apos++] = NODE_ANNOUNCE_DONE;
	assert(apos < ADATA_SIZE);

	// parse our own data (but don't create remote proc/mem objects)
	machine->parse_node_announce_data(gasnet_mynode(),
					  num_procs,
					  num_memories,
					  adata, apos*sizeof(adata[0]), 
					  false);

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  TimeStamp ts("sending announcements", false);
	  fflush(stdout);
	}
#endif

	// now announce ourselves to everyone else
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    NodeAnnounceMessage::send_request(i,
						     num_procs,
						     num_memories,
						     adata, apos*sizeof(adata[0]),
						     PAYLOAD_COPY);

	NodeAnnounceMessage::await_all_announcements();

#ifdef DEBUG_REALM_STARTUP
	if(gasnet_mynode() == 0) {
	  TimeStamp ts("received all announcements", false);
	  fflush(stdout);
	}
#endif
      }

      // start dma system at the very ending of initialization
      // since we need list of local gpus to create channels
      LegionRuntime::LowLevel::start_dma_system(dma_worker_threads, 100
#ifdef USE_CUDA
                       ,local_gpus
#endif
                       );

      return true;
    }

    struct MachineRunArgs {
      RuntimeImpl *r;
      Processor::TaskFuncID task_id;
      Runtime::RunStyle style;
      const void *args;
      size_t arglen;
    };  

    static bool running_as_background_thread = false;

    static void *background_run_thread(void *data)
    {
      MachineRunArgs *args = (MachineRunArgs *)data;
      running_as_background_thread = true;
      args->r->run(args->task_id, args->style, args->args, args->arglen,
		   false /* foreground from this thread's perspective */);
      delete args;
      return 0;
    }

    void RuntimeImpl::run(Processor::TaskFuncID task_id /*= 0*/,
			  Runtime::RunStyle style /*= ONE_TASK_ONLY*/,
			  const void *args /*= 0*/, size_t arglen /*= 0*/,
			  bool background /*= false*/)
    { 
      if(background) {
        log_runtime.info("background operation requested\n");
	fflush(stdout);
	MachineRunArgs *margs = new MachineRunArgs;
	margs->r = this;
	margs->task_id = task_id;
	margs->style = style;
	margs->args = args;
	margs->arglen = arglen;
	
        pthread_t *threadp = (pthread_t*)malloc(sizeof(pthread_t));
	pthread_attr_t attr;
	CHECK_PTHREAD( pthread_attr_init(&attr) );
	CHECK_PTHREAD( pthread_create(threadp, &attr, &background_run_thread, (void *)margs) );
	CHECK_PTHREAD( pthread_attr_destroy(&attr) );
        background_pthread = threadp;
#ifdef DEADLOCK_TRACE
        this->add_thread(threadp); 
#endif
	return;
      }

      // Initialize the shutdown counter
      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;
      Atomic<int> running_proc_count(local_procs.size());

      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
	(*it)->run(&running_proc_count);

      // now that we've got the machine description all set up, we can start
      //  the worker threads for local processors, which'll probably ask the
      //  high-level runtime to set itself up
      for(std::vector<ProcessorImpl*>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->start_processor();

      for (std::vector<ProcessorImpl*>::iterator it = local_io_procs.begin();
            it != local_io_procs.end();
            it++)
        (*it)->start_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->start_processor();

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->start_processor();
#endif

      if(task_id != 0 && 
	 ((style != Runtime::ONE_TASK_ONLY) || 
	  (gasnet_mynode() == 0))) {//(gasnet_nodes()-1)))) {
	for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	    it != local_procs.end();
	    it++) {
	  (*it)->spawn_task(task_id, args, arglen, 
			    Event::NO_EVENT, Event::NO_EVENT, 0/*priority*/);
	  if(style != Runtime::ONE_TASK_PER_PROC) break;
	}
      }

      // wait for idle-ness somehow?
      int timeout = -1;
#ifdef TRACE_RESOURCES
      RuntimeImpl *rt = get_runtime();
#endif
      while(running_proc_count.get() > 0) {
	if(timeout >= 0) {
	  timeout--;
	  if(timeout == 0) {
	    printf("TIMEOUT!\n");
	    exit(1);
	  }
	}
	fflush(stdout);
	sleep(1);
#ifdef TRACE_RESOURCES
        log_runtime.info("total events: %d", rt->local_event_free_list->next_alloc);
        log_runtime.info("total reservations: %d", rt->local_reservation_free_list->next_alloc);
        log_runtime.info("total index spaces: %d", rt->local_index_space_free_list->next_alloc);
        log_runtime.info("total proc groups: %d", rt->local_proc_group_free_list->next_alloc);
#endif
      }
      log_runtime.info("running proc count is now zero - terminating\n");
#ifdef REPORT_REALM_RESOURCE_USAGE
      {
        RuntimeImpl *rt = get_runtime();
        printf("node %d realm resource usage: ev=%d, rsrv=%d, idx=%d, pg=%d\n",
               gasnet_mynode(),
               rt->local_event_free_list->next_alloc,
               rt->local_reservation_free_list->next_alloc,
               rt->local_index_space_free_list->next_alloc,
               rt->local_proc_group_free_list->next_alloc);
      }
#endif
#ifdef EVENT_GRAPH_TRACE
      {
        //FILE *log_file = Logger::get_log_file();
        show_event_waiters(/*log_file*/);
      }
#endif

      // Shutdown all the threads
      for(std::vector<ProcessorImpl*>::iterator it = local_util_procs.begin();
	  it != local_util_procs.end();
	  it++)
	(*it)->shutdown_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_io_procs.begin();
          it != local_io_procs.end();
          it++)
        (*it)->shutdown_processor();

      for(std::vector<ProcessorImpl*>::iterator it = local_cpus.begin();
	  it != local_cpus.end();
	  it++)
	(*it)->shutdown_processor();

#ifdef USE_CUDA
      for(std::vector<GPUProcessor *>::iterator it = local_gpus.begin();
	  it != local_gpus.end();
	  it++)
	(*it)->shutdown_processor(); 
#endif


      // delete local processors and memories
      {
	Node& n = nodes[gasnet_mynode()];

	for(std::vector<MemoryImpl *>::iterator it = n.memories.begin();
	    it != n.memories.end();
	    it++)
	  delete (*it);

	// node 0 also deletes the gasnet memory
	if(gasnet_mynode() == 0)
	  delete global_memory;
      }

      // need to kill other threads too so we can actually terminate process
      // Exit out of the thread
      //LegionRuntime::LowLevel::stop_dma_worker_threads();
      LegionRuntime::LowLevel::stop_dma_system();
#ifdef USE_CUDA
      GPUWorker::stop_gpu_worker_thread();
#endif
      stop_activemsg_threads();

      // if we are running as a background thread, just terminate this thread
      // if not, do a full process exit - gasnet may have started some threads we don't have handles for,
      //   and if they're left running, the app will hang
      if(running_as_background_thread) {
	pthread_exit(0);
      } else {
	exit(0);
      }
    }

    void RuntimeImpl::shutdown(bool local_request /*= true*/)
    {
      if(local_request) {
	log_runtime.info("shutdown request - notifying other nodes\n");
	for(unsigned i = 0; i < gasnet_nodes(); i++)
	  if(i != gasnet_mynode())
	    RuntimeShutdownMessage::send_request(i);
      }

      log_runtime.info("shutdown request - cleaning up local processors\n");

      const std::vector<ProcessorImpl *>& local_procs = nodes[gasnet_mynode()].processors;
      for(std::vector<ProcessorImpl *>::const_iterator it = local_procs.begin();
	  it != local_procs.end();
	  it++)
      {
        Event e = GenEventImpl::create_genevent()->current_event();
	(*it)->spawn_task(0 /* shutdown task id */, 0, 0,
			  Event::NO_EVENT, e, 0/*priority*/);
      }
    }

    void RuntimeImpl::wait_for_shutdown(void)
    {
      bool exit_process = true;
      if (background_pthread != 0)
      {
        pthread_t *background_thread = (pthread_t*)background_pthread;
        void *result;
        pthread_join(*background_thread, &result);
        free(background_thread);
        // Set this to null so we don't wait anymore
        background_pthread = 0;
        exit_process = false;
      }

#ifdef EVENT_TRACING
      if(event_trace_file) {
	printf("writing event trace to %s\n", event_trace_file);
        Tracer<EventTraceItem>::dump_trace(event_trace_file, false);
	free(event_trace_file);
	event_trace_file = 0;
      }
#endif
#ifdef LOCK_TRACING
      if (lock_trace_file)
      {
        printf("writing lock trace to %s\n", lock_trace_file);
        Tracer<LockTraceItem>::dump_trace(lock_trace_file, false);
        free(lock_trace_file);
        lock_trace_file = 0;
      }
#endif

      // this terminates the process, so control never gets back to caller
      // would be nice to fix this...
      if (exit_process)
        gasnet_exit(0);
    }

    EventImpl *RuntimeImpl::get_event_impl(Event e)
    {
      ID id(e);
      switch(id.type()) {
      case ID::ID_EVENT:
	return get_genevent_impl(e);
      case ID::ID_BARRIER:
	return get_barrier_impl(e);
      default:
	assert(0);
      }
    }

    GenEventImpl *RuntimeImpl::get_genevent_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_EVENT);

      Node *n = &nodes[id.node()];
      GenEventImpl *impl = n->events.lookup_entry(id.index(), id.node());
      assert(impl->me == id);

      // check to see if this is for a generation more than one ahead of what we
      //  know of - this should only happen for remote events, but if it does it means
      //  there are some generations we don't know about yet, so we can catch up (and
      //  notify any local waiters right away)
      impl->check_for_catchup(e.gen - 1);

      return impl;
    }

    BarrierImpl *RuntimeImpl::get_barrier_impl(Event e)
    {
      ID id(e);
      assert(id.type() == ID::ID_BARRIER);

      Node *n = &nodes[id.node()];
      BarrierImpl *impl = n->barriers.lookup_entry(id.index(), id.node());
      assert(impl->me == id);
      return impl;
    }

    ReservationImpl *RuntimeImpl::get_lock_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_LOCK:
	{
	  Node *n = &nodes[id.node()];
	  ReservationImpl *impl = n->reservations.lookup_entry(id.index(), id.node());
	  assert(impl->me == id.convert<Reservation>());
	  return impl;
#if 0
	  std::vector<ReservationImpl>& locks = nodes[id.node()].locks;

	  unsigned index = id.index();
	  if(index >= n->num_locks) {
	    AutoHSLLock a(n->mutex); // take lock before we actually resize

	    // grow our array to mirror additions by other nodes
	    //  this should never happen for our own node
	    assert(id.node() != gasnet_mynode());

	    unsigned oldsize = n->locks.size();
	    assert(oldsize < MAX_LOCAL_LOCKS);
	    if(index >= oldsize) { // only it's still too small
              assert((index+1) < MAX_LOCAL_LOCKS);
	      n->locks.resize(index + 1);
	      for(unsigned i = oldsize; i <= index; i++)
		n->locks[i].init(ID(ID::ID_LOCK, id.node(), i).convert<Reservation>(),
				 id.node());
	      n->num_locks = index + 1;
	    }
	  }
	  return &(locks[index]);
#endif
	}

      case ID::ID_INDEXSPACE:
	return &(get_index_space_impl(id)->lock);

      case ID::ID_INSTANCE:
	return &(get_instance_impl(id)->lock);

      case ID::ID_PROCGROUP:
	return &(get_procgroup_impl(id)->lock);

      default:
	assert(0);
      }
    }

    template <class T>
    inline T *null_check(T *ptr)
    {
      assert(ptr != 0);
      return ptr;
    }

    MemoryImpl *RuntimeImpl::get_memory_impl(ID id)
    {
      switch(id.type()) {
      case ID::ID_MEMORY:
      case ID::ID_ALLOCATOR:
      case ID::ID_INSTANCE:
	if(id.index_h() == ID::ID_GLOBAL_MEM)
	  return global_memory;
	return null_check(nodes[id.node()].memories[id.index_h()]);

      default:
	assert(0);
      }
    }

    ProcessorImpl *RuntimeImpl::get_processor_impl(ID id)
    {
      if(id.type() == ID::ID_PROCGROUP)
	return get_procgroup_impl(id);

      assert(id.type() == ID::ID_PROCESSOR);
      return null_check(nodes[id.node()].processors[id.index()]);
    }

    ProcessorGroup *RuntimeImpl::get_procgroup_impl(ID id)
    {
      assert(id.type() == ID::ID_PROCGROUP);

      Node *n = &nodes[id.node()];
      ProcessorGroup *impl = n->proc_groups.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<Processor>());
      return impl;
    }

    IndexSpaceImpl *RuntimeImpl::get_index_space_impl(ID id)
    {
      assert(id.type() == ID::ID_INDEXSPACE);

      Node *n = &nodes[id.node()];
      IndexSpaceImpl *impl = n->index_spaces.lookup_entry(id.index(), id.node());
      assert(impl->me == id.convert<IndexSpace>());
      return impl;
#if 0
      unsigned index = id.index();
      if(index >= n->index_spaces.size()) {
	AutoHSLLock a(n->mutex); // take lock before we actually resize

	if(index >= n->index_spaces.size())
	  n->index_spaces.resize(index + 1);
      }

      if(!n->index_spaces[index]) { // haven't seen this metadata before?
	//printf("UNKNOWN METADATA " IDFMT "\n", id.id());
	AutoHSLLock a(n->mutex); // take lock before we actually allocate
	if(!n->index_spaces[index]) {
	  n->index_spaces[index] = new IndexSpaceImpl(id.convert<IndexSpace>());
	} 
      }

      return n->index_spaces[index];
#endif
    }

    RegionInstanceImpl *RuntimeImpl::get_instance_impl(ID id)
    {
      assert(id.type() == ID::ID_INSTANCE);
      MemoryImpl *mem = get_memory_impl(id);
      
      AutoHSLLock al(mem->mutex);

      if(id.index_l() >= mem->instances.size()) {
	assert(id.node() != gasnet_mynode());

	size_t old_size = mem->instances.size();
	if(id.index_l() >= old_size) {
	  // still need to grow (i.e. didn't lose the race)
	  mem->instances.resize(id.index_l() + 1);

	  // don't have region/offset info - will have to pull that when
	  //  needed
	  for(unsigned i = old_size; i <= id.index_l(); i++) 
	    mem->instances[i] = 0;
	}
      }

      if(!mem->instances[id.index_l()]) {
	if(!mem->instances[id.index_l()]) {
	  //printf("[%d] creating proxy instance: inst=" IDFMT "\n", gasnet_mynode(), id.id());
	  mem->instances[id.index_l()] = new RegionInstanceImpl(id.convert<RegionInstance>(), mem->me);
	}
      }
	  
      return mem->instances[id.index_l()];
    }

  
  ////////////////////////////////////////////////////////////////////////
  //
  // class Node
  //

    Node::Node(void)
    {
    }


  ////////////////////////////////////////////////////////////////////////
  //
  // class RuntimeShutdownMessage
  //

  /*static*/ void RuntimeShutdownMessage::handle_request(RequestArgs args)
  {
    log_runtime.info("received shutdown request from node %d", args.initiating_node);

    get_runtime()->shutdown(false);
  }

  /*static*/ void RuntimeShutdownMessage::send_request(gasnet_node_t target)
  {
    RequestArgs args;

    args.initiating_node = gasnet_mynode();
    args.dummy = 0;
    Message::request(target, args);
  }

  
}; // namespace Realm

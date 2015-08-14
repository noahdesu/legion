#include <rados/librados.hpp>
  
namespace Realm {

  RadosMemory::RadosMemory(Memory m, const std::string pool)
      : MemoryImpl(m, 0, MKIND_RADOS, 256, Memory::RADOS_MEM)
    {
      std::cout << "Connecting to RADOS..." << std::endl;
      std::flush(std::cout);
      cluster.init(NULL);
      // FIXME: how to inject env vars through gasnet?
      cluster.conf_read_file("/home/nwatkins/ceph/src/ceph.conf");
      cluster.conf_parse_env(NULL);
      int ret = cluster.connect();
      assert(ret == 0);

      ret = cluster.ioctx_create(pool.c_str(), ioctx);
      assert(ret == 0);

      ret = pthread_mutex_init(&lock, NULL);
      assert(ret == 0);
    }

    RadosMemory::~RadosMemory()
    {}

    RegionInstance RadosMemory::create_instance(IndexSpace is,
        const int *linearization_bits,
        size_t bytes_needed,
        size_t block_size,
        size_t element_size,
        const std::vector<size_t>& field_sizes,
        ReductionOpID redopid,
        off_t list_size,
        const ProfilingRequestSet &reqs,
        RegionInstance parent_inst)
    {
      assert(0);
      return RegionInstance::NO_INST;
    }

    RegionInstance RadosMemory::create_instance(IndexSpace is,
        const int *linearization_bits,
        size_t bytes_needed,
        size_t block_size,
        size_t element_size,
        const std::vector<size_t>& field_sizes,
        ReductionOpID redopid,
        off_t list_size,
        const ProfilingRequestSet &reqs,
        RegionInstance parent_inst,
        const char* file,
        const std::vector<const char*>& path_names,
        Domain domain,
        bool read_only)
    {
      RegionInstance inst = create_instance_local(is,
                 linearization_bits, bytes_needed,
                 block_size, element_size, field_sizes, redopid,
                 list_size, reqs, parent_inst);

      RadosMemoryInst *rinst = new RadosMemoryInst;
      rinst->read_only = read_only;
      rinst->memory = this;
      rinst->file = std::string(file);

      rinst->ndims = domain.get_dim();
      for (int i = 0; i < rinst->ndims; i++) {
        rinst->lo[i] = domain.rect_data[i];
      }

      for (unsigned idx = 0; idx < path_names.size(); idx++) {
        std::stringstream ss;
        ss << rinst->file << "." << path_names[idx];
        rinst->objnames.push_back(ss.str());
      }

      pthread_mutex_lock(&lock);

      assert(instances.find(ID(inst).id()) == instances.end());
      instances[ID(inst).id()] = rinst;

      pthread_mutex_unlock(&lock);

      return inst;
    }

    RadosMemory::RadosMemoryInst *RadosMemory::get_specific_instance(RegionInstance inst)
    {
      pthread_mutex_lock(&lock);
      std::map<ID::IDType, RadosMemoryInst*>::const_iterator it =
        instances.find(ID(inst).id());
      assert(it != instances.end());
      RadosMemoryInst *ret = it->second;
      pthread_mutex_unlock(&lock);
      return ret;
    }

    void RadosMemory::destroy_instance(RegionInstance inst,
        bool local_destroy)
    {
      pthread_mutex_lock(&lock);
      std::map<ID::IDType, RadosMemoryInst*>::const_iterator it =
        instances.find(ID(inst).id());
      assert(it != instances.end());
      RadosMemoryInst *ret = it->second;
      pthread_mutex_unlock(&lock);
      delete ret;
      destroy_instance_local(inst, local_destroy);
    }

    off_t RadosMemory::alloc_bytes(size_t size)
    {
      return 0;
    }

    void RadosMemory::free_bytes(off_t offset, size_t size)
    {
    }

    void RadosMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void RadosMemory::get_bytes(RegionInstance inst, const DomainPoint& dp,
        int fid, void *dst, size_t size)
    {
      RadosMemoryInst *rinst = get_specific_instance(inst);

      int offset[3];
      for (int i = 0; i < rinst->ndims; i++) {
        offset[i] = dp.point_data[i] - rinst->lo[i];
      }

      // FIXME
      assert(size == sizeof(int));
      assert(rinst->ndims == 1);
      uint64_t objoffset = offset[0];

      std::string objname = rinst->objnames[fid];

      ceph::bufferlist bl;
      ceph::bufferptr bp = ceph::buffer::create_static(size, (char*)dst);
      bl.push_back(bp);

      int ret = ioctx.read(objname, bl, size, objoffset * size);
      assert(ret == (int)size);
      assert(bl.length() == size);
      if (bl.c_str() != dst)
        bl.copy(0, size, (char*)dst);
    }

    void RadosMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

    void RadosMemory::put_bytes(RegionInstance inst, const DomainPoint& dp,
        int fid, const void *src, size_t size)
    {
      RadosMemoryInst *rinst = get_specific_instance(inst);

      int offset[3];
      for (int i = 0; i < rinst->ndims; i++) {
        offset[i] = dp.point_data[i] - rinst->lo[i];
      }

      // FIXME
      assert(size == sizeof(int));
      assert(rinst->ndims == 1);
      uint64_t objoffset = offset[0];

      std::string objname = rinst->objnames[fid];
      ceph::bufferlist bl;
      bl.append((const char*)src, size);
      int ret = ioctx.write(objname, bl, size, objoffset * size);
      assert(ret == 0);
    }

    void *RadosMemory::get_direct_ptr(off_t offset, size_t size)
    {
      assert(0);
      return NULL;
    }

    int RadosMemory::get_home_node(off_t offset, size_t size)
    {
      assert(0);
      return 0;
    }
}

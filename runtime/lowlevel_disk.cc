/* Copyright 2015 Stanford University, NVIDIA Corporation
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

#include "lowlevel_impl.h"
#include "lowlevel.h"
#include <aio.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#ifdef USE_RADOS
#include <rados/librados.hpp>
#endif

namespace Realm {
  
    DiskMemory::DiskMemory(Memory _me, size_t _size, std::string _file)
      : MemoryImpl(_me, _size, MKIND_DISK, ALIGNMENT, Memory::DISK_MEM), file(_file)
    {
      printf("file = %s\n", _file.c_str());
      // do not overwrite an existing file
      fd = open(_file.c_str(), O_CREAT | O_EXCL | O_RDWR, 00777);
      assert(fd != -1);
      // resize the file to what we want
      int ret = ftruncate(fd, _size);
      assert(ret == 0);
      free_blocks[0] = _size;
    }

    DiskMemory::~DiskMemory(void)
    {
      close(fd);
      // attempt to delete the file
      unlink(file.c_str());
    }

    RegionInstance DiskMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     const Realm::ProfilingRequestSet &reqs,
                     RegionInstance parent_inst)
    {
      return create_instance_local(is, linearization_bits, bytes_needed,
                     block_size, element_size, field_sizes, redopid,
                     list_size, reqs, parent_inst);
    }

    void DiskMemory::destroy_instance(RegionInstance i,
				     bool local_destroy)
    {
      destroy_instance_local(i, local_destroy);
    }

    off_t DiskMemory::alloc_bytes(size_t size)
    {
      return alloc_bytes_local(size);
    }

    void DiskMemory::free_bytes(off_t offset, size_t size)
    {
      free_bytes_local(offset, size);
    }

    void DiskMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      aiocb cb;
      memset(&cb, 0, sizeof(cb));
      cb.aio_nbytes = size;
      cb.aio_fildes = fd;
      cb.aio_offset = offset;
      cb.aio_buf = dst;
      int ret = aio_read(&cb);
      assert(ret == 0);
      while(true) {
	ret = aio_error(&cb);
	if(ret == 0) break; // completed
	assert(ret == EINPROGRESS); // only other choice we expect to see
      }
      ssize_t count = aio_return(&cb);
      assert(count == (ssize_t)size);
    }

    void DiskMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      aiocb cb;
      memset(&cb, 0, sizeof(cb));
      cb.aio_nbytes = size;
      cb.aio_fildes = fd;
      cb.aio_offset = offset;
      cb.aio_buf = (void *)src;
      int ret = aio_write(&cb);
      assert(ret == 0);
      while(true) {
	ret = aio_error(&cb);
	if(ret == 0) break; // completed
	assert(ret == EINPROGRESS); // only other choice we expect to see
      }
      ssize_t count = aio_return(&cb);
      assert(count == (ssize_t)size);
    }

    void DiskMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                        size_t count, const void *entry_buffer)
    {
    }

    void *DiskMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    int DiskMemory::get_home_node(off_t offset, size_t size)
    {
      return gasnet_mynode();
    }

#ifdef USE_RADOS
    RadosMemory::RadosMemory(Memory m, const std::string pool)
      : MemoryImpl(m, 0, MKIND_RADOS, 256, Memory::RADOS_MEM)
    {
      std::cout << "Connecting to RADOS..." << std::endl;
      std::flush(std::cout);
      cluster.init(NULL);
      // FIXME: how to inject env vars through gasnet?
      cluster.conf_read_file(NULL);
      cluster.conf_read_file("/home/nwatkins/ceph/src/ceph.conf");
      cluster.conf_read_file("/users/nwatkins/ceph/src/ceph.conf");
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

        ceph::bufferlist bl;
        int ret = ioctx.omap_get_header(ss.str(), &bl);
        assert(ret == 0);

        LegionRuntime::HighLevel::RadosMetadata obj_md;
        bl.copy(0, sizeof(obj_md), (char*)&obj_md);

        assert(obj_md.ndim == rinst->ndims);
        switch (obj_md.ndim) {
          case 2:
            assert(obj_md.min_bounds[0] == rinst->lo[0]);
            assert(obj_md.min_bounds[1] == rinst->lo[1]);
            break;
          case 3:
            assert(obj_md.min_bounds[0] == rinst->lo[0]);
            assert(obj_md.min_bounds[1] == rinst->lo[1]);
            assert(obj_md.min_bounds[2] == rinst->lo[2]);
            break;
          default:
            assert(0);
        }

        rinst->objmd.push_back(obj_md);
#if 0
        fprintf(stdout, "objname=%s, field_size=%u, ndim=%d, size=%d,%d shard=%d,%d..%d,%d lo=%d,%d\n",
            ss.str().c_str(),
            (unsigned)obj_md.field_size,
            obj_md.ndim,
            obj_md.size[0],
            obj_md.size[1],
            obj_md.min_bounds[0],
            obj_md.max_bounds[0],
            obj_md.min_bounds[1],
            obj_md.max_bounds[1],
            rinst->lo[0],
            rinst->lo[1]);
#endif
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
      assert(0); // broken

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
      assert(0); // broken

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

    void RadosMemory::read_array(librados::AioCompletion **pc,
        ceph::bufferlist *bl,
        const char *objname,
        int *offset, int *count, size_t nbytes, void *dst,
        size_t objoffset)
    {
      librados::AioCompletion *c = librados::Rados::aio_create_completion();
      int ret = ioctx.aio_read(objname, c, bl, nbytes, objoffset);
      assert(ret == 0);

      *pc = c;
#if 0
      std::stringstream key_ss;
      key_ss << offset[0] << "." << offset[1];
      std::set<std::string> keys;
      keys.insert(key_ss.str());

      librados::ObjectReadOperation op;
      op.omap_get_vals_by_keys(keys, pvals, pretval);

      librados::AioCompletion *c = librados::Rados::aio_create_completion();
      int ret = ioctx.aio_operate(objname, c, &op, NULL);
      assert(ret == 0);

      *pc = c;
#endif
    }

    /*
     * Dispatch an AIO write to the array.
     *
     * Note that in its current form this would have trouble with reading your
     * own writes and that sort of single threaded consistency as there is no
     * cache consistency in librados...
     */
    void RadosMemory::write_array(librados::AioCompletion **pc,
        const char *objname, int *offset, int *count,
        size_t nbytes, void *src, size_t objoffset)
    {
      ceph::bufferlist bl;
      bl.append((const char*)src, nbytes);

      librados::AioCompletion *c = librados::Rados::aio_create_completion();
      int ret = ioctx.aio_write(objname, c, bl, nbytes, objoffset);
      assert(ret == 0);

      *pc = c;
#if 0
      std::map<std::string, ceph::bufferlist> kvs;
      std::stringstream key_ss;
      key_ss << offset[0] << "." << offset[1];

      ceph::bufferlist bl;
      bl.append((const char*)src, nbytes);
      kvs[key_ss.str()] = bl;

      librados::ObjectWriteOperation op;
      op.omap_set(kvs);

      librados::AioCompletion *c = librados::Rados::aio_create_completion();
      int ret = ioctx.aio_operate(objname, c, &op);
      assert(ret == 0);

      *pc = c;
#endif
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
#endif

#ifdef USE_HDF
    HDFMemory::HDFMemory(Memory _me)
      : MemoryImpl(_me, 0 /*HDF doesn't have memory space*/, MKIND_HDF, ALIGNMENT, Memory::HDF_MEM)
    {
      int rc = 0;
      rc = pthread_rwlock_init(&rwlock, NULL);
      assert(rc==0); 
    }

    HDFMemory::~HDFMemory(void)
    {
      // close all HDF metadata
    }

    RegionInstance HDFMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     const Realm::ProfilingRequestSet &reqs,
                     RegionInstance parent_inst)
    {
      // we use a new create_instance, which could provide
      // more information for creating HDF metadata
      assert(0);
      return RegionInstance::NO_INST;
    }

    RegionInstance HDFMemory::create_instance(
                     IndexSpace is,
                     const int *linearization_bits,
                     size_t bytes_needed,
                     size_t block_size,
                     size_t element_size,
                     const std::vector<size_t>& field_sizes,
                     ReductionOpID redopid,
                     off_t list_size,
                     const Realm::ProfilingRequestSet &reqs,
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

      HDFMetadata* new_hdf = new HDFMetadata;
      new_hdf->hdf_memory = this;
      new_hdf->ndims = domain.get_dim();
      for (int i = 0; i < domain.get_dim(); i++) {
        new_hdf->lo[i] = domain.rect_data[i];
        new_hdf->dims[i] = domain.rect_data[i + domain.get_dim()] - domain.rect_data[i];
      }
      unsigned flags;
      if (read_only)
        flags = H5F_ACC_RDONLY;
      else
        flags = H5F_ACC_RDWR;
      pthread_rwlock_wrlock(&this->rwlock);
      new_hdf->file_id = H5Fopen(file, flags, H5P_DEFAULT);
      for (ID::IDType idx = 0; idx < path_names.size(); idx ++) {
        new_hdf->dataset_ids.push_back(H5Dopen2(new_hdf->file_id, path_names[idx], H5P_DEFAULT));
        new_hdf->datatype_ids.push_back(H5Dget_type(new_hdf->dataset_ids[idx]));
        pthread_rwlock_t rwlock_temp;
        new_hdf->dataset_rwlocks.push_back(rwlock_temp); 
        int rc = pthread_rwlock_init(&new_hdf->dataset_rwlocks[idx], NULL);
        assert(rc==0);
      }
      if (ID(inst).index_l() < hdf_metadata_vec.size()) {
        hdf_metadata_vec[ID(inst).index_l()] = new_hdf;
      } else {
        hdf_metadata_vec.push_back(new_hdf);
      }
      pthread_rwlock_unlock(&this->rwlock);
      return inst;
    }

    void HDFMemory::destroy_instance(RegionInstance i,
				     bool local_destroy)
    {
      pthread_rwlock_wrlock(&this->rwlock);
      HDFMetadata* new_hdf = hdf_metadata_vec[ID(i).index_l()];
      assert(new_hdf->dataset_ids.size() == new_hdf->datatype_ids.size());
      for (size_t idx = 0; idx < new_hdf->dataset_ids.size(); idx++) {
        H5Dclose(new_hdf->dataset_ids[idx]);
        H5Tclose(new_hdf->datatype_ids[idx]);
      }
      H5Fclose(new_hdf->file_id);
      new_hdf->dataset_ids.clear();
      new_hdf->datatype_ids.clear();
      delete new_hdf;
      destroy_instance_local(i, local_destroy);
      pthread_rwlock_unlock(&this->rwlock);
     
    }

    off_t HDFMemory::alloc_bytes(size_t size)
    {
      // We don't have to actually allocate bytes
      // for HDF memory. So we do nothing in this
      // function
      return 0;
    }

    void HDFMemory::free_bytes(off_t offset, size_t size)
    {
      // We don't have to free bytes for HDF memory
    }

    void HDFMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      assert(0);
    }

    void HDFMemory::get_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, void *dst, size_t size)
    {
      pthread_rwlock_rdlock(&this->rwlock);
      HDFMetadata *metadata = hdf_metadata_vec[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(metadata->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      pthread_rwlock_unlock(&this->rwlock);
      pthread_rwlock_wrlock(&this->rwlock);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      pthread_rwlock_unlock(&this->rwlock);
      pthread_rwlock_rdlock(&metadata->dataset_rwlocks[fid]);
//      std::cout << "in HDFMemory::get_bytes reading dataset_id: " << metadata->dataset_ids[fid] << std::endl;
      H5Dread(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, dst);
      pthread_rwlock_unlock(&metadata->dataset_rwlocks[fid]);
      pthread_rwlock_wrlock(&this->rwlock);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
      pthread_rwlock_unlock(&this->rwlock);
    }

    void HDFMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      assert(0);
    }

    void HDFMemory::put_bytes(ID::IDType inst_id, const DomainPoint& dp, int fid, const void *src, size_t size)
    {
      pthread_rwlock_rdlock(&this->rwlock);
      HDFMetadata *metadata = hdf_metadata_vec[inst_id];
      // use index to compute position in space
      assert(size == H5Tget_size(metadata->datatype_ids[fid]));
      hsize_t offset[3], count[3];
      for (int i = 0; i < metadata->ndims; i++) {
        offset[i] = dp.point_data[i] - metadata->lo[i];
      }
      count[0] = count[1] = count[2] = 1;
      hid_t dataspace_id = H5Dget_space(metadata->dataset_ids[fid]);
      H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);
      pthread_rwlock_unlock(&this->rwlock);
      pthread_rwlock_wrlock(&this->rwlock);
      hid_t memspace_id = H5Screate_simple(metadata->ndims, count, NULL);
      pthread_rwlock_unlock(&this->rwlock);
      pthread_rwlock_wrlock(&metadata->dataset_rwlocks[fid]);
//      std::cout << "in HDFMemory::put_bytes writing dataset_id: " << metadata->dataset_ids[fid] << std::endl;
      
      H5Dwrite(metadata->dataset_ids[fid], metadata->datatype_ids[fid], memspace_id, dataspace_id, H5P_DEFAULT, src);
      pthread_rwlock_unlock(&metadata->dataset_rwlocks[fid]);
      pthread_rwlock_wrlock(&this->rwlock);
      H5Sclose(dataspace_id);
      H5Sclose(memspace_id);
      pthread_rwlock_unlock(&this->rwlock);

    }

    void HDFMemory::apply_reduction_list(off_t offset, const ReductionOpUntyped *redop,
                        size_t count, const void *entry_buffer)
    {
    }

    void *HDFMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return 0; // cannot provide a pointer for it.
    }

    int HDFMemory::get_home_node(off_t offset, size_t size)
    {
      return gasnet_mynode();
    }
#endif
}


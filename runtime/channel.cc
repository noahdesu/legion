/* Copyright 2016 Stanford University
 * Copyright 2016 Los Alamos National Laboratory
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

#include "channel.h"

namespace LegionRuntime {
  namespace LowLevel {
    Logger::Category log_new_dma("new_dma");
    Logger::Category log_request("request");
#ifdef USE_DISK 
    inline int io_setup(unsigned nr, aio_context_t *ctxp)
      {
        return syscall(__NR_io_setup, nr, ctxp);
      }

      inline int io_destroy(aio_context_t ctx)
      {
        return syscall(__NR_io_destroy, ctx);
      }

      inline int io_submit(aio_context_t ctx, long nr, struct iocb **iocbpp)
      {
        return syscall(__NR_io_submit, ctx, nr, iocbpp);
      }

      inline int io_getevents(aio_context_t ctx, long min_nr, long max_nr,
                              struct io_event *events, struct timespec *timeout)
      {
        return syscall(__NR_io_getevents, ctx, min_nr, max_nr, events, timeout);
      }
#endif /*USE_DISK*/

      // TODO: currently we use dma_all_gpus to track the set of GPU* created
#ifdef USE_CUDA
      std::vector<GPU*> dma_all_gpus;
#endif
      // we use a single queue for all xferDes
      static XferDesQueue *xferDes_queue = 0;

      // we use a single manager to organize all channels
      static ChannelManager *channel_manager = 0;

      static inline off_t max(off_t a, off_t b) { return (a < b) ? b : a; }
      static inline size_t umin(size_t a, size_t b) { return (a < b) ? a : b; }

      static inline bool cross_ib(off_t start, size_t nbytes, size_t buf_size)
      {
        return (nbytes > 0) && (start / buf_size < (start + nbytes - 1) / buf_size);
      }

      void XferDes::mark_completed() {
        // notify owning DmaRequest upon completion of this XferDes
        //printf("complete XD = %lu\n", guid);
        if (launch_node == gasnet_mynode()) {
          complete_fence->mark_finished(true/*successful*/);
        } else {
          NotifyXferDesCompleteMessage::send_request(launch_node, complete_fence);
        }
      }

      static inline off_t calc_mem_loc_ib(off_t alloc_offset,
                                          off_t field_start,
                                          int field_size,
                                          size_t elmt_size,
                                          size_t block_size,
                                          size_t buf_size,
                                          size_t domain_size,
                                          off_t index)
      {
        off_t idx2 = domain_size / block_size * block_size;
        off_t offset;
        if (index < idx2) {
          offset = calc_mem_loc(alloc_offset, field_start, field_size, elmt_size, block_size, index);
        } else {
          offset = (alloc_offset + field_start * domain_size + (elmt_size - field_start) * idx2 + (index - idx2) * field_size);
        }
        return offset % buf_size;
      }

#ifdef SIMPLE_GET_MASK_REQUEST
      bool XferDes::simple_get_mask_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                                            MaskEnumerator* me,
                                            unsigned &offset_idx, coord_t available_slots)
      {
        assert((size_t) offset_idx < oas_vec.size());
        assert(me->any_left());
        nbytes = 0;
        coord_t src_idx, dst_idx;
        // cannot exceed the max_req_size
        coord_t todo = min(max_req_size / oas_vec[offset_idx].size, me->continuous_steps(src_idx, dst_idx));
        coord_t src_in_block = src_buf.block_size - src_idx % src_buf.block_size;
        coord_t dst_in_block = dst_buf.block_size - dst_idx % dst_buf.block_size;
        todo = min(todo, min(src_in_block, dst_in_block));
        bool scatter_src_ib = false, scatter_dst_ib = false;
        // make sure we have source data ready
        if (src_buf.is_ib) {
          src_start = calc_mem_loc_ib(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                      src_buf.elmt_size, src_buf.block_size, src_buf.buf_size, domain.get_volume(), src_idx);
          todo = min(todo, max(0, pre_bytes_write - src_start) / oas_vec[offset_idx].size);
          scatter_src_ib = scatter_ib(src_start, todo * oas_vec[offset_idx].size, src_buf.buf_size);
        } else {
          src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                   src_buf.elmt_size, src_buf.block_size, src_idx);
        }
        // make sure there are enough space in destination
        if (dst_buf.is_ib) {
          dst_start = calc_mem_loc_ib(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                      dst_buf.elmt_size, dst_buf.block_size, dst_buf.buf_size, domain.get_volume(), dst_idx);
          todo = min(todo, max(0, next_bytes_read + dst_buf.buf_size - dst_start) / oas_vec[offset_idx].size);
          scatter_dst_ib = scatter_ib(dst_start, todo * oas_vec[offset_idx].size, dst_buf.buf_size);
        } else {
          dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                   dst_buf.elmt_size, dst_buf.block_size, dst_idx);
        }
        if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
        ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
          return false; // case we don't have enough slots

        nbytes = todo * oas_vec[offset_idx].size;
        me->move(todo);
        if (!me->any_left()) {
          me->reset();
          offset_idx ++;
        }
        return true;
      }
#endif
#define MAX_GEN_REQS 3

      bool support_2d_xfers(XferDes::XferKind kind)
      {
        return (kind == XferDes::XFER_GPU_TO_FB)
               || (kind == XferDes::XFER_GPU_FROM_FB)
               || (kind == XferDes::XFER_GPU_IN_FB)
               || (kind == XferDes::XFER_GPU_PEER_FB)
               || (kind == XferDes::XFER_REMOTE_WRITE)
               || (kind == XferDes::XFER_MEM_CPY);
      }
      /* void print_request_info(Request* req)
      {
        printf("request(%dD): src_off(%zd) dst_off(%zd) src_str(%zd)"
               " dst_str(%zd) nbytes(%zu) nlines(%zu)",
               req->dim, req->src_off, req->dst_off, req->src_str,
               req->dst_str, req->nbytes, req->nlines);
      }*/

      template<unsigned DIM>
      long XferDes::default_get_requests(Request** reqs, long nr)
      {
        long idx = 0;
        coord_t src_idx, dst_idx, todo, src_str, dst_str;
        size_t nitems, nlines;
        while (idx + MAX_GEN_REQS <= nr && offset_idx < oas_vec.size()
        && MAX_GEN_REQS <= available_reqs.size()) {
          if (DIM == 0) {
            todo = min(max_req_size / oas_vec[offset_idx].size,
                       me->continuous_steps(src_idx, dst_idx));
            nitems = src_str = dst_str = todo;
            nlines = 1;
          }
          else
            todo = min(max_req_size / oas_vec[offset_idx].size,
                       li->continuous_steps(src_idx, dst_idx,
                                            src_str, dst_str,
                                            nitems, nlines));
          coord_t src_in_block = src_buf.block_size
                               - src_idx % src_buf.block_size;
          coord_t dst_in_block = dst_buf.block_size
                               - dst_idx % dst_buf.block_size;
          todo = min(todo, min(src_in_block, dst_in_block));
          coord_t src_start, dst_start;
          if (src_buf.is_ib) {
            src_start = calc_mem_loc_ib(0,
                                        oas_vec[offset_idx].src_offset,
                                        oas_vec[offset_idx].size,
                                        src_buf.elmt_size,
                                        src_buf.block_size,
                                        src_buf.buf_size,
                                        domain.get_volume(), src_idx);
            todo = min(todo, max(0, pre_bytes_write - src_start)
                                    / oas_vec[offset_idx].size);
          } else {
            src_start = calc_mem_loc(0,
                                     oas_vec[offset_idx].src_offset,
                                     oas_vec[offset_idx].size,
                                     src_buf.elmt_size,
                                     src_buf.block_size, src_idx);
          }
          if (dst_buf.is_ib) {
            dst_start = calc_mem_loc_ib(0,
                                        oas_vec[offset_idx].dst_offset,
                                        oas_vec[offset_idx].size,
                                        dst_buf.elmt_size,
                                        dst_buf.block_size,
                                        dst_buf.buf_size,
                                        domain.get_volume(), dst_idx);
            todo = min(todo, max(0, next_bytes_read + dst_buf.buf_size - dst_start)
                                    / oas_vec[offset_idx].size);
          } else {
            dst_start = calc_mem_loc(0,
                                     oas_vec[offset_idx].dst_offset,
                                     oas_vec[offset_idx].size,
                                     dst_buf.elmt_size,
                                     dst_buf.block_size, dst_idx);
          }
          bool cross_src_ib = false, cross_dst_ib = false;
          if (src_buf.is_ib)
            cross_src_ib = cross_ib(src_start,
                                    todo * oas_vec[offset_idx].size,
                                    src_buf.buf_size);
          if (dst_buf.is_ib)
            cross_dst_ib = cross_ib(dst_start,
                                    todo * oas_vec[offset_idx].size,
                                    dst_buf.buf_size);
          // We are crossing ib, fallback to 1d case
          // We don't support 2D, fallback to 1d case
          if (cross_src_ib || cross_dst_ib || !support_2d_xfers(kind))
            todo = min(todo, nitems);
          if ((size_t)todo <= nitems) {
            // fallback to 1d case
            nitems = (size_t)todo;
            nlines = 1;
          } else {
            nlines = todo / nitems;
            todo = nlines * nitems;
          }
          if (nlines == 1) {
            // 1D case
            size_t nbytes = todo * oas_vec[offset_idx].size;
            while (nbytes > 0) {
              size_t req_size = nbytes;
              Request* new_req = dequeue_request();
              new_req->dim = Request::DIM_1D;
              new_req->src_off = src_start;
              new_req->dst_off = dst_start;
              if (src_buf.is_ib)
                req_size = umin(req_size, src_buf.buf_size - src_start);
              if (dst_buf.is_ib)
                req_size = umin(req_size, dst_buf.buf_size - dst_start);
              new_req->nbytes = req_size;
              new_req->nlines = 1;
              reqs[idx++] = new_req;
              nbytes -= req_size;
            }
          } else {
            // 2D case
            Request* new_req = dequeue_request();
            new_req->dim = Request::DIM_2D;
            new_req->src_off = src_start;
            new_req->dst_off = dst_start;
            new_req->src_str = src_str * oas_vec[offset_idx].size;
            new_req->dst_str = dst_str * oas_vec[offset_idx].size;
            new_req->nbytes = nitems * oas_vec[offset_idx].size;
            new_req->nlines = nlines;
            reqs[idx++] = new_req;
          }
          if (DIM == 0) {
            me->move(todo);
            if (!me->any_left()) {
              me->reset();
              offset_idx ++;
            }
          } else {
            li->move(todo);
            if (!li->any_left()) {
              li->reset();
              offset_idx ++;
            }
          }
        } // while
        return idx;
      }

#ifdef SIMPLE_GET_REQUEST
      template<unsigned DIM>
      bool XferDes::simple_get_request(off_t &src_start, off_t &dst_start, size_t &nbytes,
                              Layouts::GenericLayoutIterator<DIM>* li,
                              unsigned &offset_idx, coord_t available_slots)
      {
        assert(offset_idx < oas_vec.size());
        assert(li->any_left());
        nbytes = 0;
        coord_t src_idx, dst_idx;
        // cannot exceed the max_req_size
        coord_t todo = min(max_req_size / oas_vec[offset_idx].size, li->continuous_steps(src_idx, dst_idx));
        coord_t src_in_block = src_buf.block_size - src_idx % src_buf.block_size;
        coord_t dst_in_block = dst_buf.block_size - dst_idx % dst_buf.block_size;
        todo = min(todo, min(src_in_block, dst_in_block));
        if (todo == 0)
          return false;
        bool scatter_src_ib = false, scatter_dst_ib = false;
        // make sure we have source data ready
        if (src_buf.is_ib) {
          src_start = calc_mem_loc_ib(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                      src_buf.elmt_size, src_buf.block_size, domain.get_volume(), src_idx);
          todo = min(todo, max(0, pre_bytes_write - src_start) / oas_vec[offset_idx].size);
          scatter_src_ib = scatter_ib(src_start, todo * oas_vec[offset_idx].size, src_buf.buf_size);
        } else {
	  src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                   src_buf.elmt_size, src_buf.block_size, src_idx);
        }
        if (todo  == 0)
          return false;
        // make sure there are enough space in destination
        if (dst_buf.is_ib) {
          dst_start = calc_mem_loc_ib(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                      dst_buf.elmt_size, dst_buf.block_size, domain.get_volume(), dst_idx);
          todo = min(todo, max(0, next_bytes_read + dst_buf.buf_size - dst_start) / oas_vec[offset_idx].size);
          scatter_dst_ib = scatter_ib(dst_start, todo * oas_vec[offset_idx].size, dst_buf.buf_size);
        } else {
          dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                   dst_buf.elmt_size, dst_buf.block_size, dst_idx);
        }
        if (todo == 0)
          return false;
        if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
        ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
          return false; // case we don't have enough slots

        nbytes = todo * oas_vec[offset_idx].size;
        log_request.info("[1D] guid(%llx) src_start(%zd) dst_start(%zd)"
                         " todo(%lld) offset_idx(%u) nbytes(%zu)",
                         guid, src_start, dst_start, todo, offset_idx, nbytes);
        li->move(todo);
        if (!li->any_left()) {
          li->reset();
          offset_idx ++;
        }
        return true;
      }

      template<unsigned DIM>
      bool XferDes::simple_get_request_2d(off_t &src_start,
                                          off_t &dst_start,
                                          off_t &src_stride_bytes,
                                          off_t &dst_stride_bytes,
                                          size_t &nbytes_per_line,
                                          size_t &nlines,
                                  Layouts::GenericLayoutIterator<DIM>* li,
                                          unsigned &offset_idx,
                                          coord_t available_slots)
      {
        assert(offset_idx < oas_vec.size());
        assert(li->any_left());
        nbytes_per_line = 0;
        coord_t src_idx, dst_idx;
        coord_t src_stride, dst_stride;
        size_t nitems_per_line;
        // cannot exceed the max_req_size
        coord_t todo = min(max_req_size / oas_vec[offset_idx].size,
                           li->continuous_steps(src_idx, dst_idx,
                                                src_stride, dst_stride,
                                                nitems_per_line, nlines));
        coord_t src_in_block = src_buf.block_size - src_idx % src_buf.block_size;
        coord_t dst_in_block = dst_buf.block_size - dst_idx % dst_buf.block_size;
        todo = min(todo, min(src_in_block, dst_in_block));
        if (todo == 0)
          return false;
        bool scatter_src_ib = false, scatter_dst_ib = false;
        // make sure we have source data ready
        // 2d doesn't support scattering ib
        if (src_buf.is_ib) {
          src_start = calc_mem_loc_ib(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                      src_buf.elmt_size, src_buf.block_size, domain.get_volume(), src_idx);
          todo = min(todo, max(0, pre_bytes_write - src_start) / oas_vec[offset_idx].size);
          scatter_src_ib = scatter_ib(src_start, todo * oas_vec[offset_idx].size, src_buf.buf_size);
        } else {
	  src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                   src_buf.elmt_size, src_buf.block_size, src_idx);
        }
        if (todo  == 0)
          return false;
        // 2d doesn't support scattering ib
        if (dst_buf.is_ib) {
          dst_start = calc_mem_loc_ib(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                      dst_buf.elmt_size, dst_buf.block_size, domain.get_volume(), dst_idx);
          todo = min(todo, max(0, next_bytes_read + dst_buf.buf_size - dst_start) / oas_vec[offset_idx].size);
          scatter_dst_ib = scatter_ib(dst_start, todo * oas_vec[offset_idx].size, dst_buf.buf_size);
        } else {
          dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                   dst_buf.elmt_size, dst_buf.block_size, dst_idx);
        }
        if (todo == 0)
          return false;
        if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
        ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
          return false;

        if (scatter_src_ib || scatter_dst_ib) {
          // We are scattering ib, fallback to 1d case
          todo = min(todo, nitems_per_line);
        }
        if ((size_t)todo <= nitems_per_line) {
          // fallback to 1d case
          nitems_per_line = (size_t)todo;
          nlines = 1;
        } else {
          nlines = todo / nitems_per_line;
        }
        todo = nitems_per_line * nlines;
        nbytes_per_line = nitems_per_line * oas_vec[offset_idx].size;
        src_stride_bytes = src_stride * oas_vec[offset_idx].size;
        dst_stride_bytes = dst_stride * oas_vec[offset_idx].size;
        log_request.info("[2D] guid(%llx) src_start(%zd) dst_start(%zd)"
                         "src_str(%zd) dst_str(%zd)"
                         " nbytes(%zu) nlines(%zu) offset_idx(%u)",
                         guid, src_start, dst_start, src_stride_bytes,
                         dst_stride_bytes, nbytes_per_line, nlines, offset_idx);

        li->move(todo);
        if (!li->any_left()) {
          li->reset();
          offset_idx ++;
        }
        return true;
      }
#endif

#ifdef DEADCODE_OLD_GENERIC_ITERATOR
      template<unsigned DIM>
      bool XferDes::simple_get_request(
                    off_t &src_start, off_t &dst_start, size_t &nbytes,
                    Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dsi,
                    Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >* &dso,
                    Rect<1> &irect, Rect<1> &orect,
                    int &done, int &offset_idx, int &block_start, int &total, int available_slots,
                    bool disable_batch)
      {
        src_start = calc_mem_loc(0, oas_vec[offset_idx].src_offset, oas_vec[offset_idx].size,
                                 src_buf.elmt_size, src_buf.block_size, done + irect.lo);
        dst_start = calc_mem_loc(0, oas_vec[offset_idx].dst_offset, oas_vec[offset_idx].size,
                                 dst_buf.elmt_size, dst_buf.block_size, done + orect.lo);
        nbytes = 0;
        bool scatter_src_ib = false, scatter_dst_ib = false;
        while (true) {
          // check to see if we can generate next request
          int src_in_block = src_buf.block_size - (done + irect.lo) % src_buf.block_size;
          int dst_in_block = dst_buf.block_size - (done + orect.lo) % dst_buf.block_size;
          int todo = min((max_req_size - nbytes) / oas_vec[offset_idx].size, min(total - done, min(src_in_block, dst_in_block)));
          // make sure we have source data ready
          if (src_buf.is_ib) {
            todo = min(todo, max(0, src_buf.alloc_offset + pre_bytes_write - (src_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_src_ib = scatter_src_ib || scatter_ib(src_start, nbytes + todo * oas_vec[offset_idx].size, src_buf.buf_size);
          }
          // make sure there are enough space in destination
          if (dst_buf.is_ib) {
            todo = min(todo, max(0, dst_buf.alloc_offset + next_bytes_read + dst_buf.buf_size - (dst_start + nbytes)) / oas_vec[offset_idx].size);
            scatter_dst_ib = scatter_dst_ib || scatter_ib(dst_start, nbytes + todo * oas_vec[offset_idx].size, dst_buf.buf_size);
          }
          if((scatter_src_ib && scatter_dst_ib && available_slots < 3)
          ||((scatter_src_ib || scatter_dst_ib) && available_slots < 2))
            break;
          //printf("min(%d, %d, %d) \n =", (int)(max_req_size - nbytes) / oas_vec[offset_idx].size, total - done, min(src_in_block, dst_in_block));
          //printf("todo = %d, size = %d\n", todo, oas_vec[offset_idx].size);
          nbytes += todo * oas_vec[offset_idx].size;
          // see if we can batch more
          if (!disable_batch && todo == src_in_block && todo == dst_in_block && offset_idx + 1 < oas_vec.size()
          && src_buf.block_size == dst_buf.block_size && todo + done >= src_buf.block_size
          && oas_vec[offset_idx + 1].src_offset == oas_vec[offset_idx].src_offset + oas_vec[offset_idx].size
          && oas_vec[offset_idx + 1].dst_offset == oas_vec[offset_idx].dst_offset + oas_vec[offset_idx].size) {
            done = block_start;
            offset_idx += 1;
          }
          else {
            done += todo;
            break;
          }
        }

        if (nbytes > 0 &&
        (((done + irect.lo) % src_buf.block_size == 0 && done + irect.lo > block_start && order == XferOrder::SRC_FIFO)
        ||((done + orect.lo) % dst_buf.block_size == 0 && done + orect.lo > block_start && order == XferOrder::DST_FIFO)
        || (done == total))) {
          offset_idx ++;
          if (offset_idx < oas_vec.size()) {
            switch (order) {
              case XferOrder::SRC_FIFO:
                done = block_start - irect.lo;
                break;
              case XferOrder::DST_FIFO:
                done = block_start - orect.lo;
                break;
              case XferOrder::ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
          else {
            int new_block_start;
            switch (order) {
              case XferOrder::SRC_FIFO:
                new_block_start = block_start + src_buf.block_size;
                new_block_start = new_block_start - new_block_start % src_buf.block_size;
                block_start = new_block_start;
                done = block_start - irect.lo;
                offset_idx = 0;
                if (block_start > irect.hi) {
                  dso->step();
                  if (dso->any_left) {
                    dsi->step();
                    if (dsi->any_left) {
                      delete dso;
                      dso = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dsi->subrect, *(dst_buf.linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    irect = src_buf.linearization.get_mapping<DIM>()->image_dense_subrect(dso->subrect, subrect_check);
                    orect = dso->image;
                    done = 0; offset_idx = 0; block_start = irect.lo; total = irect.hi[0] - irect.lo[0] + 1;
                  }
                }
                break;
              case XferOrder::DST_FIFO:
                new_block_start = block_start + dst_buf.block_size;
                new_block_start = new_block_start - new_block_start % dst_buf.block_size;
                block_start = new_block_start;
                done = block_start - orect.lo;
                offset_idx = 0;
                if (block_start > orect.hi) {
                  dsi->step();
                  if (!dsi->any_left) {
                    dso->step();
                    if (dso->any_left) {
                      delete dsi;
                      dsi = new Arrays::GenericDenseSubrectIterator<Arrays::Mapping<DIM, 1> >(dso->subrect, *(src_buf.linearization.get_mapping<DIM>()));
                    }
                  }
                  if (dso->any_left && dsi->any_left) {
                    Rect<DIM> subrect_check;
                    orect = dst_buf.linearization.get_mapping<DIM>()->image_dense_subrect(dsi->subrect, subrect_check);
                    irect = dsi->image;
                    done = 0; offset_idx = 0; block_start = orect.lo; total = orect.hi[0] - orect.lo[0] + 1;
                  }
                }
                break;
              case XferOrder::ANY_ORDER:
                assert(0);
                break;
              default:
                assert(0);
            }
          }
        }
        return (nbytes > 0);
      }
#endif

      inline void XferDes::simple_update_bytes_read(int64_t offset, uint64_t size)
      {
        //printf("update_read[%lx]: offset = %ld, size = %lu, pre = %lx, next = %lx\n", guid, offset, size, pre_xd_guid, next_xd_guid);
        if (pre_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_read % src_buf.buf_size) == offset) {
            bytes_read += size;
            update = true;
          }
          else {
            //printf("[%lx] insert: key = %ld, value = %lu\n", guid, offset, size);
            segments_read[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_read.find(bytes_read % src_buf.buf_size);
            if (it == segments_read.end())
              break;
            bytes_read += it->second;
            update = true;
            //printf("[%lx] erase: key = %ld, value = %lu\n", guid, it->first, it->second);
            segments_read.erase(it);
          }
          if (update) {
            xferDes_queue->update_next_bytes_read(pre_xd_guid, bytes_read);
          }
        }
        else {
          bytes_read += size;
        }
      }

      inline void XferDes::simple_update_bytes_write(int64_t offset, uint64_t size)
      {
        log_request.info(
            "update_write: guid(%llx) off(%" PRId64") size(%" PRIu64") "
            "pre(%llx) next(%llx)", guid, offset, size, pre_xd_guid, next_xd_guid);
        if (next_xd_guid != XFERDES_NO_GUID) {
          bool update = false;
          if ((int64_t)(bytes_write % dst_buf.buf_size) == offset) {
            bytes_write += size;
            update = true;
          } else {
            segments_write[offset] = size;
          }
          std::map<int64_t, uint64_t>::iterator it;
          while (true) {
            it = segments_write.find(bytes_write % dst_buf.buf_size);
            if (it == segments_write.end())
              break;
            bytes_write += it->second;
            update = true;
            segments_write.erase(it);
          }
          if (update) {
            xferDes_queue->update_pre_bytes_write(next_xd_guid, bytes_write);
          }
        }
        else {
          bytes_write += size;
        }
        //printf("[%d] offset(%ld), size(%lu), bytes_writes(%lx): %ld\n", gasnet_mynode(), offset, size, guid, bytes_write);
      }

      void XferDes::default_notify_request_read_done(Request* req)
      {  
        req->is_read_done = true;
        if (req->dim == Request::DIM_1D)
          simple_update_bytes_read(req->src_off, req->nbytes);
        else
          simple_update_bytes_read(req->src_off, req->nbytes * req->nlines);
      }

      void XferDes::default_notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        if (req->dim == Request::DIM_1D)
          simple_update_bytes_write(req->dst_off, req->nbytes);
        else
          simple_update_bytes_write(req->dst_off, req->nbytes * req->nlines);
        enqueue_request(req);
      }

      template<unsigned DIM>
      MemcpyXferDes<DIM>::MemcpyXferDes(DmaRequest* _dma_request,
                                        gasnet_node_t _launch_node,
                                        XferDesID _guid,
                                        XferDesID _pre_xd_guid,
                                        XferDesID _next_xd_guid,
                                        bool mark_started,
                                        const Buffer& _src_buf,
                                        const Buffer& _dst_buf,
                                        const Domain& _domain,
                                        const std::vector<OffsetsAndSize>& _oas_vec,
                                        uint64_t _max_req_size,
                                        long max_nr,
                                        int _priority,
                                        XferOrder::Type _order,
                                        XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                  _next_xd_guid, mark_started, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order,
                  XferDes::XFER_MEM_CPY, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        channel = channel_manager->get_memcpy_channel();
        src_buf_base = (char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
        dst_buf_base = (char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
        memcpy_reqs = (MemcpyRequest*) calloc(max_nr, sizeof(MemcpyRequest));
        for (int i = 0; i < max_nr; i++) {
          memcpy_reqs[i].xd = this;
          enqueue_request(&memcpy_reqs[i]);
        }
      }

      template<unsigned DIM>
      long MemcpyXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        MemcpyRequest** reqs = (MemcpyRequest**) requests;
        long new_nr = default_get_requests<DIM>(requests, nr);
        for (long i = 0; i < new_nr; i++)
        {
          reqs[i]->src_base = (char*)(src_buf_base + reqs[i]->src_off);
          reqs[i]->dst_base = (char*)(dst_buf_base + reqs[i]->dst_off);
        }
        return new_nr;

#ifdef TO_BE_DELETE
        long idx = 0;
        while (idx < nr && !available_reqs.empty() && offset_idx < oas_vec.size()) {
          off_t src_start, dst_start;
          size_t nbytes;
          if (DIM == 0) {
            simple_get_mask_request(src_start, dst_start, nbytes, me, offset_idx, min(available_reqs.size(), nr - idx));
          } else {
            simple_get_request<DIM>(src_start, dst_start, nbytes, li, offset_idx, min(available_reqs.size(), nr - idx));
          }
          if (nbytes == 0)
            break;
          //printf("[MemcpyXferDes] guid = %lx, offset_idx = %lld, oas_vec.size() = %lu, nbytes = %lu\n", guid, offset_idx, oas_vec.size(), nbytes);
          while (nbytes > 0) {
            size_t req_size = nbytes;
            if (src_buf.is_ib) {
              src_start = src_start % src_buf.buf_size;
              req_size = umin(req_size, src_buf.buf_size - src_start);
            }
            if (dst_buf.is_ib) {
              dst_start = dst_start % dst_buf.buf_size;
              req_size = umin(req_size, dst_buf.buf_size - dst_start);
            }
            mem_cpy_reqs[idx] = (MemcpyRequest*) available_reqs.front();
            available_reqs.pop();
            //printf("[MemcpyXferDes] src_start = %ld, dst_start = %ld, nbytes = %lu\n", src_start, dst_start, nbytes);
            mem_cpy_reqs[idx]->is_read_done = false;
            mem_cpy_reqs[idx]->is_write_done = false;
            mem_cpy_reqs[idx]->src_buf = (char*)(src_buf_base + src_start);
            mem_cpy_reqs[idx]->dst_buf = (char*)(dst_buf_base + dst_start);
            mem_cpy_reqs[idx]->nbytes = req_size;
            src_start += req_size; // here we don't have to mod src_buf.buf_size since it will be performed in next loop
            dst_start += req_size; //
            nbytes -= req_size;
            idx++;
          }
        }
        return idx;
#endif
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      template<unsigned DIM>
      void MemcpyXferDes<DIM>::flush()
      {
      }

      template<unsigned DIM>
      GASNetXferDes<DIM>::GASNetXferDes(DmaRequest* _dma_request,
                                        gasnet_node_t _launch_node,
                                        XferDesID _guid,
                                        XferDesID _pre_xd_guid,
                                        XferDesID _next_xd_guid,
                                        bool mark_started,
                                        const Buffer& _src_buf,
                                        const Buffer& _dst_buf,
                                        const Domain& _domain,
                                        const std::vector<OffsetsAndSize>& _oas_vec,
                                        uint64_t _max_req_size,
                                        long max_nr,
                                        int _priority,
                                        XferOrder::Type _order,
                                        XferKind _kind,
                                        XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                  _next_xd_guid, mark_started, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order,
                  _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        gasnet_reqs = (GASNetRequest*) calloc(max_nr, sizeof(GASNetRequest));
        for (int i = 0; i < max_nr; i++) {
          gasnet_reqs[i].xd = this;
          enqueue_request(&gasnet_reqs[i]);
        }
        switch (kind) {
          case XferDes::XFER_GASNET_READ:
          {
            channel = channel_manager->get_gasnet_read_channel();
            buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            break;
          }
          case XferDes::XFER_GASNET_WRITE:
          {
            channel = channel_manager->get_gasnet_write_channel();
            buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            break;
          }
          default:
            assert(false);
        }
      }

      template<unsigned DIM>
      long GASNetXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        GASNetRequest** reqs = (GASNetRequest**) requests;
        long new_nr = default_get_requests<DIM>(requests, nr);
        switch (kind) {
          case XferDes::XFER_GASNET_READ:
          {
            for (long i = 0; i < new_nr; i++) {
              reqs[i]->gas_off = src_buf.alloc_offset + reqs[i]->src_off;
              reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
            }
            break;
          }
          case XferDes::XFER_GASNET_WRITE:
          {
            for (long i = 0; i < new_nr; i++) {
              reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
              reqs[i]->gas_off = dst_buf.alloc_offset + reqs[i]->dst_off;
            }
            break;
          }
          default:
            assert(0);
        }
        return new_nr;
      }

      template<unsigned DIM>
      void GASNetXferDes<DIM>::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      template<unsigned DIM>
      void GASNetXferDes<DIM>::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      template<unsigned DIM>
      void GASNetXferDes<DIM>::flush()
      {
      }

      template<unsigned DIM>
      RemoteWriteXferDes<DIM>::RemoteWriteXferDes(DmaRequest* _dma_request,
                                                  gasnet_node_t _launch_node,
                                                  XferDesID _guid,
                                                  XferDesID _pre_xd_guid,
                                                  XferDesID _next_xd_guid,
                                                  bool mark_started,
                                                  const Buffer& _src_buf,
                                                  const Buffer& _dst_buf,
                                                  const Domain& _domain,
                                                  const std::vector<OffsetsAndSize>& _oas_vec,
                                                  uint64_t _max_req_size,
                                                  long max_nr,
                                                  int _priority,
                                                  XferOrder::Type _order,
                                                  XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                  _next_xd_guid, mark_started, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority, _order,
                  XferDes::XFER_REMOTE_WRITE, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        // make sure dst buffer is registered memory
        assert(dst_mem_impl->kind == MemoryImpl::MKIND_RDMA);
        channel = channel_manager->get_remote_write_channel();
        src_buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
        // Note that we cannot use get_direct_ptr to get dst_buf_base, since it always returns 0
        dst_buf_base = ((const char*)((Realm::RemoteMemory*)dst_mem_impl)->regbase) + dst_buf.alloc_offset;
        requests = (RemoteWriteRequest*) calloc(max_nr, sizeof(RemoteWriteRequest));
        for (int i = 0; i < max_nr; i++) {
          requests[i].xd = this;
          requests[i].dst_node = ID(_dst_buf.memory).memory.owner_node;
          enqueue_request(&requests[i]);
        }
      }

      template<unsigned DIM>
      long RemoteWriteXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        pthread_mutex_lock(&xd_lock);
        RemoteWriteRequest** reqs = (RemoteWriteRequest**) requests;
        long new_nr = default_get_requests<DIM>(requests, nr);
        for (long i = 0; i < new_nr; i++)
        {
          reqs[i]->src_base = (char*)(src_buf_base + reqs[i]->src_off);
          reqs[i]->dst_base = (char*)(dst_buf_base + reqs[i]->dst_off);
        }
        pthread_mutex_unlock(&xd_lock);
        return new_nr;
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::notify_request_read_done(Request* req)
      {
        pthread_mutex_lock(&xd_lock);
        default_notify_request_read_done(req);
        pthread_mutex_unlock(&xd_lock);
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::notify_request_write_done(Request* req)
      {
        pthread_mutex_lock(&xd_lock);
        default_notify_request_write_done(req);
        pthread_mutex_unlock(&xd_lock);
      }

      template<unsigned DIM>
      void RemoteWriteXferDes<DIM>::flush()
      {
        pthread_mutex_lock(&xd_lock);
        pthread_mutex_unlock(&xd_lock);
      }

#ifdef USE_DISK
      template<unsigned DIM>
      DiskXferDes<DIM>::DiskXferDes(DmaRequest* _dma_request,
                                    gasnet_node_t _launch_node,
                                    XferDesID _guid,
                                    XferDesID _pre_xd_guid,
                                    XferDesID _next_xd_guid,
                                    bool mark_started,
                                    const Buffer& _src_buf,
                                    const Buffer& _dst_buf,
                                    const Domain& _domain,
                                    const std::vector<OffsetsAndSize>& _oas_vec,
                                    uint64_t _max_req_size,
                                    long max_nr,
                                    int _priority,
                                    XferOrder::Type _order,
                                    XferKind _kind,
                                    XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                  _next_xd_guid, mark_started, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority,
                  _order, _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        switch (kind) {
          case XferDes::XFER_DISK_READ:
          {
            channel = channel_manager->get_disk_read_channel();
            fd = ((Realm::DiskMemory*)src_mem_impl)->fd;
            buf_base = (const char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_DISK);
            break;
          }
          case XferDes::XFER_DISK_WRITE:
          {
            channel = channel_manager->get_disk_write_channel();
            fd = ((Realm::DiskMemory*)dst_mem_impl)->fd;
            buf_base = (const char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_DISK);
            break;
          }
          default:
            assert(0);
        }
        disk_reqs = (DiskRequest*) calloc(max_nr, sizeof(DiskRequest));
        for (int i = 0; i < max_nr; i++) {
          disk_reqs[i].xd = this;
          disk_reqs[i].fd = fd;
          enqueue_request(&disk_reqs[i]);
        }
      }

      template<unsigned DIM>
      long DiskXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        DiskRequest** reqs = (DiskRequest**) requests;
        long new_nr = default_get_requests<DIM>(requests, nr);
        switch (kind) {
          case XferDes::XFER_DISK_READ:
          {
            for (long i = 0; i < new_nr; i++) {
              reqs[i]->disk_off = src_buf.alloc_offset + reqs[i]->src_off;
              reqs[i]->mem_base = (char*)(buf_base + reqs[i]->dst_off);
            }
            break;
          }
          case XferDes::XFER_DISK_WRITE:
          {
            for (long i = 0; i < new_nr; i++) {
              reqs[i]->mem_base = (char*)(buf_base + reqs[i]->src_off);
              reqs[i]->disk_off = dst_buf.alloc_offset + reqs[i]->dst_off;
            }
            break;
          }
          default:
            assert(0);
        }
        return new_nr;
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      template<unsigned DIM>
      void DiskXferDes<DIM>::flush()
      {
        fsync(fd);
      }
#endif /*USE_DISK*/
#ifdef USE_CUDA
      template<unsigned DIM>
      GPUXferDes<DIM>::GPUXferDes(DmaRequest* _dma_request,
                                  gasnet_node_t _launch_node,
                                  XferDesID _guid,
                                  XferDesID _pre_xd_guid,
                                  XferDesID _next_xd_guid,
                                  bool mark_started,
                                  const Buffer& _src_buf,
                                  const Buffer& _dst_buf,
                                  const Domain& _domain,
                                  const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size,
                                  long max_nr,
                                  int _priority,
                                  XferOrder::Type _order,
                                  XferKind _kind,
                                  XferDesFence* _complete_fence)
      : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                _next_xd_guid, mark_started, _src_buf, _dst_buf,
                _domain, _oas_vec, _max_req_size, _priority,
                _order, _kind, _complete_fence)
      {
        MemoryImpl* src_mem_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_mem_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        //gpu_reqs = (GPURequest*) calloc(max_nr, sizeof(GPURequest));
        for (int i = 0; i < max_nr; i++) {
          GPURequest* gpu_req = new GPURequest;
          gpu_req->xd = this;
          enqueue_request(gpu_req);
        }
 
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
          {
            src_gpu = NULL;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;;
            channel = channel_manager->get_gpu_to_fb_channel(dst_gpu);
            src_buf_base = (char*) src_mem_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            dst_buf_base = NULL;
            assert(dst_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            break;
          }
          case XferDes::XFER_GPU_FROM_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = NULL;
            channel = channel_manager->get_gpu_from_fb_channel(src_gpu);
            src_buf_base = NULL;
            dst_buf_base = (char*) dst_mem_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            break;
          }
          case XferDes::XFER_GPU_IN_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;
            channel = channel_manager->get_gpu_in_fb_channel(src_gpu);
            src_buf_base = dst_buf_base = NULL;
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_gpu == dst_gpu);
            break;
          }
          case XferDes::XFER_GPU_PEER_FB:
          {
            src_gpu = ((GPUFBMemory*)src_mem_impl)->gpu;
            dst_gpu = ((GPUFBMemory*)dst_mem_impl)->gpu;
            channel = channel_manager->get_gpu_peer_fb_channel(src_gpu);
            src_buf_base = dst_buf_base = NULL;
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_mem_impl->kind == MemoryImpl::MKIND_GPUFB);
            assert(src_gpu != dst_gpu);
            break;
          }
          default:
            assert(0);
        }
      }

      template<unsigned DIM>
      long GPUXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        GPURequest** reqs = (GPURequest**) requests;
        long new_nr = default_get_requests<DIM>(requests, nr);
        for (long i = 0; i < new_nr; i++) {
          reqs[i]->event.reset();
          switch (kind) {
            case XferDes::XFER_GPU_TO_FB:
            {
              reqs[i]->src_base = src_buf_base + reqs[i]->src_off;
              reqs[i]->dst_gpu_off = dst_buf.alloc_offset + reqs[i]->dst_off;
              break;
            }
            case XferDes::XFER_GPU_FROM_FB:
            {
              reqs[i]->src_gpu_off = src_buf.alloc_offset + reqs[i]->src_off;
              reqs[i]->dst_base = dst_buf_base + reqs[i]->dst_off;
              break;
            }
            case XferDes::XFER_GPU_IN_FB:
            {
              reqs[i]->src_gpu_off = src_buf.alloc_offset + reqs[i]->src_off;
              reqs[i]->dst_gpu_off = dst_buf.alloc_offset + reqs[i]->dst_off;
              break;
            }
            case XferDes::XFER_GPU_PEER_FB:
            {
              reqs[i]->src_gpu_off = src_buf.alloc_offset + reqs[i]->src_off;
              reqs[i]->dst_gpu_off = dst_buf.alloc_offset + reqs[i]->dst_off;
              // also need to set dst_gpu for peer xfer
              reqs[i]->dst_gpu = dst_gpu;
              break;
            }
            default:
              assert(0);
          }
        }
        return new_nr;
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::notify_request_read_done(Request* req)
      {
        default_notify_request_read_done(req);
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::notify_request_write_done(Request* req)
      {
        default_notify_request_write_done(req);
      }

      template<unsigned DIM>
      void GPUXferDes<DIM>::flush()
      {
      }
#endif

#ifdef USE_HDF
      template<unsigned DIM>
      HDFXferDes<DIM>::HDFXferDes(DmaRequest* _dma_request,
                                  gasnet_node_t _launch_node,
                                  XferDesID _guid,
                                  XferDesID _pre_xd_guid,
                                  XferDesID _next_xd_guid,
                                  bool mark_started,
                                  RegionInstance inst,
                                  const Buffer& _src_buf,
                                  const Buffer& _dst_buf,
                                  const Domain& _domain,
                                  const std::vector<OffsetsAndSize>& _oas_vec,
                                  uint64_t _max_req_size,
                                  long max_nr,
                                  int _priority,
                                  XferOrder::Type _order,
                                  XferKind _kind,
                                  XferDesFence* _complete_fence)
        : XferDes(_dma_request, _launch_node, _guid, _pre_xd_guid,
                  _next_xd_guid, mark_started, _src_buf, _dst_buf,
                  _domain, _oas_vec, _max_req_size, _priority,
                  _order, _kind, _complete_fence)
      {
        MemoryImpl* src_impl = get_runtime()->get_memory_impl(_src_buf.memory);
        MemoryImpl* dst_impl = get_runtime()->get_memory_impl(_dst_buf.memory);
        // for now, we didn't consider HDF transfer for intermediate buffer
        // since ib may involve a different address space model
        assert(!src_buf.is_ib);
        assert(!dst_buf.is_ib);
        Rect<DIM> subrect_check;
        HDF5Memory* hdf_mem;
        switch (kind) {
          case XferDes::XFER_HDF_READ:
          {
            hdf_mem = (HDF5Memory*) get_runtime()->get_memory_impl(src_buf.memory);
            //pthread_rwlock_rdlock(&hdf_mem->rwlock);
            std::map<RegionInstance, HDFMetadata*>::iterator it;
            it = hdf_mem->hdf_metadata.find(inst);
            assert(it != hdf_mem->hdf_metadata.end());
            hdf_metadata = it->second;
            //pthread_rwlock_unlock(&hdf_mem->rwlock);
            channel = channel_manager->get_hdf_read_channel();
            buf_base = (char*) dst_impl->get_direct_ptr(_dst_buf.alloc_offset, 0);
            assert(src_impl->kind == MemoryImpl::MKIND_HDF);
            hli = new Layouts::HDFLayoutIterator<DIM>(domain.get_rect<DIM>(), dst_buf.linearization.get_mapping<DIM>(), dst_buf.block_size);
            fit = oas_vec.begin();
            break;
          }
          case XferDes::XFER_HDF_WRITE:
          {
            hdf_mem = (HDF5Memory*) get_runtime()->get_memory_impl(dst_buf.memory);
            //pthread_rwlock_rdlock(&hdf_mem->rwlock);
            std::map<RegionInstance, HDFMetadata*>::iterator it;
            it = hdf_mem->hdf_metadata.find(inst);
            assert(it != hdf_mem->hdf_metadata.end());
            hdf_metadata = it->second;
            //pthread_rwlock_unlock(&hdf_mem->rwlock);
            channel = channel_manager->get_hdf_write_channel();
            buf_base = (char*) src_impl->get_direct_ptr(_src_buf.alloc_offset, 0);
            assert(dst_impl->kind == MemoryImpl::MKIND_HDF);
            hli = new Layouts::HDFLayoutIterator<DIM>(domain.get_rect<DIM>(), src_buf.linearization.get_mapping<DIM>(), src_buf.block_size);
            fit = oas_vec.begin();
            break;
          }
          default:
            assert(0);
        }
        hdf_reqs = (HDFRequest*) calloc(max_nr, sizeof(HDFRequest));
        for (int i = 0; i < max_nr; i++) {
          hdf_reqs[i].xd = this;
          hdf_reqs[i].hdf_memory = hdf_mem;
          enqueue_request(&hdf_reqs[i]);
        }
     }

      template<unsigned DIM>
      long HDFXferDes<DIM>::get_requests(Request** requests, long nr)
      {
        long ns = 0;
        while (ns < nr && !available_reqs.empty() && fit != oas_vec.end()) {
          requests[ns] = dequeue_request();
          switch (kind) {
            case XferDes::XFER_HDF_READ:
            {
              off_t hdf_ofs = fit->src_offset;
              assert(hdf_metadata->dataset_ids.count(hdf_ofs) > 0);
              //pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_ofs]);
              HDFRequest* hdf_read_req = (HDFRequest*) requests[ns];
              hdf_read_req->dataset_id = hdf_metadata->dataset_ids[hdf_ofs];
              //hdf_read_req->rwlock = &hdf_metadata->dataset_rwlocks[hdf_idx];
              hdf_read_req->mem_type_id = hdf_metadata->datatype_ids[hdf_ofs];
              hsize_t count[DIM], offset[DIM];
              size_t todo = 1;
              for (unsigned i = 0; i < DIM; i++) {
                count[i] = hli->sub_rect.hi[i] - hli->sub_rect.lo[i] + 1;
                todo *= count[i];
                offset[i] = hli->sub_rect.lo[i] - hdf_metadata->lo[i];
              }
              hdf_read_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_ofs]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              herr_t ret = H5Sselect_hyperslab(hdf_read_req->file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
              assert(ret >= 0);
              //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              //pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
              hdf_read_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              off_t dst_offset = calc_mem_loc(0, fit->dst_offset, fit->size,
                                              dst_buf.elmt_size, dst_buf.block_size, hli->cur_idx);
              hdf_read_req->mem_base = buf_base + dst_offset;
              hdf_read_req->nbytes = todo * elemnt_size;
              break;
            }
            case XferDes::XFER_HDF_WRITE:
            {
              off_t hdf_ofs = fit->dst_offset;
              assert(hdf_metadata->dataset_ids.count(hdf_ofs) > 0);
              //pthread_rwlock_rdlock(&hdf_metadata->hdf_memory->rwlock);
              size_t elemnt_size = H5Tget_size(hdf_metadata->datatype_ids[hdf_ofs]);
              HDFRequest* hdf_write_req = (HDFRequest*) requests[ns];
              hdf_write_req->dataset_id = hdf_metadata->dataset_ids[hdf_ofs];
              //hdf_write_req->rwlock = &hdf_metadata->dataset_rwlocks[hdf_idx];
              hdf_write_req->mem_type_id = hdf_metadata->datatype_ids[hdf_ofs];
              hsize_t count[DIM], offset[DIM];
              size_t todo = 1;
              for (unsigned i = 0; i < DIM; i++) {
                count[i] = hli->sub_rect.hi[i] - hli->sub_rect.lo[i] + 1;
                todo *= count[i];
                offset[i] = hli->sub_rect.lo[i] - hdf_metadata->lo[i];
              }
              hdf_write_req->file_space_id = H5Dget_space(hdf_metadata->dataset_ids[hdf_ofs]);
              // HDF dimension always start with zero, but Legion::Domain may start with any integer
              // We need to deal with the offset between them here
              herr_t ret = H5Sselect_hyperslab(hdf_write_req->file_space_id, H5S_SELECT_SET, offset, NULL, count, NULL);
              assert(ret >= 0);
              //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              //pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
              hdf_write_req->mem_space_id = H5Screate_simple(DIM, count, NULL);
              //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
              off_t src_offset = calc_mem_loc(0, fit->src_offset, fit->size,
                                              src_buf.elmt_size, src_buf.block_size, hli->cur_idx);
              hdf_write_req->mem_base = buf_base + src_offset;
              hdf_write_req->nbytes = todo * elemnt_size;
              break;
            }
            default:
              assert(0);
          }
          hli->step();
          if(!hli->any_left()) {
            fit++;
            Layouts::HDFLayoutIterator<DIM>* new_hli = new Layouts::HDFLayoutIterator<DIM>(hli->orig_rect, hli->mapping, hli->block_size);
            delete hli;
            hli = new_hli;
          }
          ns ++;
        }
        return ns;
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_read_done(Request* req)
      {
        req->is_read_done = true;
        // close and release HDF resources
        // currently we don't support ib case
        assert(pre_xd_guid == XFERDES_NO_GUID);
        HDFRequest* hdf_req = (HDFRequest*) req;
        bytes_read += hdf_req->nbytes;
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::notify_request_write_done(Request* req)
      {
        req->is_write_done = true;
        // currently we don't support ib case
        assert(next_xd_guid == XFERDES_NO_GUID);
        HDFRequest* hdf_req = (HDFRequest*) req;
        bytes_write += hdf_req->nbytes;
        //pthread_rwlock_wrlock(&hdf_metadata->hdf_memory->rwlock);
        H5Sclose(hdf_req->mem_space_id);
        H5Sclose(hdf_req->file_space_id);
        //pthread_rwlock_unlock(&hdf_metadata->hdf_memory->rwlock);
        enqueue_request(req);
      }

      template<unsigned DIM>
      void HDFXferDes<DIM>::flush()
      {
        if (kind == XferDes::XFER_HDF_READ) {
        } else {
          assert(kind == XferDes::XFER_HDF_WRITE);
          for (fit = oas_vec.begin(); fit != oas_vec.end(); fit++) {
            off_t hdf_idx = fit->dst_offset;
            hid_t dataset_id = hdf_metadata->dataset_ids[hdf_idx];
            //TODO: I am not sure if we need a lock here to protect HDFflush
            H5Fflush(dataset_id, H5F_SCOPE_LOCAL);
          }
        }
      }
#endif

      /*static*/ void* MemcpyThread::start(void* arg)
      {
        MemcpyThread* worker = (MemcpyThread*) arg;
        worker->thread_loop();
        return NULL;
      }

      void MemcpyThread::thread_loop()
      {
        while (!channel->is_stopped) {
          channel->get_request(thread_queue);
          if (channel->is_stopped)
            break;
          std::deque<MemcpyRequest*>::const_iterator it;
          for (it = thread_queue.begin(); it != thread_queue.end(); it++) {
            MemcpyRequest* req = *it;
            //double starttime = Realm::Clock::current_time_in_microseconds();
            if (req->dim == Request::DIM_1D) {
              memcpy(req->dst_base, req->src_base, req->nbytes);
            } else {
              assert(req->dim == Request::DIM_2D);
              char *src = req->src_base, *dst = req->dst_base;
              for (size_t i = 0; i < req->nlines; i++) {
                memcpy(dst, src, req->nbytes);
                src += req->src_str;
                dst += req->dst_str;
              }
            }
            //double stoptime = Realm::Clock::current_time_in_microseconds();
            //fprintf(stderr, "t = %.2lfus, tp = %.2lfMB/s\n", stoptime - starttime, (req->nbytes / (stoptime - starttime)));
          }
          channel->return_request(thread_queue);
          thread_queue.clear();
        }
      }

      void MemcpyThread::stop()
      {
        channel->stop();
      }

      MemcpyChannel::MemcpyChannel(long max_nr)
      {
        kind = XferDes::XFER_MEM_CPY;
        capacity = max_nr;
        is_stopped = false;
        sleep_threads = false;
        pthread_mutex_init(&pending_lock, NULL);
        pthread_mutex_init(&finished_lock, NULL);
        pthread_cond_init(&pending_cond, NULL);
        //cbs = (MemcpyRequest**) calloc(max_nr, sizeof(MemcpyRequest*));
      }

      MemcpyChannel::~MemcpyChannel()
      {
        pthread_mutex_destroy(&pending_lock);
        pthread_mutex_destroy(&finished_lock);
        pthread_cond_destroy(&pending_cond);
        //free(cbs);
      }

      void MemcpyChannel::stop()
      {
        pthread_mutex_lock(&pending_lock);
        if (!is_stopped)
          pthread_cond_broadcast(&pending_cond);
        is_stopped = true;
        pthread_mutex_unlock(&pending_lock);
      }

      void MemcpyChannel::get_request(std::deque<MemcpyRequest*>& thread_queue)
      {
        pthread_mutex_lock(&pending_lock);
        while (pending_queue.empty() && !is_stopped) {
          sleep_threads = true;
          pthread_cond_wait(&pending_cond, &pending_lock);
        }
        if (!is_stopped) {
          // TODO: enable the following optimization
          //thread_queue.insert(thread_queue.end(), pending_queue.begin(), pending_queue.end());
          thread_queue.push_back(pending_queue.front());
          pending_queue.pop_front();
          //fprintf(stderr, "[%d] thread_queue.size = %lu\n", gettid(), thread_queue.size());
          //pending_queue.clear();
        }
        pthread_mutex_unlock(&pending_lock);
      }

      void MemcpyChannel::return_request(std::deque<MemcpyRequest*>& thread_queue)
      {
        pthread_mutex_lock(&finished_lock);
        finished_queue.insert(finished_queue.end(), thread_queue.begin(), thread_queue.end());
        pthread_mutex_unlock(&finished_lock);
      }

      long MemcpyChannel::submit(Request** requests, long nr)
      {
        MemcpyRequest** mem_cpy_reqs = (MemcpyRequest**) requests;
        pthread_mutex_lock(&pending_lock);
        //if (nr > 0)
          //printf("MemcpyChannel::submit[nr = %ld]\n", nr);
        for (long i = 0; i < nr; i++) {
          pending_queue.push_back(mem_cpy_reqs[i]);
        }
        if (sleep_threads) {
          pthread_cond_broadcast(&pending_cond);
          sleep_threads = false;
        }
        pthread_mutex_unlock(&pending_lock);
        return nr;
        /*
        for (int i = 0; i < nr; i++) {
          push_request(mem_cpy_reqs[i]);
          memcpy(mem_cpy_reqs[i]->dst_buf, mem_cpy_reqs[i]->src_buf, mem_cpy_reqs[i]->nbytes);
          mem_cpy_reqs[i]->xd->notify_request_read_done(mem_cpy_reqs[i]);
          mem_cpy_reqs[i]->xd->notify_request_write_done(mem_cpy_reqs[i]);
        }
        return nr;
        */
      }

      void MemcpyChannel::pull()
      {
        pthread_mutex_lock(&finished_lock);
        while (!finished_queue.empty()) {
          MemcpyRequest* req = finished_queue.front();
          finished_queue.pop_front();
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        pthread_mutex_unlock(&finished_lock);
        /*
        while (true) {
          long np = worker->pull(cbs, capacity);
          for (int i = 0; i < np; i++) {
            cbs[i]->xd->notify_request_read_done(cbs[i]);
            cbs[i]->xd->notify_request_write_done(cbs[i]);
          }
          if (np != capacity)
            break;
        }
        */
      }

      long MemcpyChannel::available()
      {
        return capacity;
      }

      GASNetChannel::GASNetChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        capacity = max_nr;
      }

      GASNetChannel::~GASNetChannel()
      {
      }

      long GASNetChannel::submit(Request** requests, long nr)
      {
        for (long i = 0; i < nr; i++) {
          GASNetRequest* req = (GASNetRequest*) requests[i];
          switch (kind) {
            case XferDes::XFER_GASNET_READ:
            {
              get_runtime()->global_memory->get_bytes(req->gas_off,
                                                      req->mem_base,
                                                      req->nbytes);
              break;
            }
            case XferDes::XFER_GASNET_WRITE:
            {
              get_runtime()->global_memory->put_bytes(req->gas_off,
                                                      req->mem_base,
                                                      req->nbytes);
              break;
            }
            default:
              assert(0);
          }
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        return nr;
      }

      void GASNetChannel::pull()
      {
      }

      long GASNetChannel::available()
      {
        return capacity;
      }

      RemoteWriteChannel::RemoteWriteChannel(long max_nr)
      {
        capacity = max_nr;
      }

      RemoteWriteChannel::~RemoteWriteChannel() {}

      long RemoteWriteChannel::submit(Request** requests, long nr)
      {
        assert(nr <= capacity);
        for (long i = 0; i < nr; i ++) {
          RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
          if (req->dim == Request::DIM_1D) {
            XferDesRemoteWriteMessage::send_request(
                req->dst_node, req->dst_base, req->src_base, req->nbytes, req);
          } else {
            assert(req->dim == Request::DIM_2D);
            // dest MUST be continuous
            assert(req->nlines <= 1 || ((size_t)req->dst_str) == req->nbytes);
            XferDesRemoteWriteMessage::send_request(
                req->dst_node, req->dst_base, req->src_base, req->nbytes,
                req->src_str, req->nlines, req);
          }
          capacity--;
        /*RemoteWriteRequest* req = (RemoteWriteRequest*) requests[i];
          req->complete_event = GenEventImpl::create_genevent()->current_event();
          Realm::RemoteWriteMessage::RequestArgs args;
          args.mem = req->dst_mem;
          args.offset = req->dst_offset;
          args.event = req->complete_event;
          args.sender = gasnet_mynode();
          args.sequence_id = 0;

          Realm::RemoteWriteMessage::Message::request(ID(args.mem).node(), args,
                                                      req->src_buf, req->nbytes,
                                                      PAYLOAD_KEEPREG,
                                                      req->dst_buf);*/
        }
        return nr;
      }

      void RemoteWriteChannel::pull()
      {
      }

      long RemoteWriteChannel::available()
      {
        return capacity;
      }

#ifdef USE_DISK
      DiskChannel::DiskChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        ctx = 0;
        capacity = max_nr;
        int ret = io_setup(max_nr, &ctx);
        assert(ret >= 0);
        assert(available_cb.empty());
        cb = (struct iocb*) calloc(max_nr, sizeof(struct iocb));
        cbs = (struct iocb**) calloc(max_nr, sizeof(struct iocb*));
        events = (struct io_event*) calloc(max_nr, sizeof(struct io_event));
        switch (kind) {
          case XferDes::XFER_DISK_READ:
            for (long i = 0; i < max_nr; i++) {
              memset(&cb[i], 0, sizeof(cb[i]));
              cb[i].aio_lio_opcode = IOCB_CMD_PREAD;
              available_cb.push_back(&cb[i]);
            }
            break;
          case XferDes::XFER_DISK_WRITE:
            for (long i = 0; i < max_nr; i++) {
              memset(&cb[i], 0, sizeof(cb[i]));
              cb[i].aio_lio_opcode = IOCB_CMD_PWRITE;
              available_cb.push_back(&cb[i]);
            }
            break;
          default:
            assert(0);
        }
      }

      DiskChannel::~DiskChannel()
      {
        io_destroy(ctx);
        free(cb);
        free(cbs);
        free(events);
      }

      long DiskChannel::submit(Request** requests, long nr)
      {
        int ns = 0;
        switch (kind) {
          case XferDes::XFER_DISK_READ:
          case XferDes::XFER_DISK_WRITE:
            while (ns < nr && !available_cb.empty()) {
              DiskRequest* req = (DiskRequest*) requests[ns];
              cbs[ns] = available_cb.back();
              available_cb.pop_back();
              cbs[ns]->aio_fildes = req->fd;
              cbs[ns]->aio_data = (uint64_t) (req);
              cbs[ns]->aio_buf = (uint64_t) (req->mem_base);
              cbs[ns]->aio_offset = req->disk_off;
              cbs[ns]->aio_nbytes = req->nbytes;
              ns++;
            }
            break;
          default:
            assert(0);
        }
        assert(ns == nr);
        int ret = io_submit(ctx, ns, cbs);
        if (ret < 0) {
          perror("io_submit error");
        }
        return ret;
      }

      void DiskChannel::pull()
      {
        int nr = io_getevents(ctx, 0, capacity, events, NULL);
        if (nr < 0)
          perror("io_getevents error");
        for (int i = 0; i < nr; i++) {
          Request* req = (Request*) events[i].data;
          struct iocb* ret_cb = (struct iocb*) events[i].obj;
          available_cb.push_back(ret_cb);
          assert(events[i].res == (int64_t)ret_cb->aio_nbytes);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
      }

      long DiskChannel::available()
      {
        return available_cb.size();
      }
#endif /*USE_DISK*/
    
#ifdef USE_CUDA
      GPUChannel::GPUChannel(GPU* _src_gpu, long max_nr, XferDes::XferKind _kind)
      {
        src_gpu = _src_gpu;
        kind = _kind;
        capacity = max_nr;
      }

      GPUChannel::~GPUChannel()
      {
      }

      long GPUChannel::submit(Request** requests, long nr)
      {
        for (long i = 0; i < nr; i++) {
          GPURequest* req = (GPURequest*) requests[i];
          if (req->dim == Request::DIM_1D) { 
            switch (kind) {
              case XferDes::XFER_GPU_TO_FB:
                src_gpu->copy_to_fb(req->dst_gpu_off, req->src_base,
                                    req->nbytes, &req->event);
                break;
              case XferDes::XFER_GPU_FROM_FB:
                src_gpu->copy_from_fb(req->dst_base, req->src_gpu_off,
                                      req->nbytes, &req->event);
                break;
              case XferDes::XFER_GPU_IN_FB:
                src_gpu->copy_within_fb(req->dst_gpu_off, req->src_gpu_off,
                                        req->nbytes, &req->event);
                break;
              case XferDes::XFER_GPU_PEER_FB:
                src_gpu->copy_to_peer(req->dst_gpu, req->dst_gpu_off,
                                      req->src_gpu_off, req->nbytes,
                                      &req->event);
                break;
              default:
                assert(0);
            }
          } else {
            assert(req->dim == Request::DIM_2D);
            switch (kind) {
              case XferDes::XFER_GPU_TO_FB:
                src_gpu->copy_to_fb_2d(req->dst_gpu_off, req->src_base,
                                       req->dst_str, req->src_str,
                                       req->nbytes, req->nlines, &req->event);
                break;
              case XferDes::XFER_GPU_FROM_FB:
                src_gpu->copy_from_fb_2d(req->dst_base, req->src_gpu_off,
                                         req->dst_str, req->src_str,
                                         req->nbytes, req->nlines,
                                         &req->event);
                break;
              case XferDes::XFER_GPU_IN_FB:
                src_gpu->copy_within_fb_2d(req->dst_gpu_off, req->src_gpu_off,
                                           req->dst_str, req->src_str,
                                           req->nbytes, req->nlines,
                                           &req->event);
                break;
              case XferDes::XFER_GPU_PEER_FB:
                src_gpu->copy_to_peer_2d(req->dst_gpu, req->dst_gpu_off,
                                         req->src_gpu_off, req->dst_str,
                                         req->src_str, req->nbytes,
                                         req->nlines, &req->event);
                break;
              default:
                assert(0);
            }
          }
          pending_copies.push_back(req);
        }
        return nr;
      }

      void GPUChannel::pull()
      {
        switch (kind) {
          case XferDes::XFER_GPU_TO_FB:
          case XferDes::XFER_GPU_FROM_FB:
          case XferDes::XFER_GPU_IN_FB:
          case XferDes::XFER_GPU_PEER_FB:
            while (!pending_copies.empty()) {
              GPURequest* req = (GPURequest*)pending_copies.front();
              if (req->event.has_triggered()) {
                req->xd->notify_request_read_done(req);
                req->xd->notify_request_write_done(req);
                pending_copies.pop_front();
              }
              else
                break;
            }
            break;
          default:
            assert(0);
        }
      }

      long GPUChannel::available()
      {
        return capacity - pending_copies.size();
      }
#endif

#ifdef USE_HDF
      HDFChannel::HDFChannel(long max_nr, XferDes::XferKind _kind)
      {
        kind = _kind;
        capacity = max_nr;
      }

      HDFChannel::~HDFChannel() {}

      long HDFChannel::submit(Request** requests, long nr)
      {
        HDFRequest** hdf_reqs = (HDFRequest**) requests;
        for (long i = 0; i < nr; i++) {
          HDFRequest* req = hdf_reqs[i];
          //pthread_rwlock_rdlock(req->rwlock);
          if (kind == XferDes::XFER_HDF_READ)
            H5Dread(req->dataset_id, req->mem_type_id,
                    req->mem_space_id, req->file_space_id,
                    H5P_DEFAULT, req->mem_base);
          else
            H5Dwrite(req->dataset_id, req->mem_type_id,
                     req->mem_space_id, req->file_space_id,
                     H5P_DEFAULT, req->mem_base);
          //pthread_rwlock_unlock(req->rwlock);
          req->xd->notify_request_read_done(req);
          req->xd->notify_request_write_done(req);
        }
        return nr;
      }

      void HDFChannel::pull() {}

      long HDFChannel::available()
      {
        return capacity;
      }
#endif

#ifdef MEMCPY_THREAD_CODE
      void MemcpyThread::work()
      {
        while (true) {
          pthread_mutex_lock(&submit_lock);
          //printf("[MemcpyThread] CP#1\n");
          if (num_pending_reqs == 0)
            pthread_cond_wait(&condvar, &submit_lock);
          //printf("[MemcpyThread] Pull from pending queue\n");
          //printf("[MemcpyThread] num_pending_reqs = %ld\n", num_pending_reqs);
          assert(pending_queue.size() > 0);
          MemcpyRequest* cur_req = pending_queue.front();
          pending_queue.pop();
          num_pending_reqs --;
          pthread_mutex_unlock(&submit_lock);
          //printf("[MemcpyThread] Begin processing copy\n");
          //printf("[MemcpyThread] dst = %ld, src = %ld, nbytes = %lu\n", (off_t) cur_req->dst_buf, (off_t) cur_req->src_buf, cur_req->nbytes);
          memcpy(cur_req->dst_buf, cur_req->src_buf, cur_req->nbytes);
          //printf("[MemcpyThread] Finish processing copy\n");
          pthread_mutex_lock(&pull_lock);
          //printf("[MemcpyThread] Push into finished queue\n");
          finished_queue.push(cur_req);
          pthread_mutex_unlock(&pull_lock);
        }
      }

      void* MemcpyThread::start(void* arg)
      {
        printf("[MemcpyThread] start...\n");
        MemcpyThread* worker = (MemcpyThread*) arg;
        worker->work();
        return NULL;
      }
#endif
      /*static*/
      void XferDesRemoteWriteMessage::handle_request(RequestArgs args,
                                                     const void *data,
                                                     size_t datalen)
      {
        // assert data copy is in right position
        assert(data == args.dst_buf);
        XferDesRemoteWriteAckMessage::send_request(args.sender, args.req);
      }

      /*static*/
      void XferDesRemoteWriteAckMessage::handle_request(RequestArgs args)
      {
        RemoteWriteRequest* req = args.req;
        req->xd->notify_request_read_done(req);
        req->xd->notify_request_write_done(req);
        channel_manager->get_remote_write_channel()->notify_completion();
      }

      /*static*/
      void XferDesCreateMessage::handle_request(RequestArgs args,
                                                const void *msgdata,
                                                size_t msglen)
      {
        const Payload *payload = (const Payload *)msgdata;
        std::vector<OffsetsAndSize> oas_vec(payload->oas_vec_size);
        for(size_t i = 0; i < payload->oas_vec_size; i++)
          oas_vec[i] = payload->oas_vec(i);
        Buffer src_buf, dst_buf;
        src_buf.deserialize(payload->src_buf_bits);
        src_buf.memory = args.src_mem;
        dst_buf.deserialize(payload->dst_buf_bits);
        dst_buf.memory = args.dst_mem;
        switch(payload->domain.dim) {
        case 0:
          create_xfer_des<0>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             false/*mark_started*/, src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        case 1:
          create_xfer_des<1>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             false/*mark_started*/, src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        case 2:
          create_xfer_des<2>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             false/*mark_started*/, src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        case 3:
          create_xfer_des<3>(payload->dma_request, payload->launch_node,
                             payload->guid, payload->pre_xd_guid, payload->next_xd_guid,
                             false/*mark_started*/, src_buf, dst_buf, payload->domain, oas_vec,
                             payload->max_req_size, payload->max_nr, payload->priority,
                             payload->order, payload->kind, args.fence, args.inst);
          break;
        default:
          assert(0);
        }
      }

      /*static*/ void XferDesDestroyMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->destroy_xferDes(args.guid);
      }

      /*static*/ void UpdateBytesWriteMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->update_pre_bytes_write(args.guid, args.bytes_write);
      }

      /*static*/ void UpdateBytesReadMessage::handle_request(RequestArgs args)
      {
        xferDes_queue->update_next_bytes_read(args.guid, args.bytes_read);
      }

      void DMAThread::dma_thread_loop()
      {
        log_new_dma.info("start dma thread loop");
        while (!is_stopped) {
          bool is_empty = true;
          std::map<Channel*, PriorityXferDesQueue*>::iterator it;
          for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
            if(!it->second->empty()) {
              is_empty = false;
              break;
            }
          }
          xd_queue->dequeue_xferDes(this, is_empty);

          for (it = channel_to_xd_pool.begin(); it != channel_to_xd_pool.end(); it++) {
            it->first->pull();
            long nr = it->first->available();
            if (nr == 0)
              continue;
            std::vector<XferDes*> finish_xferdes;
            PriorityXferDesQueue::iterator it2;
            for (it2 = it->second->begin(); it2 != it->second->end(); it2++) {
              assert((*it2)->channel == it->first);
              // If we haven't mark started and we are the first xd, mark start
              if ((*it2)->mark_start) {
                (*it2)->dma_request->mark_started();
                (*it2)->mark_start = false;
              }
              long nr_got = (*it2)->get_requests(requests, min(nr, max_nr));
              long nr_submitted = it->first->submit(requests, nr_got);
              nr -= nr_submitted;
              assert(nr_got == nr_submitted);
              if ((*it2)->is_completed()) {
                finish_xferdes.push_back(*it2);
                //printf("finish_xferdes.size() = %lu\n", finish_xferdes.size());
              }
              if (nr == 0)
                break;
            }
            while(!finish_xferdes.empty()) {
              XferDes *xd = finish_xferdes.back();
              finish_xferdes.pop_back();
              it->second->erase(xd);
              // We flush all changes into destination before mark this XferDes as completed
              xd->flush();
              log_new_dma.info("Finish XferDes : id(" IDFMT ")", xd->guid);
              xd->mark_completed();
              /*bool need_to_delete_dma_request = xd->mark_completed();
              if (need_to_delete_dma_request) {
                DmaRequest* dma_request = xd->dma_request;
                delete dma_request;
              }*/
            }
          }
        }
        log_new_dma.info("finish dma thread loop");
      }

      XferDesQueue* get_xdq_singleton()
      {
        return xferDes_queue;
      }
#ifdef USE_CUDA
      void register_gpu_in_dma_systems(GPU* gpu)
      {
        dma_all_gpus.push_back(gpu);
      }
#endif
      void start_channel_manager(int count, int max_nr, Realm::CoreReservationSet& crs)
      {
        xferDes_queue = new XferDesQueue(crs);
        channel_manager = new ChannelManager;
        xferDes_queue->start_worker(count, max_nr, channel_manager);
      }

      void XferDesQueue::start_worker(int count, int max_nr, ChannelManager* channel_manager) 
      {
        log_new_dma.info("XferDesQueue: start_workers");
        // TODO: count is currently ignored
        num_threads = 3;
        num_memcpy_threads = 2;
#ifdef USE_HDF
        // Need a dedicated thread for handling HDF requests
        num_threads ++;
#endif
#ifdef USE_CUDA
        num_threads ++;
#endif
        int idx = 0;
        dma_threads = (DMAThread**) calloc(num_threads, sizeof(DMAThread*));
        // dma thread #1: memcpy
        MemcpyChannel* memcpy_channel = channel_manager->create_memcpy_channel(max_nr);
        dma_threads[idx++] = new DMAThread(max_nr, xferDes_queue, memcpy_channel);
        // dma thread #2: async xfer
        std::vector<Channel*> async_channels, gasnet_channels;
        async_channels.push_back(channel_manager->create_remote_write_channel(max_nr));
#ifdef USE_DISK
        async_channels.push_back(channel_manager->create_disk_read_channel(max_nr));
        async_channels.push_back(channel_manager->create_disk_write_channel(max_nr));
#endif /*USE_DISK*/
#ifdef USE_FILE
        async_channels.push_back(channel_manager->create_file_read_channel(max_nr));
        async_channels.push_back(channel_manager->create_file_write_channel(max_nr));
#endif
        dma_threads[idx++] = new DMAThread(max_nr, xferDes_queue, async_channels);
        gasnet_channels.push_back(channel_manager->create_gasnet_read_channel(max_nr));
        gasnet_channels.push_back(channel_manager->create_gasnet_write_channel(max_nr));
        dma_threads[idx++] = new DMAThread(max_nr, xferDes_queue, gasnet_channels);
#ifdef USE_CUDA
        std::vector<Channel*> gpu_channels;
        std::vector<GPU*>::iterator it;
        for (it = dma_all_gpus.begin(); it != dma_all_gpus.end(); it ++) {
          gpu_channels.push_back(channel_manager->create_gpu_to_fb_channel(max_nr, *it));
          gpu_channels.push_back(channel_manager->create_gpu_from_fb_channel(max_nr, *it));
          gpu_channels.push_back(channel_manager->create_gpu_in_fb_channel(max_nr, *it));
          gpu_channels.push_back(channel_manager->create_gpu_peer_fb_channel(max_nr, *it));
        }
        dma_threads[idx++] = new DMAThread(max_nr, xferDes_queue, gpu_channels);
#endif
#ifdef USE_HDF
        std::vector<Channel*> hdf_channels;
        hdf_channels.push_back(channel_manager->create_hdf_read_channel(max_nr));
        hdf_channels.push_back(channel_manager->create_hdf_write_channel(max_nr));
        dma_threads[idx++] = new DMAThread(max_nr, xferDes_queue, hdf_channels);
#endif
        assert(idx == num_threads);
        for (int i = 0; i < num_threads; i++) {
          // register dma thread to XferDesQueue
           register_dma_thread(dma_threads[i]);
        }

        Realm::ThreadLaunchParameters tlp;

        for(int i = 0; i < num_threads; i++) {
          log_new_dma.info("Create a DMA worker thread");
          Realm::Thread *t = Realm::Thread::create_kernel_thread<DMAThread,
                                            &DMAThread::dma_thread_loop>(dma_threads[i],
  						                         tlp,
  					                                 core_rsrv,
  					                                 0 /* default scheduler*/);
          worker_threads.push_back(t);
        }

        // Next we create memcpy threads
        memcpy_threads =(MemcpyThread**) calloc(num_memcpy_threads, sizeof(MemcpyThread*));
        for (int i = 0; i < num_memcpy_threads; i++) {
          memcpy_threads[i] = new MemcpyThread(memcpy_channel);
          Realm::Thread *t = Realm::Thread::create_kernel_thread<MemcpyThread,
                                            &MemcpyThread::thread_loop>(memcpy_threads[i],
                                                                        tlp,
                                                                        core_rsrv,
                                                                        0 /*default scheduler*/);
          worker_threads.push_back(t);
        }
        assert(worker_threads.size() == (size_t)(num_threads + num_memcpy_threads));
      }

      void stop_channel_manager()
      {
        xferDes_queue->stop_worker();
        delete xferDes_queue;
        delete channel_manager;
      }

      void XferDesQueue::stop_worker() {
        for (int i = 0; i < num_threads; i++)
          dma_threads[i]->stop();
        for (int i = 0; i < num_memcpy_threads; i++)
          memcpy_threads[i]->stop();
        // reap all the threads
        for(std::vector<Realm::Thread *>::iterator it = worker_threads.begin();
            it != worker_threads.end();
            it++) {
          (*it)->join();
          delete (*it);
        }
        worker_threads.clear();
        for (int i = 0; i < num_threads; i++)
          delete dma_threads[i];
        for (int i = 0; i < num_memcpy_threads; i++)
          delete memcpy_threads[i];
        delete[] dma_threads;
        delete[] memcpy_threads;
      }


      template<unsigned DIM>
      void create_xfer_des(DmaRequest* _dma_request,
                           gasnet_node_t _launch_node,
                           XferDesID _guid,
                           XferDesID _pre_xd_guid,
                           XferDesID _next_xd_guid,
                           bool mark_started,
                           const Buffer& _src_buf,
                           const Buffer& _dst_buf,
                           const Domain& _domain,
                           const std::vector<OffsetsAndSize>& _oas_vec,
                           uint64_t _max_req_size,
                           long max_nr,
                           int _priority,
                           XferOrder::Type _order,
                           XferDes::XferKind _kind,
                           XferDesFence* _complete_fence,
                           RegionInstance inst)
      {
        if (ID(_src_buf.memory).memory.owner_node == gasnet_mynode()) {
          size_t total_field_size = 0;
          for (unsigned i = 0; i < _oas_vec.size(); i++) {
            total_field_size += _oas_vec[i].size;
          }
          log_new_dma.info("Create local XferDes: id(" IDFMT "), pre(" IDFMT
                           "), next(" IDFMT "), type(%d), domain(%zu), "
                           "total_field_size(%zu)",
                           _guid, _pre_xd_guid, _next_xd_guid, _kind,
                           _domain.get_volume(), total_field_size);
          XferDes* xd;
          switch (_kind) {
          case XferDes::XFER_MEM_CPY:
            xd = new MemcpyXferDes<DIM>(_dma_request, _launch_node,
                                        _guid, _pre_xd_guid, _next_xd_guid,
                                        mark_started,
                                        _src_buf, _dst_buf, _domain, _oas_vec,
                                        _max_req_size, max_nr, _priority,
                                        _order, _complete_fence);
            break;
          case XferDes::XFER_GASNET_READ:
          case XferDes::XFER_GASNET_WRITE:
            xd = new GASNetXferDes<DIM>(_dma_request, _launch_node,
                                        _guid, _pre_xd_guid, _next_xd_guid,
                                        mark_started,
                                        _src_buf, _dst_buf, _domain, _oas_vec,
                                        _max_req_size, max_nr, _priority,
                                        _order, _kind, _complete_fence);
            break;
          case XferDes::XFER_REMOTE_WRITE:
            xd = new RemoteWriteXferDes<DIM>(_dma_request, _launch_node,
                                             _guid, _pre_xd_guid, _next_xd_guid,
                                             mark_started,
                                             _src_buf, _dst_buf, _domain, _oas_vec,
                                             _max_req_size, max_nr, _priority,
                                             _order, _complete_fence);
            break;
#ifdef USE_DISK
          case XferDes::XFER_DISK_READ:
          case XferDes::XFER_DISK_WRITE:
            xd = new DiskXferDes<DIM>(_dma_request, _launch_node,
                                      _guid, _pre_xd_guid, _next_xd_guid,
                                      mark_started,
                                      _src_buf, _dst_buf, _domain, _oas_vec,
                                      _max_req_size, max_nr, _priority,
                                      _order, _kind, _complete_fence);
            break;
#endif
#ifdef USE_CUDA
          case XferDes::XFER_GPU_FROM_FB:
          case XferDes::XFER_GPU_TO_FB:
          case XferDes::XFER_GPU_IN_FB:
          case XferDes::XFER_GPU_PEER_FB:
            xd = new GPUXferDes<DIM>(_dma_request, _launch_node,
                                     _guid, _pre_xd_guid, _next_xd_guid,
                                     mark_started,
                                     _src_buf, _dst_buf, _domain, _oas_vec,
                                     _max_req_size, max_nr, _priority,
                                     _order, _kind, _complete_fence);
            break;
#endif
#ifdef USE_HDF
          case XferDes::XFER_HDF_READ:
          case XferDes::XFER_HDF_WRITE:
            xd = new HDFXferDes<DIM>(_dma_request, _launch_node,
                                     _guid, _pre_xd_guid, _next_xd_guid,
                                     mark_started,
                                     inst, _src_buf, _dst_buf, _domain, _oas_vec,
                                     _max_req_size, max_nr, _priority,
                                     _order, _kind, _complete_fence);
            break;
#endif
        default:
          printf("_kind = %d\n", _kind);
          assert(false);
        }
        xferDes_queue->enqueue_xferDes_local(xd);
      } else {
        log_new_dma.info("Create remote XferDes: id(" IDFMT "),"
                         " pre(" IDFMT "), next(" IDFMT "), type(%d)",
                         _guid, _pre_xd_guid, _next_xd_guid, _kind);
        XferDesCreateMessage::send_request(ID(_src_buf.memory).memory.owner_node,
                                           _dma_request, _launch_node,
                                           _guid, _pre_xd_guid, _next_xd_guid,
                                           _src_buf, _dst_buf, _domain, _oas_vec,
                                           _max_req_size, max_nr, _priority,
                                           _order, _kind, _complete_fence, inst);
      }
    }

    void destroy_xfer_des(XferDesID _guid)
    {
      log_new_dma.info("Destroy XferDes: id(" IDFMT ")", _guid);
      gasnet_node_t execution_node = _guid >> (XferDesQueue::NODE_BITS + XferDesQueue::INDEX_BITS);
      if (execution_node == gasnet_mynode()) {
        xferDes_queue->destroy_xferDes(_guid);
      }
      else {
        XferDesDestroyMessage::send_request(execution_node, _guid);
      }
    }

      template class MemcpyXferDes<1>;
      template class MemcpyXferDes<2>;
      template class MemcpyXferDes<3>;
      template class GASNetXferDes<1>;
      template class GASNetXferDes<2>;
      template class GASNetXferDes<3>;
      template class RemoteWriteXferDes<1>;
      template class RemoteWriteXferDes<2>;
      template class RemoteWriteXferDes<3>;
#ifdef USE_DISK
      template class DiskXferDes<1>;
      template class DiskXferDes<2>;
      template class DiskXferDes<3>;
#endif /*USE_DISK*/
#ifdef USE_CUDA
      template class GPUXferDes<1>;
      template class GPUXferDes<2>;
      template class GPUXferDes<3>;
#endif
#ifdef USE_HDF
      template class HDFXferDes<1>;
      template class HDFXferDes<2>;
      template class HDFXferDes<3>;
#endif
 } // namespace LowLevel
} // namespace LegionRuntime

#ifdef DEADCODE
void DiskReadChannel::poll()
{
  int nr = io_getevents(ctx, 0, capacity, events, NULL);
  if (nr < 0)
    perror("io_getevents error");
  for (int i = 0; i < nr; i++) {
    DiskReadRequest* req = (DiskReadRequest*) events[i].data;
    struct iocb* ret_cb = (struct iocb*) events[i].obj;
    available_cb.push_back(ret_cb);
    assert(events[i].res == ret_cb->aio_nbytes);
    req->num_flying_aios --;
    if (req->num_flying_aios == 0 && cur_req != req) {
      // this indicates we have finished all aios within this req
      // time to recycle the request space
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }
  }
  // see if we can launch more aios
  int ns = 0;
  while (!available_cb.empty()) {
    if (cur_req != NULL) {
      // Case 1: we are dealing with a DiskReadRequest
      while (iter_1d != cur_req->copies_1D.end() && !available_cb.empty()) {
        cbs[ns] = available_cb.back();
        available_cb.pop_back();
        cbs[ns]->aio_fildes = cur_req->fd;
        cbs[ns]->aio_data = (int64_t) (cur_req);
        cbs[ns]->aio_buf = (*iter_1d)->dst_offset;
        cbs[ns]->aio_offset = (*iter_1d)->src_offset;
        cbs[ns]->aio_nbytes = (*iter_1d)->nbytes;
        cur_req->num_flying_aios ++;
        ns ++;
        iter_1d ++;
      }
      // handle 2D cases
      while (iter_2d != cur_req->copies_2D.end() && !available_cb.empty()) {
        cbs[ns] = available_cb.back();
        available_cb.pop_back();
        cbs[ns]->aio_fildes = cur_req->fd;
        cbs[ns]->aio_data = (int64_t) (cur_req);
        cbs[ns]->aio_buf = (*iter_2d)->dst_offset + (*iter_2d)->dst_stride * cur_line;
        cbs[ns]->aio_offset = (*iter_2d)->src_offset + (*iter_2d)->src_stride * cur_line;
        cbs[ns]->aio_nbytes = (*iter_2d)->nbytes;
        cur_req->num_flying_aios ++;
        ns ++;
        cur_line ++;
        if (cur_line == (*iter_2d)->nlines) {
          iter_2d ++; cur_line = 0;
        }
      }
      // submit aios
      if (ns > 0) {
        int ret = io_submit(ctx, ns, cbs);
        assert(ret == ns);
      }
      if (iter_1d == cur_req->copies_1D.end() && iter_2d == cur_req->copies_2D.end())
        cur_req = NULL;
    }
    else if (pending_reqs.empty()) {
      // Case 2: this indicates there is no more DiskReadRequest pending
      break;
    }
    else {
      // Case 3: move to the next DiskReadRequest
      cur_req = pending_reqs.front();
      iter_1d = cur_req->copies_1D.begin();
      iter_2d = cur_req->copies_2D.begin();
      cur_line = 0;
      pending_reqs.pop();
    }
  }
}

#endif

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

// INCLDUED FROM bytearray.h - DO NOT INCLUDE THIS DIRECTLY

// this is a nop, but it's for the benefit of IDEs trying to parse this file
#include "bytearray.h"

#include <string.h>
#include <stdlib.h>
#include <assert.h>

namespace Realm {

  ////////////////////////////////////////////////////////////////////////
  //
  // class ByteArray
  //

  ByteArray::ByteArray(void)
    : array_base(0), array_size(0)
  {}

  ByteArray::ByteArray(const void *copy_from, size_t copy_size)
    : array_base(0), array_size(0)
  {
    if(copy_size) {
      array_base = malloc(copy_size);
      assert(array_base != 0);
      memcpy(array_base, copy_from, copy_size);
      array_size = copy_size;
    }
  }

  ByteArray::ByteArray(const ByteArray& copy_from)
    : array_base(0), array_size(0)
  {
    if(copy_from.size()) {
      array_base = malloc(copy_from.size());
      assert(array_base != 0);
      memcpy(array_base, copy_from.base(), copy_from.size());
      array_size = copy_from.size();
    }
  }

  ByteArray::~ByteArray(void)
  {
    if(array_size)
      free(array_base);
  }

  // copies the contents of the rhs ByteArray
  ByteArray& ByteArray::operator=(const ByteArray& copy_from)
  {
    clear();  // throw away any data we had before
    if(copy_from.size()) {
      array_base = malloc(copy_from.size());
      assert(array_base != 0);
      memcpy(array_base, copy_from.base(), copy_from.size());
      array_size = copy_from.size();
    }
    return *this;
  }

  // swaps the contents of two ByteArrays - returns a reference to the first one
  // this allows you to transfer ownership of a byte array to a called function via:
  //   ByteArray().swap(old_array)
  ByteArray& ByteArray::swap(ByteArray& swap_with)
  {
    std::swap(array_base, swap_with.array_base);
    std::swap(array_size, swap_with.array_size);
    return *this;
  }

  // copy raw data in
  ByteArray& ByteArray::set(const void *copy_from, size_t copy_size)
  {
    clear();  // throw away any data we had before
    if(copy_size) {
      array_base = malloc(copy_size);
      assert(array_base != 0);
      memcpy(array_base, copy_from, copy_size);
      array_size = copy_size;
    }
    return *this;
  }

  // give ownership of a buffer to a ByteArray
  ByteArray& ByteArray::attach(void *new_base, size_t new_size)
  {
    clear();  // throw away any data we had before
    if(new_size) {
      array_base = new_base;
      assert(array_base != 0);
      array_size = new_size;
    } else {
      // if we were given ownership of a 0-length allocation, free it rather than leaking it
      if(new_base)
	free(new_base);
    }
    return *this;
  }

  // access to base pointer and size
  void *ByteArray::base(void)
  {
    return array_base;
  }

  const void *ByteArray::base(void) const
  {
    return array_base;
  }

  size_t ByteArray::size(void) const
  {
    return array_size;
  }

  // helper to access bytes as typed references
  template <typename T>
  T& ByteArray::at(size_t offset)
  {
    // always range check?
    assert((offset + sizeof(T)) <= array_size);
    return *reinterpret_cast<T *>(reinterpret_cast<char *>(array_base) + offset);
  }

  template <typename T>
  const T& ByteArray::at(size_t offset) const
  {
    // always range check?
    assert((offset + sizeof(T)) <= array_size);
    return *reinterpret_cast<const T *>(reinterpret_cast<char *>(array_base) + offset);
  }

  // explicitly deallocate any held storage
  void ByteArray::clear(void)
  {
    if(array_size) {
      free(array_base);
      array_base = 0;
      array_size = 0;
    }
  }

  // extract the pointer from the ByteArray (caller assumes ownership)
  void *ByteArray::detach(void)
  {
    if(array_size) {
      void *retval = array_base;
      array_base = 0;
      array_size = 0;
      return retval;
    } else
      return 0;
  }

  // support for realm-style serialization
  template <typename S>
  bool operator<<(S& serdez, const ByteArray& a)
  {
    return((serdez << a.size()) &&
	   ((a.size() == 0) ||
	    serdez.append_bytes(a.base(), a.size())));
  }

  template <typename S>
  bool operator>>(S& serdez, ByteArray& a)
  {
    size_t new_size;
    if(!(serdez >> new_size)) return false;
    void *new_base = 0;
    if(new_size) {
      new_base = malloc(new_size);
      assert(new_base != 0);
      if(!serdez.extract_bytes(new_base, new_size)) {
	free(new_base);  // no leaks plz
	return false;
      }
    }
    a.attach(new_base, new_size);
    return true;
  }

}; // namespace Realm

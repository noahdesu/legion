/* Copyright 2016 Stanford University, NVIDIA Corporation
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

#ifndef __BISHOP_MAPPER_H__
#define __BISHOP_MAPPER_H__

#include <vector>
#include <string>

#include "legion.h"
#include "default_mapper.h"

extern "C" {
#include "bishop_c.h"
}

namespace Legion {
  namespace Mapping {

		class BishopMapper : public DefaultMapper
		{
			public:
				BishopMapper(const std::vector<bishop_task_rule_t>&,
                     const std::vector<bishop_region_rule_t>&,
                     bishop_mapper_state_init_fn_t,
                     MapperRuntime*, Machine, Processor);
				~BishopMapper();

        virtual void select_task_options(const MapperContext ctx,
                                         const Task&         task,
                                               TaskOptions&  output);
        virtual void slice_task(const MapperContext    ctx,
                                const Task&            task,
                                const SliceTaskInput&  input,
                                      SliceTaskOutput& output);
        virtual void map_task(const MapperContext  ctx,
                              const Task&          task,
                              const MapTaskInput&  input,
                                    MapTaskOutput& output);

      private:
        std::vector<bishop_task_rule_t> task_rules;
        std::vector<bishop_region_rule_t> region_rules;
        bishop_mapper_state_init_fn_t mapper_init;
        bishop_mapper_state_t mapper_state;
		};

	};
};
#endif // __BISHOP_MAPPER_H__

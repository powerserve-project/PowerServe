// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "core/typedefs.hpp"

namespace powerserve {

struct SpeculativeConfig {
    size_t draft_batch_size = 12;

    struct {
        size_t top_k      = 15;
        float temperature = 1.5f;
        float p_base      = 0.9f;
    } draft_sampler;

    struct {
        size_t max_fan_out = 3;
        float min_prob     = 0.2f;
        bool early_stop    = true;
        bool debug         = false;
    } token_tree;
};

} // namespace powerserve

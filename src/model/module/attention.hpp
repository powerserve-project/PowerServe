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

#include "core/config.hpp"
#include "graph/node.hpp"
#include "model/common/weights.hpp"

namespace powerserve {

struct Attention {

public:
    const ModelConfig::LLMConfig &m_config;
    std::shared_ptr<Weight> m_weights;

public:
    Attention(const ModelConfig::LLMConfig &config, const std::shared_ptr<Weight> &weights) :
        m_config(config),
        m_weights(weights) {}

    virtual ~Attention() = default;

public:
    virtual TensorNode *build(
        Graph &g,
        TensorNode *x,
        int64_t L,
        const TensorNode *k_cache,
        const TensorNode *v_cache,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask,
        bool is_need_bias = false
    ) = 0;
};

} // namespace powerserve

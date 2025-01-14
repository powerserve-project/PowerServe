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

#include "graph/graph.hpp"
#include "model/common/weights.hpp"

namespace powerserve {

struct FFN {
private:
    const ModelConfig::LLMConfig &m_config;
    std::shared_ptr<Weight> m_weights;

public:
    FFN(const ModelConfig::LLMConfig &config, std::shared_ptr<Weight> weights) : m_config(config), m_weights(weights) {}

public:
    TensorNode *build(Graph &g, TensorNode *attn_o, int64_t L);
};

} // namespace powerserve

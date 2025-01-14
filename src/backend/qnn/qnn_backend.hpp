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

#include "backend/backend.hpp"
#include "causal_models.hpp"
#include "core/config.hpp"
#include "core/tensor.hpp"
#include "core/typedefs.hpp"
#include "graph_interface.hpp"
#include "qnn.hpp"

namespace powerserve::qnn {
struct QNNBackend : powerserve::Backend {
    Session m_session;
    std::map<std::string, std::unique_ptr<CausalLM>> m_models;
    std::map<std::string, std::unique_ptr<Vision>> m_visions;

    QNNBackend(Path libs_path);
    virtual ~QNNBackend() noexcept override = default;

    void load_model(const Path &path, const std::shared_ptr<powerserve::ModelConfig> &model_config);
    void unload_model(const std::shared_ptr<powerserve::ModelConfig> &model_config);

    void forward(
        const std::string &model_id,
        const Tensor *dst,
        const Tensor *src,
        const std::vector<int> &pos,
        const CausalAttentionMask &mask
    );
    void forward(
        const std::string &model_id,
        const Tensor *dst,
        const Tensor *src,
        const std::vector<std::vector<float>> &pixel_values_list,
        const std::vector<std::pair<int, size_t>> &img_infos,
        std::vector<int> &pos,
        const CausalAttentionMask &mask
    );
};

} // namespace powerserve::qnn

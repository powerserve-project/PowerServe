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

#include "platform.hpp"

namespace powerserve {

void Platform::init_ggml_backend(const std::shared_ptr<ModelConfig> &config, const HyperParams &hparams) {
    ggml_backends.insert({config->model_id, std::make_unique<ggml::GGMLBackend>(config->llm, hparams)});
}

void Platform::destroy_ggml_backend(const std::shared_ptr<ModelConfig> &config) {
    ggml_backends.erase(config->model_id);
}

#if defined(POWERSERVE_WITH_QNN)
void Platform::init_qnn_backend(const Path &qnn_path) {
    qnn_backend = std::make_unique<qnn::QNNBackend>(qnn_path);
}
#endif

size_t Platform::get_kv_position(std::string &model_id) const {
    size_t position = ggml_backends.at(model_id)->m_kv->kv_cache->position;
#if defined(POWERSERVE_WITH_QNN)
    if (qnn_backend) {
        position = qnn_backend->m_models[model_id]->kv_cache->position;
    }
#endif
    return position;
}

void Platform::reset_kv_position(std::string &model_id) {
    ggml_backends[model_id]->m_kv->reset_kv_cache();
#if defined(POWERSERVE_WITH_QNN)
    if (qnn_backend) {
        qnn_backend->m_models[model_id]->reset_kv_cache();
    }
#endif
}

} // namespace powerserve

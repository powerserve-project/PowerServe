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

#include "model/model_loader.hpp"

#include "model/internvl/internvl_model.hpp"
#include "model/llama/llama_model.hpp"
#include "model/qwen2/qwen2_model.hpp"

namespace powerserve {

auto load_model(const Path &model_dir) -> std::shared_ptr<Model> {
    std::shared_ptr<Model> out_model;
    auto out_config = std::make_shared<ModelConfig>(model_dir / MODEL_CONFIG_FILENAME);

    auto arch        = out_config->arch;
    auto weight_path = model_dir / MODEL_WEIGHTS_DIR / MODEL_WEIGHTS_FILENAME;
    if (arch == "llama") {
        out_model = std::make_shared<LlamaModel>(weight_path, out_config);
    } else if (arch == "qwen2") {
        out_model = std::make_shared<Qwen2Model>(weight_path, out_config);
    } else if (arch == "internvl") {
        out_model = std::make_shared<InternVL>(weight_path, out_config);
    } else {
        POWERSERVE_ABORT("unknown model type: {}", arch);
    }

    POWERSERVE_LOG_INFO("Load model {} ...", arch);
    return out_model;
}

} // namespace powerserve

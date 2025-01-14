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
#include "core/logger.hpp"
#include "nlohmann/json.hpp"

#include <fstream>

namespace powerserve::qnn {

const std::string QNN_WORKSPACE_DIR_NAME = "qnn";
const std::string QNN_LIB_DIR_NAME       = "qnn_libs";

struct QNNGraphConfig {
    std::string type;
    std::string graph_name;
    size_t batch_size;
    Path model_path;
    std::string x_name;
    std::string out_name;
    virtual ~QNNGraphConfig() = default;
};

struct ChunkConfig : QNNGraphConfig {
    size_t start_layer_id;
    size_t end_layer_id;
    size_t cache_size;
    size_t context_size;
    std::string kv_path_format;
    size_t head_dim;
    size_t kv_size;
};

struct VisionConfig : QNNGraphConfig {
    size_t image_size;
    size_t num_channels;
    size_t num_patches;
    size_t num_patches_tokens;
};

struct QNNConfig {

    size_t n_hvx_threads;
    float attention_mask_value;
    std::vector<QNNGraphConfig> lm_heads;
    const std::shared_ptr<ModelConfig> &model_config;
    std::vector<ChunkConfig> chunks;
    VisionConfig vision;

    QNNConfig(const Path &path, const std::shared_ptr<ModelConfig> &model_config) : model_config(model_config) {
        std::ifstream f(path);
        auto json = nlohmann::json::parse(f);
        {
            auto data = json.at("model_parameters");
            data.at("attention_mask_value").get_to(attention_mask_value);
        }
        {
            auto data = json.at("qnn_parameters");
            data.at("n_hvx_threads").get_to(n_hvx_threads);
        }

        {
            auto data_array = json.at("graphs");
            POWERSERVE_ASSERT(data_array.is_array());

            chunks.reserve(data_array.size());
            for (auto data : data_array) {
                ChunkConfig info;
                data.at("type").get_to(info.type);
                data.at("graph_name").get_to(info.graph_name);
                data.at("start_layer_id").get_to(info.start_layer_id);
                data.at("end_layer_id").get_to(info.end_layer_id);
                data.at("batch_size").get_to(info.batch_size);
                data.at("cache_size").get_to(info.cache_size);
                data.at("context_size").get_to(info.context_size);
                data.at("model_path").get_to(info.model_path);
                data.at("kv_path_format").get_to(info.kv_path_format);
                data.at("kv_size").get_to(info.head_dim);
                data.at("kv_size").get_to(info.kv_size);
                data.at("x_name").get_to(info.x_name);
                data.at("out_name").get_to(info.out_name);
                chunks.push_back(info);
            }
        }

        if (json.contains("embeddings")) {
            auto data_array = json.at("embeddings");
            lm_heads.reserve(data_array.size());
            for (auto data : data_array) {
                QNNGraphConfig info;
                data.at("graph_name").get_to(info.graph_name);
                data.at("batch_size").get_to(info.batch_size);
                data.at("model_path").get_to(info.model_path);
                data.at("x_name").get_to(info.x_name);
                data.at("out_name").get_to(info.out_name);
                lm_heads.push_back(info);
            }
        }
        if (json.contains("vision")) {
            auto &data = json.at("vision");
            data.at("graph_name").get_to(vision.graph_name);
            data.at("model_path").get_to(vision.model_path);
            data.at("num_channels").get_to(vision.num_channels);
            data.at("image_size").get_to(vision.image_size);
            data.at("num_patches").get_to(vision.num_patches);
            data.at("num_patches_tokens").get_to(vision.num_patches_tokens);
            data.at("x_name").get_to(vision.x_name);
            data.at("out_name").get_to(vision.out_name);
        }
    }
};

} // namespace powerserve::qnn

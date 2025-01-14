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

#include "core/config.hpp"

#include "core/exception.hpp"
#include "core/typedefs.hpp"
#include "nlohmann/json.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

namespace powerserve {

HyperParams::HyperParams(const Path &params_file) {
    nlohmann::json j;
    std::ifstream file(params_file);
    POWERSERVE_ASSERT_CONFIG(file.good(), "HyperConfig", "failed to open hparams config file {}", params_file);

    try {
        file >> j;

        n_threads  = j.value("n_threads", n_threads);
        batch_size = j.value("batch_size", batch_size);

        const uint32_t max_concurrency = std::thread::hardware_concurrency();
        if (max_concurrency != 0) {
            n_threads = std::min((unsigned int)n_threads, max_concurrency);
        }

        auto sampler_j = j.value("sampler", nlohmann::json::object());
        if (!sampler_j.empty()) {
            sampler_config.seed        = sampler_j.value("seed", sampler_config.seed);
            sampler_config.temperature = sampler_j.value("temperature", sampler_config.temperature);
            sampler_config.top_p       = sampler_j.value("top_p", sampler_config.top_p);
            sampler_config.top_k       = sampler_j.value("top_k", sampler_config.top_k);
            sampler_config.min_keep    = sampler_j.value("min_keep", sampler_config.min_keep);

            sampler_config.penalty_last_n  = sampler_j.value("penalty_last_n", sampler_config.penalty_last_n);
            sampler_config.penalty_repeat  = sampler_j.value("penalty_repeat", sampler_config.penalty_repeat);
            sampler_config.penalty_freq    = sampler_j.value("penalty_freq", sampler_config.penalty_freq);
            sampler_config.penalty_present = sampler_j.value("penalty_present", sampler_config.penalty_present);
            sampler_config.penalize_nl     = sampler_j.value("penalize_nl", sampler_config.penalize_nl);
            sampler_config.ignore_eos      = sampler_j.value("ignore_eos", sampler_config.ignore_eos);
        }
    } catch (const std::exception &err) {
        throw ConfigException(
            "HyperParams", fmt::format("failed parsing hyper param config file {}:\n{}", params_file, err.what())
        );
    }
}

ModelConfig::ModelConfig(const Path &model_config_file) {
    nlohmann::json j;
    std::ifstream file(model_config_file);
    POWERSERVE_ASSERT_CONFIG(file.good(), "ModelConfig", "failed to open model config file {}", model_config_file);

    try {
        file >> j;

        j["version"].get_to(version);
        j["model_arch"].get_to(arch);
        j["model_id"].get_to(model_id);
        {
            auto &llm_info = j.at("llm_config");
            llm_info["embed_dim"].get_to(llm.dim);
            llm_info["ffn_dim"].get_to(llm.hidden_dim);
            llm_info["n_layers"].get_to(llm.n_layers);
            llm_info["n_attn_heads"].get_to(llm.n_heads);
            llm_info["n_attn_kv_heads"].get_to(llm.n_kv_heads);
            llm_info["n_ctx"].get_to(llm.seq_len);
            llm_info["vocab_size"].get_to(llm.vocab_size);
            llm_info["kv_dim"].get_to(llm.kv_dim);
            llm_info["head_size"].get_to(llm.head_size);
            llm_info["norm_eps"].get_to(llm.norm_eps);
            {
                auto &rope_info = llm_info.at("rope_config");
                rope_info["rope_dim"].get_to(llm.rope_config.n_dims);
                rope_info["n_rope_ctx_orig"].get_to(llm.rope_config.n_ctx_orig);
                rope_info["rope_freq_base"].get_to(llm.rope_config.freq_base);
                rope_info["rope_freq_scale"].get_to(llm.rope_config.freq_scale);
                llm.rope_config.ext_factor = 0.0f; // TODO: depends on scale type
                rope_info["rope_attn_factor"].get_to(llm.rope_config.attn_factor);
                // TODO: config by user
                llm.rope_config.beta_fast = 32.0f;
                llm.rope_config.beta_slow = 0.0f;
                rope_info["rope_type"].get_to(llm.rope_config.rope_type);
            }
        }
        {
            if (j.contains("vision_config")) {
                auto &vision_config = j.at("vision_config");
                vision_config["embed_dim"].get_to(vision.embed_dim);
                vision_config["num_channels"].get_to(vision.in_chans);
                vision_config["image_size"].get_to(vision.image_size);
                vision_config["num_tokens_per_patch"].get_to(vision.num_tokens_per_patch);
                vision_config["num_patches"].get_to(vision.num_patches);
            }
        }
    } catch (const std::exception &err) {
        throw ConfigException(
            "ModelConfig", fmt::format("failed parsing model config file {}:\n{}", model_config_file, err.what())
        );
    }
}

Config::Config(const Path &work_folder, const Path &workspace_config_path) {
    POWERSERVE_ASSERT_CONFIG(
        std::filesystem::is_directory(work_folder), "Config", "work folder {} is not a directory", work_folder
    );
    std::ifstream file(workspace_config_path);
    POWERSERVE_ASSERT_CONFIG(file.good(), "Config", "failed to open workspace config file {}", workspace_config_path);

    nlohmann::json j;
    try {
        file >> j;
        if (j.contains(HYPER_PARAMS_FILENAME_KEY)) {
            hyper_params = HyperParams(work_folder / j[HYPER_PARAMS_FILENAME_KEY].get<std::string>());
        } else {
            hyper_params = HyperParams();
        }

        main_model_dir = "";
        if (j.contains(MAIN_MODEL_KEY)) {
            main_model_dir = work_folder / j[MAIN_MODEL_KEY].get<std::string>();
        }

        draft_model_dir = "";
        if (j.contains(DRAFT_MODEL_KEY)) {
            draft_model_dir = work_folder / j[DRAFT_MODEL_KEY].get<std::string>();
        }
    } catch (const std::exception &err) {
        throw ConfigException(
            "Config", fmt::format("failed parsing artifact config file {}:\n{}", workspace_config_path, err.what())
        );
    }
}
} // namespace powerserve

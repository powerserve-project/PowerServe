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

#include <cstdint>
#include <string>

namespace powerserve {

const std::string HYPER_PARAMS_FILENAME_KEY = "hparams_config";
const std::string MAIN_MODEL_KEY            = "model_main";
const std::string DRAFT_MODEL_KEY           = "model_draft";
const std::string MODEL_CONFIG_FILENAME     = "model.json";
const std::string MODEL_WEIGHTS_DIR         = "ggml";
const std::string MODEL_WEIGHTS_FILENAME    = "weights.gguf";
const std::string MODEL_VOCAB_FILENAME      = "vocab.gguf";
const std::string WORKSPACE_CONFIG_FILENAME = "workspace.json";

struct HyperParams {
    struct SamplerConfig {
        uint64_t seed     = -1; // -1 = randomly choose a new seed
        float temperature = 0.80f;
        float top_p       = 0.95f; // 1.0 = disabled
        size_t top_k      = 40;
        size_t min_keep   = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

        int penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
        float penalty_repeat  = 1.00f; // 1.0 = disabled
        float penalty_freq    = 0.00f; // 0.0 = disabled
        float penalty_present = 0.00f; // 0.0 = disabled
        bool penalize_nl      = false; // consider newlines as a repeatable token
        bool ignore_eos       = false;
    } sampler_config;

    size_t n_threads  = 4;
    size_t batch_size = 128;

    HyperParams() = default;
    HyperParams(const Path &params_file);

    ~HyperParams() noexcept = default;
};

struct SamplerConfig {
    uint64_t seed     = 0;
    float temperature = 0.80f;
    float top_p       = 0.95f; // 1.0 = disabled
    size_t top_k      = 40;
    size_t min_keep   = 0; // 0 = disabled, otherwise samplers should return at least min_keep tokens

    int penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float penalty_repeat  = 1.00f; // 1.0 = disabled
    float penalty_freq    = 0.00f; // 0.0 = disabled
    float penalty_present = 0.00f; // 0.0 = disabled
    bool penalize_nl      = false; // consider newlines as a repeatable token
    bool ignore_eos       = false;

    SamplerConfig() = default;

    ~SamplerConfig() noexcept = default;

    SamplerConfig(const Path &sampler_config_file);

    SamplerConfig(const SamplerConfig &other) = default;
};

struct ModelConfig {
    uint32_t version;
    std::string arch;
    std::string model_id;

    struct LLMConfig {
        struct RopeConfig {
            int n_dims        = 128; // rope_dim_count
            int n_ctx_orig    = 2048;
            float freq_base   = 10000.0f;
            float freq_scale  = 1.0f;
            float ext_factor  = 0.0f; // linear scaling factor, 1.0f for yarn
            float attn_factor = 1.0f;
            float beta_fast   = 32.0f;
            float beta_slow   = 0.0f;
            int rope_type     = -1;
        } rope_config;

        uint32_t dim        = 0; // n_embd
        uint32_t hidden_dim = 0;
        uint32_t n_layers   = 0;
        uint32_t n_heads    = 0;
        uint32_t n_kv_heads = 0;
        uint32_t seq_len    = 0; // n_ctx_orig in rope
        uint32_t vocab_size = 0;
        uint32_t kv_dim     = 0; // head_size * n_kv_heads
        uint32_t head_size  = 0; // dim / n_heads
        float norm_eps      = 1e-5f;
    } llm;

    struct VisionConfig {
        std::string model_id;
        uint32_t depth                = 0;
        uint32_t embed_dim            = 0;
        uint32_t mlp_ratio            = 0;
        uint32_t num_heads            = 0;
        uint32_t in_chans             = 0;
        uint32_t hidden_size          = 0;
        uint32_t patch_size           = 0;
        uint32_t spatial_merge_size   = 0;
        uint32_t spatial_patch_size   = 0;
        uint32_t temporal_patch_size  = 0;
        uint32_t image_size           = 0;
        uint32_t num_patches          = 0;
        uint32_t num_tokens_per_patch = 0;
    } vision;

    ModelConfig() = default;
    ModelConfig(const Path &model_config_file);

    ~ModelConfig() noexcept = default;
};

struct Config {
public:
    Path main_model_dir;
    Path draft_model_dir;

    HyperParams hyper_params;

public:
    Config() = default;

    Config(const Path &work_folder, const Path &workspace_config_path);

    ~Config() noexcept = default;
};

} // namespace powerserve

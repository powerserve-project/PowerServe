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

#include "CLI/CLI.hpp"
#include "core/logger.hpp"
#include "ggml.h"
#include "nlohmann/json.hpp"

#include <string>

constexpr int VERSION = 20241202;

enum class rope_type {
    NONE   = -1,
    NORMAL = 0,
    NEOX   = 2,
};

static uint32_t get_u32(gguf_context *ctx, const std::string &key, bool required = true, uint32_t default_value = 0) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        POWERSERVE_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_u32(ctx, idx);
}

static float get_f32(gguf_context *ctx, const std::string &key, bool required = true, float default_value = 0.0f) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        POWERSERVE_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_f32(ctx, idx);
}

static std::string get_str(
    gguf_context *ctx, const std::string &key, bool required = true, const std::string &default_value = ""
) {
    int idx = gguf_find_key(ctx, key.c_str());
    if (idx == -1) {
        POWERSERVE_ASSERT(required == false);
        return default_value;
    }
    return gguf_get_val_str(ctx, idx);
}

// see: llama_rope_type in llama.cpp
rope_type get_rope_type(std::string arch) {
    // TODO: wait for complement
    if (arch == "llama") {
        return rope_type::NORMAL;
    } else if (arch == "qwen" || arch == "phi3") {
        return rope_type::NEOX;
    }
    return rope_type::NONE;
}

void collect_config(gguf_context *gguf_ctx, nlohmann::json &config);

int main(int argc, char *argv[]) {
    std::string file_path   = "/home/zwb/Downloads/Llama-2-7b-chat-hf/llama-2-7b.f32.gguf";
    std::string target_path = "./llama-2-7b.json";
    CLI::App app("Config Generator");

    app.add_option("--file-path", file_path)->required();
    app.add_option("--target-path", target_path)->required();

    CLI11_PARSE(app, argc, argv);

    ggml_context *ggml_ctx;
    gguf_context *gguf_ctx;

    gguf_init_params params = {.no_alloc = false, .ctx = &ggml_ctx};
    gguf_ctx                = gguf_init_from_file(file_path.c_str(), params);
    POWERSERVE_ASSERT(gguf_ctx != nullptr);
    POWERSERVE_ASSERT(ggml_ctx != nullptr);

    nlohmann::json config;
    config["version"] = VERSION;

    collect_config(gguf_ctx, config);

    std::ofstream ofs(target_path);
    ofs << config.dump(4);

    gguf_free(gguf_ctx);
}

void collect_config(gguf_context *ctx, nlohmann::json &config) {
    std::string model_arch = gguf_get_val_str(ctx, gguf_find_key(ctx, "general.architecture"));
    config["model_arch"]   = model_arch;
    auto get_arch_config([&model_arch](const std::string &c) { return fmt::format(fmt::runtime(c), model_arch); });

    { // embed_dim, ffn_dim, n_heads, n_kv_heads, n_layers, n_ctx
        config["embed_dim"]       = get_u32(ctx, get_arch_config("{}.embedding_length"));
        config["ffn_dim"]         = get_u32(ctx, get_arch_config("{}.feed_forward_length"));
        config["n_attn_heads"]    = get_u32(ctx, get_arch_config("{}.attention.head_count"));
        config["n_attn_kv_heads"] = get_u32(ctx, get_arch_config("{}.attention.head_count_kv"));
        config["n_layers"]        = get_u32(ctx, get_arch_config("{}.block_count"));
        config["n_ctx"]           = get_u32(ctx, get_arch_config("{}.context_length"));
        config["head_size"]       = (uint32_t)config["embed_dim"] / (uint32_t)config["n_attn_heads"];
        config["kv_dim"]          = (uint32_t)config["head_size"] * (uint32_t)config["n_attn_kv_heads"];
    }
    { // vocab_size
        auto idx = gguf_find_key(ctx, get_arch_config("{}.vocab_size").c_str());
        if (idx != -1) {
            config["vocab_size"] = gguf_get_val_u32(ctx, idx);
        } else {
            idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
            POWERSERVE_ASSERT(idx != -1);
            config["vocab_size"] = gguf_get_arr_n(ctx, idx);
        }
    }
    { // kv_dim, norm_eps or norm_rms_eps
        auto idx      = gguf_find_key(ctx, get_arch_config("{}.attention.layer_norm_epsilon").c_str());
        auto norm_eps = 1e-5f;
        if (idx == -1) {
            idx = gguf_find_key(ctx, get_arch_config("{}.attention.layer_norm_rms_epsilon").c_str());
        }
        if (idx != -1) {
            norm_eps = gguf_get_val_f32(ctx, idx);
        }
        config["norm_eps"] = norm_eps;
    }
    { // rope
        config["rope_dim"] =
            get_u32(ctx, get_arch_config("{}.rope.dimension_count"), false, (uint32_t)config["head_size"]);
        config["rope_freq_base"]   = get_f32(ctx, get_arch_config("{}.rope.freq_base"), false, 10000.0f);
        auto scale_type            = get_str(ctx, get_arch_config("{}.rope.scaling.type"), false, "linear");
        config["rope_scale_type"]  = scale_type;
        config["rope_attn_factor"] = get_f32(ctx, get_arch_config("{}.rope.scaling.attn_factor"), false, 1.0f);
        config["n_rope_ctx_orig"] =
            get_u32(ctx, get_arch_config("{}.rope.scaling.original_context_length"), false, (uint32_t)config["n_ctx"]);

        auto rope_scale = 0.0f;
        auto idx        = gguf_find_key(ctx, get_arch_config("{}.rope.scaling.factor").c_str());
        if (idx == -1) {
            idx = gguf_find_key(ctx, get_arch_config("{}.rope.scale_linear").c_str());
        }
        if (idx != -1) {
            rope_scale = gguf_get_val_f32(ctx, idx);
        }
        rope_scale                = rope_scale == 0.0f ? 1.0f : 1.0f / rope_scale;
        config["rope_freq_scale"] = rope_scale;
        config["rope_type"]       = get_rope_type(model_arch);
    }
}

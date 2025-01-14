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
#include "core/kv_cache.hpp"
#include "core/tensor.hpp"

namespace powerserve::ggml {

struct GGMLKV {
public:
    using KVBuffer = std::vector<std::vector<float>>;

    size_t m_kv_dim     = 0;
    size_t m_n_kv_heads = 0;
    size_t m_n_ctx      = 0;
    size_t m_n_layers   = 0;
    size_t m_head_size  = 0;
    size_t m_batch_size = 0;
    size_t kv_size      = 0; // system_prompt size
    const ModelConfig::LLMConfig &m_config;

    struct GGMLChunk {
        KVBuffer key_buffer;          // [n_layers][seq_len * kv_dim]) kv_dim == n_kv_heads * head_size
        KVBuffer value_buffer;        // [n_layers][seq_len * kv_dim])
        KVBuffer current_k;           // [n_layers][batch_size * kv_dim])
        KVBuffer current_v;           // [n_layers][batch_size * kv_dim])
        std::vector<float> attn_bias; // [batch_size * n_ctx]

        std::vector<Tensor> key_tensors;   // n_layers
        std::vector<Tensor> value_tensors; // n_layers
    };

    GGMLChunk chunk;

    struct GGMLKVInterface {
        GGMLKV &parent;
        GGMLChunk &chunk;

        GGMLKVInterface(GGMLKV &parent, GGMLChunk &chunk) : parent(parent), chunk(chunk) {}

        // Note: get entry from temporary kv
        ALWAYS_INLINE auto get_key(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_k[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto get_value(KVPosition token_pos) const -> KVView {
            auto &chk       = chunk.current_v[token_pos.layer_id];
            auto buffer     = chk.data() + token_pos.index * parent.m_kv_dim + token_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        };

        // Note: get entry from KV cache
        ALWAYS_INLINE auto key_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = chunk.key_buffer[cache_pos.layer_id];
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE auto value_entry(KVPosition cache_pos) const -> KVView {
            auto &chk       = chunk.value_buffer[cache_pos.layer_id];
            auto buffer     = chk.data() + cache_pos.index * parent.m_kv_dim + cache_pos.head_id * parent.m_head_size;
            size_t n_elem   = parent.m_head_size;
            size_t n_stride = sizeof(float); // TODO: transpose will change stride

            return {
                .n_elements   = n_elem,
                .element_size = sizeof(float),
                .stride       = n_stride,
                .data         = buffer,
            };
        }

        ALWAYS_INLINE void set_mask(size_t cache_index, bool mask) {
            for (size_t i = 0; i < parent.m_batch_size; i++) {
                auto attn_bias         = chunk.attn_bias.data() + i * parent.m_n_ctx;
                attn_bias[cache_index] = mask ? -INFINITY : 0;
            }
        }
    };

public:
    std::unique_ptr<KVCache<GGMLKVInterface>> kv_cache;

public:
    GGMLKV(const ModelConfig::LLMConfig &config);

    ~GGMLKV() = default;

public:
    void reset_batch_size(const size_t &batch_size) {
        if (m_batch_size == batch_size)
            return;
        // fmt::println("Resize batch size: {} -> {}", m_batch_size, batch_size);
        m_batch_size = batch_size;

        auto &k = chunk.current_k;
        auto &v = chunk.current_v;
        for (size_t L = 0; L < m_n_layers; L++) {
            k[L].resize(m_batch_size * m_kv_dim);
            v[L].resize(m_batch_size * m_kv_dim);
        }
        chunk.attn_bias.resize(m_batch_size * m_n_ctx);
    }

    void reset_kv_cache() {
        kv_cache->truncate_tokens(kv_size);
    }

    void save_kv(int size) {
        kv_cache->save_tokens(size);
    }

    void advance(int size) {
        kv_cache->advance_tokens(size);
    }

    auto get_cache(size_t L) -> std::pair<Tensor &, Tensor &> {
        return {chunk.key_tensors[L], chunk.value_tensors[L]};
    }

private:
    void prepare_model_chunk();
};

} // namespace powerserve::ggml

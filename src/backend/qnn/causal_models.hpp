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

#include "graph_interface.hpp"
#include "model/module/attention_mask.hpp"

#include <map>
#include <span>

namespace powerserve::qnn {

struct CausalLM {
    static constexpr const char *m_config_file_name = "config.json";

    Path m_model_folder;
    QNNConfig m_config;
#if defined(QNN_TIMER)
    int64_t m_excute_time_ns{};
#endif
    const std::shared_ptr<ModelConfig> m_model_config;

    struct {
        size_t max_batch_size = 0;
        size_t context_size   = 0;
        size_t cache_size     = 0;
        size_t kv_size        = 0;
    } m_gparams;

    Session &m_session;

    struct RopeEmbedding {
        std::vector<float> cos_values;
        std::vector<float> sin_values;
    };

    std::vector<RopeEmbedding> m_rope_embeds;

    std::map<Path, ContextBinary> m_context_binaries;

    struct ChunkVector : std::vector<std::unique_ptr<ModelChunk>> {
        auto get_chunk(size_t layer_id) -> ModelChunk &;
    };

    struct CausalLMKV {
        CausalLM &parent;
        ChunkVector &chunks;

        CausalLMKV(CausalLM &parent, ChunkVector &chunks) : parent(parent), chunks(chunks) {}

        ALWAYS_INLINE auto get_key(KVPosition token_pos) const -> KVView {
            auto &chunk  = chunks.get_chunk(token_pos.layer_id);
            auto &buffer = *chunk.m_buffers.at(fmt::format("layer_{}_key_{}", token_pos.layer_id, token_pos.head_id));
            POWERSERVE_ASSERT(token_pos.index < chunk.m_config.batch_size);

            return {
                .n_elements   = chunk.m_model_config.llm.head_size,
                .element_size = chunk.kv_element_size,
                .stride       = chunk.kv_element_size,
                .data         = (char *)buffer.m_data +
                        token_pos.index * chunk.m_model_config.llm.head_size * chunk.kv_element_size,
            };
        }

        ALWAYS_INLINE auto get_value(KVPosition token_pos) const -> KVView {
            auto &chunk  = chunks.get_chunk(token_pos.layer_id);
            auto &buffer = *chunk.m_buffers.at(fmt::format("layer_{}_value_{}", token_pos.layer_id, token_pos.head_id));
            POWERSERVE_ASSERT(token_pos.index < chunk.m_config.batch_size);

            return {
                .n_elements   = chunk.m_model_config.llm.head_size,
                .element_size = chunk.kv_element_size,
                .stride       = chunk.kv_element_size,
                .data         = (char *)buffer.m_data +
                        token_pos.index * chunk.m_model_config.llm.head_size * chunk.kv_element_size,
            };
        }

        ALWAYS_INLINE auto key_entry(KVPosition cache_pos) const -> KVView {
            auto &chunk = chunks.get_chunk(cache_pos.layer_id);
            auto &buffer =
                *chunk.m_buffers.at(fmt::format("layer_{}_key_t_cache_{}", cache_pos.layer_id, cache_pos.head_id));
            POWERSERVE_ASSERT(cache_pos.index < chunk.m_config.cache_size);

            return {
                .n_elements   = chunk.m_model_config.llm.head_size,
                .element_size = chunk.kv_element_size,
                .stride       = chunk.kv_element_size * chunk.m_config.cache_size,
                .data         = (char *)buffer.m_data + cache_pos.index * chunk.kv_element_size,
            };
        }

        ALWAYS_INLINE auto value_entry(KVPosition cache_pos) const -> KVView {
            auto &chunk = chunks.get_chunk(cache_pos.layer_id);
            auto &buffer =
                *chunk.m_buffers.at(fmt::format("layer_{}_value_cache_{}", cache_pos.layer_id, cache_pos.head_id));
            POWERSERVE_ASSERT(cache_pos.index < chunk.m_config.cache_size);

            return {
                .n_elements   = chunk.m_model_config.llm.head_size,
                .element_size = chunk.kv_element_size,
                .stride       = chunk.kv_element_size,
                .data         = (char *)buffer.m_data +
                        cache_pos.index * chunk.m_model_config.llm.head_size * chunk.kv_element_size,
            };
        }

        ALWAYS_INLINE void set_mask(size_t cache_index, bool mask) {
            __fp16 fill_value = mask ? parent.m_config.attention_mask_value : 0;
            POWERSERVE_ASSERT(cache_index < chunks[0]->m_config.cache_size);

            for (auto &chunk : chunks) {
                for (size_t i = 0; i < chunk->m_config.batch_size; i++) {
                    auto attn_bias = (__fp16 *)chunk->m_buffers["attn_bias"]->m_data + i * chunk->m_config.context_size;
                    attn_bias[cache_index] = fill_value;
                }
            }
        }
    };

    std::map<size_t, ChunkVector> m_chunks_map;

    std::map<size_t, std::unique_ptr<Embedding>> m_lm_heads;

    std::unique_ptr<KVCache<CausalLMKV>> kv_cache;

    CausalLM(const Path &model_folder, const std::shared_ptr<ModelConfig> &model_config, Session &environment);
    virtual ~CausalLM() = default;
    auto load_context_binary(const Path &path) -> ContextBinary &;
    void load_model_chunks();

    auto largest_chunks() -> ChunkVector & {
        return m_chunks_map.rbegin()->second;
    }

    void compute_rope_embeds();
    void fill_rope_embeds(std::span<const size_t> pos);
    void fill_attention_mask(AttentionMaskView mask);
    void reset_kv_cache();

    struct Batch {
        CausalLM &parent;
        std::span<const float> token_embeddings;
        std::span<const size_t> pos;
        AttentionMaskView mask;
        ChunkVector &chunks;
        // Embedding &lm_head;
        Embedding *lm_head;

        void forward();
        void compute_logits();
        void save_kv();
        void advance();
    };

    auto split_batch(
        std::span<const float> token_embeddings, std::span<const size_t> pos, const CausalAttentionMask &mask
    ) -> std::vector<Batch>;

    auto create_batch(
        std::span<const float> token_embeddings, std::span<const size_t> pos, const CausalAttentionMask &mask
    ) -> Batch;
};

struct CausalVLM : CausalLM {
    std::unique_ptr<Vision> m_vision;

    CausalVLM(const Path model_folder, const std::shared_ptr<ModelConfig> &model_config, Session &environment);
    virtual ~CausalVLM() override = default;
};

} // namespace powerserve::qnn

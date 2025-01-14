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

#include "causal_models.hpp"

#include "core/logger.hpp"
#include "core/perfetto_trace.hpp"

namespace powerserve::qnn {

auto CausalLM::ChunkVector::get_chunk(size_t layer_id) -> ModelChunk & {
    for (auto &chunk_ptr : *this) {
        if (chunk_ptr->m_config.start_layer_id <= layer_id && layer_id < chunk_ptr->m_config.end_layer_id) {
            return *chunk_ptr;
        }
    }

    POWERSERVE_ABORT("cannot found mode chunk containing layer: {}", layer_id);
}

CausalLM::CausalLM(const Path &model_folder, const std::shared_ptr<ModelConfig> &model_config, Session &environment) :
    m_model_folder(model_folder),
    m_config(model_folder / m_config_file_name, model_config),
    m_model_config(model_config),
    m_session(environment) {
    m_gparams.cache_size   = m_config.chunks[0].cache_size;
    m_gparams.kv_size      = m_config.chunks[0].kv_size;
    m_gparams.context_size = m_config.chunks[0].context_size;
    for (auto &info : m_config.chunks) {
        POWERSERVE_ASSERT(info.cache_size == m_gparams.cache_size);
        POWERSERVE_ASSERT(info.kv_size == m_gparams.kv_size);
        POWERSERVE_ASSERT(info.context_size == m_gparams.context_size);
        m_gparams.max_batch_size = std::max(m_gparams.max_batch_size, info.batch_size);
    }
    load_model_chunks();
    if (!m_config.lm_heads.empty()) {
        for (auto &config : m_config.lm_heads) {
            m_lm_heads.emplace(config.batch_size, std::make_unique<Embedding>(*this, config));
        }
        auto max_lm_head_ptr   = m_lm_heads.at(m_gparams.max_batch_size).get();
        auto &max_lm_head      = *max_lm_head_ptr;
        auto &context_binary   = load_context_binary(max_lm_head.m_graph_config.model_path);
        context_binary.m_alloc = std::make_unique<SharedBufferAllocator>(max_lm_head.io_tensor_size());
        max_lm_head.setup_buffers();
        POWERSERVE_ASSERT(context_binary.m_alloc->unallocated_size() == 0);
        context_binary.m_context->free_system_context();
        for (auto &[batch_size, lm_head_ptr] : m_lm_heads) {
            if (batch_size == max_lm_head.m_graph_config.batch_size) {
                continue;
            }
            auto &lm_head = *lm_head_ptr;
            POWERSERVE_ASSERT(lm_head.io_tensor_size() <= max_lm_head.io_tensor_size());
            lm_head.m_sibling = &max_lm_head;
            lm_head.setup_buffers();
        }
    }

    compute_rope_embeds();
}

auto CausalLM::load_context_binary(const Path &path) -> ContextBinary & {
    auto iter = m_context_binaries.find(path);
    if (iter != m_context_binaries.end()) {
        return iter->second;
    }

    POWERSERVE_LOG_INFO("Loading \"{}\"...", path);
    auto result          = m_context_binaries.emplace(path, ContextBinary(*m_session.m_backend, m_model_folder / path));
    auto &context_binary = result.first->second;

    if (m_context_binaries.size() == 1) {
        context_binary.m_context->print_info();
    }

    return context_binary;
}

void CausalLM::load_model_chunks() {
    std::vector<ChunkConfig *> chunk_configs;
    chunk_configs.reserve(m_config.chunks.size());
    for (auto &config : m_config.chunks) {
        chunk_configs.push_back(&config);
    }

    auto cmp = [](ChunkConfig *a, ChunkConfig *b) {
        return a->batch_size == b->batch_size ? a->start_layer_id < b->start_layer_id : a->batch_size > b->batch_size;
    };
    std::sort(chunk_configs.begin(), chunk_configs.end(), cmp);
    std::unique_ptr<SharedBufferAllocator> dummy_alloc;
    std::unique_ptr<SharedBuffer> dummy_buffer;
    constexpr size_t dummy_sizes[] = {1024 * 1024 * 512, 1024 * 1024 * 256, 1024 * 1024 * 128};
    for (auto dummy_size : dummy_sizes) {
        try {
            dummy_alloc          = std::make_unique<SharedBufferAllocator>(dummy_size);
            auto &context_binary = load_context_binary(chunk_configs[0]->model_path);
            dummy_buffer =
                std::make_unique<SharedBuffer>(*context_binary.m_context, *dummy_alloc, QNN_DATATYPE_INT_8, dummy_size);
            break;
        } catch (const std::runtime_error &e) {
            dummy_alloc.reset(nullptr);
            dummy_buffer.reset(nullptr);
        }
    }
    for (auto config : chunk_configs) {
        auto &chunks = m_chunks_map[config->batch_size];
        chunks.emplace_back(std::make_unique<ModelChunk>(*this, *config));
    }

    POWERSERVE_ASSERT(m_chunks_map.find(m_gparams.max_batch_size) != m_chunks_map.end());

    auto &max_chunks = m_chunks_map[m_gparams.max_batch_size];
    kv_cache         = std::make_unique<KVCache<CausalLMKV>>(
        m_model_config->llm.n_layers, m_model_config->llm.n_kv_heads, m_gparams.cache_size, *this, max_chunks
    );
    dummy_buffer.reset(nullptr);
    dummy_alloc.reset(nullptr);
    for (size_t i = 0; i < max_chunks.size(); i++) {
        auto &max_chunk = *max_chunks[i];

        auto &context_binary   = load_context_binary(max_chunk.m_config.model_path);
        size_t buf_size        = max_chunk.io_tensor_size();
        context_binary.m_alloc = std::make_unique<SharedBufferAllocator>(buf_size);

        max_chunk.initialize(*kv_cache);

        POWERSERVE_ASSERT(context_binary.m_alloc->unallocated_size() == 0);
        context_binary.m_context->free_system_context();

        for (auto &[batch_size, chunks] : m_chunks_map) {
            if (batch_size == max_chunk.m_config.batch_size) {
                continue;
            }

            POWERSERVE_ASSERT(i < chunks.size());
            auto &chunk = *chunks[i];
            POWERSERVE_ASSERT(chunk.m_config.start_layer_id == max_chunk.m_config.start_layer_id);
            POWERSERVE_ASSERT(chunk.m_config.end_layer_id == max_chunk.m_config.end_layer_id);
            POWERSERVE_ASSERT(chunk.m_config.kv_size == max_chunk.m_config.kv_size);
            POWERSERVE_ASSERT(chunk.io_tensor_size() <= max_chunk.io_tensor_size());

            chunk.m_sibling = &max_chunk;
            chunk.initialize(*kv_cache);
        }
    }

    kv_cache->advance_tokens(m_gparams.kv_size);
}

void CausalLM::compute_rope_embeds() {
    // TODO: Only support linear ROPE now
    auto head_dim = m_model_config->llm.head_size;
    std::vector<float> inv_freqs(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freqs[i] = 1.0f / std::pow(m_model_config->llm.rope_config.freq_base, 2.0f * i / head_dim);
    }

    m_rope_embeds.resize(m_gparams.context_size);
    for (size_t i = 0; i < m_gparams.context_size; i++) {
        m_rope_embeds[i].cos_values.resize(head_dim / 2);
        m_rope_embeds[i].sin_values.resize(head_dim / 2);

        for (size_t j = 0; j < head_dim / 2; j++) {
            float freq                     = i * inv_freqs[j];
            m_rope_embeds[i].cos_values[j] = std::cos(freq);
            m_rope_embeds[i].sin_values[j] = std::sin(freq);
        }
    }
}

void CausalLM::fill_rope_embeds(std::span<const size_t> pos) {
    // TODO: Only support linear ROPE now
    auto head_dim = m_model_config->llm.head_size;
    for (auto &chunk_ptr : largest_chunks()) {
        auto &chunk = *chunk_ptr;
        POWERSERVE_ASSERT(pos.size() <= chunk.m_config.batch_size);

        for (size_t i = 0; i < pos.size(); i++) {
            size_t p = pos[i];
            POWERSERVE_ASSERT(p < m_rope_embeds.size());

            auto &src = m_rope_embeds[p];
            memcpy(
                (float *)chunk.m_buffers.at("rope_embed_cos")->m_data + i * head_dim / 2,
                src.cos_values.data(),
                sizeof(src.cos_values[0]) * src.cos_values.size()
            );
            memcpy(
                (float *)chunk.m_buffers.at("rope_embed_sin")->m_data + i * head_dim / 2,
                src.sin_values.data(),
                sizeof(src.sin_values[0]) * src.sin_values.size()
            );
        }
    }
}

void CausalLM::fill_attention_mask(AttentionMaskView mask) {
    __fp16 mask_value = m_config.attention_mask_value;

    for (auto &chunk_ptr : largest_chunks()) {
        auto &chunk = *chunk_ptr;
        for (size_t i = 0; i < mask.size; i++) {
            auto attn_bias = (__fp16 *)chunk.m_buffers.at("attn_bias")->m_data + i * m_gparams.context_size;
            for (size_t j = 0; j < mask.size; j++) {
                attn_bias[m_gparams.cache_size + j] = mask.not_masked(i, j) ? 0 : mask_value;
            }
            for (size_t j = mask.size; j < m_gparams.max_batch_size; j++) {
                attn_bias[m_gparams.cache_size + j] = mask_value;
            }
        }
    }
}

void CausalLM::reset_kv_cache() {
    kv_cache->truncate_tokens(m_gparams.kv_size);
}

void CausalLM::Batch::forward() {
    size_t batch_size = pos.size();

    for (size_t i = 0; i < batch_size; i++) {
        auto buffer = (float *)chunks[0]->input_buffer() + i * parent.m_model_config->llm.dim;
        memcpy(
            buffer,
            token_embeddings.data() + i * parent.m_model_config->llm.dim,
            sizeof(float) * parent.m_model_config->llm.dim
        );
    }

    parent.fill_rope_embeds(pos);
    parent.fill_attention_mask(mask);

    for (size_t i = 0; i < chunks.size(); i++) {
#if defined(QNN_TIMER)
        chunks[i]->execute(parent.m_excute_time_ns);
#else
        chunks[i]->execute();
#endif
        if (i + 1 < chunks.size()) {
            memcpy(
                chunks[i + 1]->input_buffer(),
                chunks[i]->output_buffer(),
                batch_size * parent.m_model_config->llm.dim * sizeof(float)
            );
        }
    }
}

void CausalLM::Batch::compute_logits() {
    PerfettoTrace::begin("qnn_compute_logits");

    POWERSERVE_ASSERT(lm_head != nullptr);
    size_t batch_size = pos.size();

    memcpy(
        lm_head->input_buffer(),
        chunks.back()->output_buffer(),
        batch_size * parent.m_model_config->llm.dim * sizeof(float)
    );
#if defined(QNN_TIMER)
    lm_head->execute(parent.m_excute_time_ns);
#else
    lm_head->execute();
#endif

    PerfettoTrace::end();
}

void CausalLM::Batch::save_kv() {
    parent.kv_cache->save_tokens(pos.size());
}

void CausalLM::Batch::advance() {
    parent.kv_cache->advance_tokens(pos.size());
}

auto CausalLM::split_batch(
    std::span<const float> token_embeddings, std::span<const size_t> pos, const CausalAttentionMask &mask
) -> std::vector<Batch> {
    auto embedding_dim = m_model_config->llm.dim;
    size_t batch_size  = token_embeddings.size() / embedding_dim;
    POWERSERVE_ASSERT(pos.size() == batch_size);
    POWERSERVE_ASSERT(mask.size == batch_size);

    std::vector<Batch> batches;
    for (size_t index = 0, step_size; index < batch_size; index += step_size) {
        size_t n_remain = batch_size - index;
        auto it_chunk   = m_chunks_map.lower_bound(n_remain);
        auto &chunks    = it_chunk == m_chunks_map.end() ? largest_chunks() : it_chunk->second;
        step_size       = std::min(n_remain, chunks[0]->m_config.batch_size);
        if (m_lm_heads.size() > 0) {
            auto it_lm_head = m_lm_heads.lower_bound(n_remain);
            auto &lm_head_ptr =
                it_lm_head == m_lm_heads.end() ? m_lm_heads.at(m_gparams.max_batch_size) : it_lm_head->second;
            auto lm_head = lm_head_ptr.get();
            batches.emplace_back(
                *this,
                token_embeddings.subspan(index * embedding_dim, step_size * embedding_dim),
                pos.subspan(index, step_size),
                AttentionMaskView(mask, index, step_size),
                chunks,
                lm_head
            );
        } else {
            batches.emplace_back(
                *this,
                token_embeddings.subspan(index * embedding_dim, step_size * embedding_dim),
                pos.subspan(index, step_size),
                AttentionMaskView(mask, index, step_size),
                chunks,
                nullptr
            );
        }
    }

    return batches;
}

auto CausalLM::create_batch(
    std::span<const float> token_embeddings, std::span<const size_t> pos, const CausalAttentionMask &mask
) -> Batch {
    auto batches = split_batch(token_embeddings, pos, mask);
    POWERSERVE_ASSERT(batches.size() == 1);
    return batches[0];
}

CausalVLM::CausalVLM(const Path model_folder, const std::shared_ptr<ModelConfig> &model_config, Session &environment) :
    CausalLM(model_folder, model_config, environment) {
    m_vision               = std::make_unique<Vision>(*this, m_config.vision);
    auto &context_binary   = load_context_binary(m_config.vision.model_path);
    context_binary.m_alloc = std::make_unique<SharedBufferAllocator>(m_vision->io_tensor_size());
    m_vision->setup_buffers();
    POWERSERVE_ASSERT(context_binary.m_alloc->unallocated_size() == 0);
    context_binary.m_context->free_system_context();
}
} // namespace powerserve::qnn

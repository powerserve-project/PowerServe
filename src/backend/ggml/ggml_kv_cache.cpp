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

#include "backend/ggml/ggml_kv_cache.hpp"

#include "backend/cpu_buffer.hpp"

namespace powerserve::ggml {

GGMLKV::GGMLKV(const ModelConfig::LLMConfig &config) :
    m_kv_dim(config.kv_dim),
    m_n_kv_heads(config.n_kv_heads),
    m_n_ctx(config.seq_len),
    m_n_layers(config.n_layers),
    m_head_size(config.head_size),
    m_batch_size(1), // FIXME:
    m_config(config) {

    prepare_model_chunk();

    kv_cache = std::make_unique<KVCache<GGMLKVInterface>>(m_n_layers, m_n_kv_heads, m_n_ctx, *this, chunk);
}

void GGMLKV::prepare_model_chunk() {
    auto &key_buffer   = chunk.key_buffer;
    auto &value_buffer = chunk.value_buffer;
    auto &k            = chunk.current_k;
    auto &v            = chunk.current_v;

    key_buffer.resize(m_n_layers);
    value_buffer.resize(m_n_layers);
    size_t layer_size = m_kv_dim * m_n_ctx;
    for (size_t L = 0; L < m_n_layers; L++) {
        key_buffer[L].reserve(layer_size);
        value_buffer[L].reserve(layer_size);

        chunk.key_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
        chunk.value_tensors.emplace_back(Tensor(DataType::FP32, {m_n_ctx, m_kv_dim, 1, 1}));
        Stride stride = {
            sizeof(float),
            sizeof(float) * m_n_ctx,
            sizeof(float) * m_kv_dim * m_n_ctx,
            sizeof(float) * m_kv_dim * m_n_ctx
        };
        chunk.key_tensors[L].m_data   = std::make_shared<CPUBuffer>(stride, key_buffer[L].data());
        chunk.value_tensors[L].m_data = std::make_shared<CPUBuffer>(stride, value_buffer[L].data());
    }

    k.resize(m_n_layers);
    v.resize(m_n_layers);
    for (size_t L = 0; L < m_n_layers; L++) {
        k[L].reserve(m_batch_size * m_kv_dim);
        v[L].reserve(m_batch_size * m_kv_dim);
    }

    auto &attn_bias = chunk.attn_bias;
    attn_bias.reserve(m_batch_size * m_n_ctx);
}

} // namespace powerserve::ggml

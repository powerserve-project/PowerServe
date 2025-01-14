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

#include "graph_interface.hpp"

#include "causal_models.hpp"
#include "fmt/base.h"
#include "storage/file_loader.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace powerserve::qnn {

GraphInterface::GraphInterface(CausalLM &parent, const QNNGraphConfig &config) :
    m_parent(parent),
    m_graph_config(config),
    m_model_config(*parent.m_model_config),
    m_context_binary(m_parent.load_context_binary(config.model_path)) {}

void GraphInterface::setup_tensor(
    const std::string &tensor_name, const std::vector<size_t> &tensor_shape, const Qnn_DataType_t tensor_datatype
) {
    m_tensors.insert({tensor_name, m_graph->get_tensor(tensor_name)->check(tensor_shape, tensor_datatype)});
}

auto GraphInterface ::io_tensor_size() const -> size_t {
    size_t size = 0;
    for (auto &tensor : m_tensors) {
        size += tensor.second->size();
    }
    return size;
}

void GraphInterface::setup_buffer(std::shared_ptr<SharedBuffer> &buffer, qnn::QNNTensor *tensor) {
    if (!buffer) {
        buffer = std::make_shared<SharedBuffer>(
            *m_context_binary.m_context, *m_context_binary.m_alloc, tensor->type(), tensor->n_elements()
        );
    }
    tensor->setup_shared_buffer(*buffer);
}

auto GraphInterface::input_buffer() const -> void * {
    POWERSERVE_ASSERT(m_buffers.at(m_graph_config.x_name)->m_type == QNN_DATATYPE_FLOAT_32);
    return (void *)m_buffers.at(m_graph_config.x_name)->m_data;
}

auto GraphInterface::output_buffer() const -> void * {
    POWERSERVE_ASSERT(m_buffers.at(m_graph_config.out_name)->m_type == QNN_DATATYPE_FLOAT_32);
    return (void *)m_buffers.at(m_graph_config.out_name)->m_data;
}

#if defined(QNN_TIMER)
void GraphInterface::execute(int64_t &time) const {
    auto start = clock::now();
    m_graph->execute();
    auto end = clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}
#else
void GraphInterface::execute() const {
    m_graph->execute();
}
#endif

Embedding::Embedding(CausalLM &parent, QNNGraphConfig &info) : GraphInterface(parent, info) {
    m_graph = std::make_unique<Graph>(*m_context_binary.m_context, info.graph_name);
    m_graph->set_n_hvx_threads(m_parent.m_config.n_hvx_threads);
    setup_tensors();
}

void Embedding::setup_tensors() {
    setup_tensor("x", {m_graph_config.batch_size, m_model_config.llm.dim}, QNN_DATATYPE_FLOAT_32);
    setup_tensor("logits", {m_graph_config.batch_size, m_model_config.llm.vocab_size}, QNN_DATATYPE_FLOAT_32);
}

void Embedding::setup_buffers() {
    if (m_sibling) {
        m_buffers = m_sibling->m_buffers;
    }
    setup_buffer(m_buffers["x"], m_tensors.at("x"));
    setup_buffer(m_buffers["logits"], m_tensors.at("logits"));
}

Vision::Vision(CausalLM &parent, VisionConfig info) : GraphInterface(parent, info), m_config(info) {
    m_graph = std::make_unique<Graph>(*m_context_binary.m_context, info.graph_name);
    m_graph->set_n_hvx_threads(m_parent.m_config.n_hvx_threads);
    setup_tensors();
}

void Vision::setup_tensors() {
    setup_tensor(
        "pixel_values",
        {m_config.num_patches, m_config.num_channels, m_config.image_size, m_config.image_size},
        QNN_DATATYPE_FLOAT_32
    );
    setup_tensor(
        "image_embeddings",
        {m_config.num_patches, m_config.num_patches_tokens, m_model_config.llm.dim},
        QNN_DATATYPE_FLOAT_32
    );
}

void Vision::setup_buffers() {
    setup_buffer(m_buffers["pixel_values"], m_tensors["pixel_values"]);
    setup_buffer(m_buffers["image_embeddings"], m_tensors["image_embeddings"]);
}

ModelChunk::ModelChunk(CausalLM &parent, ChunkConfig &info) : GraphInterface(parent, info), m_config(info) {
    m_graph = std::make_unique<Graph>(*m_context_binary.m_context, m_config.graph_name);
    m_graph->set_n_hvx_threads(m_parent.m_config.n_hvx_threads);

    setup_tensors();
}

auto ModelChunk::n_layers() const -> size_t {
    return m_config.end_layer_id - m_config.start_layer_id;
}

void ModelChunk::setup_tensors() {
    auto head_dim = m_model_config.llm.head_size;
    setup_tensor("x", {m_config.batch_size, m_model_config.llm.dim}, QNN_DATATYPE_FLOAT_32);
    setup_tensor("attn_bias", {m_config.batch_size, m_config.context_size}, QNN_DATATYPE_FLOAT_16);
    setup_tensor("rope_embed_cos", {m_config.batch_size, head_dim / 2}, QNN_DATATYPE_FLOAT_32);
    setup_tensor("rope_embed_sin", {m_config.batch_size, head_dim / 2}, QNN_DATATYPE_FLOAT_32);

    for (size_t i = 0; i < n_layers(); i++) {
        for (size_t j = 0; j < m_model_config.llm.n_kv_heads; j++) {
            setup_tensor(
                fmt::format("layer_{}_key_t_cache_{}", m_config.start_layer_id + i, j),
                {head_dim, m_config.cache_size},
                kv_type
            );
            setup_tensor(
                fmt::format("layer_{}_value_cache_{}", m_config.start_layer_id + i, j),
                {m_config.cache_size, head_dim},
                kv_type
            );
        }
    }
    setup_tensor("out", {m_config.batch_size, m_model_config.llm.dim}, QNN_DATATYPE_FLOAT_32);
    for (size_t i = 0; i < n_layers(); i++) {
        for (size_t j = 0; j < m_model_config.llm.n_kv_heads; j++) {
            setup_tensor(
                fmt::format("layer_{}_key_{}", m_config.start_layer_id + i, j),
                {m_config.batch_size, head_dim},
                QNN_DATATYPE_FLOAT_16
            );
            setup_tensor(
                fmt::format("layer_{}_value_{}", m_config.start_layer_id + i, j),
                {m_config.batch_size, head_dim},
                QNN_DATATYPE_FLOAT_16
            );
        }
    }
}

void ModelChunk::initialize(KVCacheInterface &kv_cache) {
    setup_buffers();

    if (!m_sibling) {
        load_kv(kv_cache);
    }

    // Initialize attn_bias with mask values
    auto attn_bias = (__fp16 *)m_buffers["attn_bias"]->m_data;
    std::fill(attn_bias, attn_bias + m_tensors["attn_bias"]->n_elements(), m_parent.m_config.attention_mask_value);
}

void ModelChunk::setup_buffers() {
    auto &llm_config = m_model_config.llm;
    auto n_kv_heads  = llm_config.n_kv_heads;
    if (m_sibling) {
        m_buffers = m_sibling->m_buffers;
    }

    setup_buffer(m_buffers["x"], m_tensors["x"]);
    setup_buffer(m_buffers["attn_bias"], m_tensors["attn_bias"]);
    setup_buffer(m_buffers["rope_embed_cos"], m_tensors["rope_embed_cos"]);
    setup_buffer(m_buffers["rope_embed_sin"], m_tensors["rope_embed_sin"]);
    setup_buffer(m_buffers["out"], m_tensors["out"]);

    for (size_t i = 0; i < n_layers(); i++) {
        for (size_t j = 0; j < n_kv_heads; j++) {
            setup_buffer(
                m_buffers[fmt::format("layer_{}_key_t_cache_{}", m_config.start_layer_id + i, j)],
                m_tensors[fmt::format("layer_{}_key_t_cache_{}", m_config.start_layer_id + i, j)]
            );
            setup_buffer(
                m_buffers[fmt::format("layer_{}_value_cache_{}", m_config.start_layer_id + i, j)],
                m_tensors[fmt::format("layer_{}_value_cache_{}", m_config.start_layer_id + i, j)]
            );
            setup_buffer(
                m_buffers[fmt::format("layer_{}_key_{}", m_config.start_layer_id + i, j)],
                m_tensors[fmt::format("layer_{}_key_{}", m_config.start_layer_id + i, j)]
            );
            setup_buffer(
                m_buffers[fmt::format("layer_{}_value_{}", m_config.start_layer_id + i, j)],
                m_tensors[fmt::format("layer_{}_value_{}", m_config.start_layer_id + i, j)]
            );
        }
    }
}

void ModelChunk::load_kv(KVCacheInterface &kv_cache) {
    auto &llm_config = m_model_config.llm;
    auto head_dim    = llm_config.head_size;
    auto n_kv_heads  = llm_config.n_kv_heads;

    auto load = [&](const std::string &kv_type, size_t layer_id, size_t head_id) {
        auto layer_id_arg = fmt::arg("layer_id", layer_id);
        auto kv_type_arg  = fmt::arg("kv_type", kv_type);
        auto head_id_arg  = fmt::arg("head_id", head_id);
        auto path =
            m_parent.m_model_folder /
            fmt::vformat(m_config.kv_path_format, fmt::make_format_args(layer_id_arg, kv_type_arg, head_id_arg));

        size_t n_elements = m_config.kv_size * head_dim;

        auto binary_loader         = storage::build_file_loader(path, storage::FileLoaderMethod::DIO);
        const auto binary_buffer   = binary_loader->get_buffer<float>();
        const float *kv_cache_data = binary_buffer.data();

        std::vector<__fp16> fp16_data(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            fp16_data[i] = kv_cache_data[i];
        }

        for (size_t i = 0; i < m_config.kv_size; i++) {
            KVView entry;

            if (kv_type == "key") {
                entry = kv_cache.key_entry({.layer_id = layer_id, .head_id = head_id, .index = i});
            } else {
                entry = kv_cache.value_entry({.layer_id = layer_id, .head_id = head_id, .index = i});
            }

            entry.copy_from({
                .n_elements   = head_dim,
                .element_size = kv_element_size,
                .stride       = kv_element_size,
                .data         = fp16_data.data() + i * head_dim,
            });
        }
    };

    for (size_t i = 0; i < n_layers(); i++) {
        size_t layer_id = m_config.start_layer_id + i;
        for (size_t j = 0; j < n_kv_heads; j++) {
            load("key", layer_id, j);
            load("value", layer_id, j);
        }
    }
}

} // namespace powerserve::qnn

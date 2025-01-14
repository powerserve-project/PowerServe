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

#include "config.hpp"
#include "core/kv_cache.hpp"
#include "qnn.hpp"

#include <chrono>

// #define QNN_TIMER 0

namespace powerserve::qnn {

struct CausalLM;

struct GraphInterface {
    CausalLM &m_parent;
    GraphInterface *m_sibling = nullptr;
    const QNNGraphConfig m_graph_config;
    const ModelConfig &m_model_config;
    ContextBinary &m_context_binary;
    std::unique_ptr<qnn::Graph> m_graph;
    std::map<const std::string, qnn::QNNTensor *> m_tensors;
    std::map<std::string, std::shared_ptr<SharedBuffer>> m_buffers;
#if defined(QNN_TIMER)
    using clock = std::chrono::high_resolution_clock;
#endif

    GraphInterface(CausalLM &parent, const QNNGraphConfig &config);
    virtual ~GraphInterface() = default;

    auto io_tensor_size() const -> size_t;
    auto input_buffer() const -> void *;
    auto output_buffer() const -> void *;
    void setup_tensor(
        const std::string &tensor_name, const std::vector<size_t> &tensor_shape, const Qnn_DataType_t tensor_type
    );
    void setup_buffer(std::shared_ptr<SharedBuffer> &buffer, qnn::QNNTensor *tensor);
    virtual void setup_tensors() = 0;
    virtual void setup_buffers() = 0;
#if defined(QNN_TIMER)
    void execute(int64_t &time) const;
#else
    void execute() const;
#endif
};

struct Embedding : GraphInterface {
    Embedding(CausalLM &parent, QNNGraphConfig &info);
    virtual ~Embedding() override = default;

    void setup_tensors() override;
    void setup_buffers() override;
};

struct ModelChunk : GraphInterface {
    static constexpr auto kv_type           = QNN_DATATYPE_FLOAT_16;
    static constexpr size_t kv_element_size = type_size(kv_type);

    ChunkConfig &m_config;

    ModelChunk(CausalLM &parent, ChunkConfig &info);
    virtual ~ModelChunk() override = default;

    auto n_layers() const -> size_t;
    void setup_tensors() override;
    void initialize(KVCacheInterface &kv_cache);
    void setup_buffers() override;
    void load_kv(KVCacheInterface &kv_cache);
};

struct Vision : GraphInterface {
    VisionConfig m_config;

    Vision(CausalLM &parent, VisionConfig info);
    virtual ~Vision() override = default;

    void setup_tensors() override;
    void setup_buffers() override;
};
} // namespace powerserve::qnn

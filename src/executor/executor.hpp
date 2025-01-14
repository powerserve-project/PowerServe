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

#include "backend/platform.hpp"
#include "graph/graph.hpp"

namespace powerserve {

struct Executor {
public:
    Platform &m_platform;
    Graph &m_graph;

public:
    Executor(Platform &platform, Graph &graph) : m_platform(platform), m_graph(graph) {}

public:
    void allocate_buffers();
    void run();
    void plan();

private:
    template <typename T>
    void create_cpu_buffer(std::shared_ptr<TensorNode> tensor) {
        if (tensor->type == NodeType::TENSOR_VIEW) {
            tensor->m_data =
                CPUBuffer::create_buffer_view<T>(tensor->tensor_view()->parent->get<CPUBuffer>(), tensor->m_shape);
        } else {
            tensor->m_data = CPUBuffer::create_buffer<T>(tensor->m_shape);
        }
    }
};

} // namespace powerserve

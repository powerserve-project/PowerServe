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

#include "core/buffer.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"

namespace powerserve {

struct CPUBuffer : BaseBuffer {
public:
    Stride m_stride; // In bytes
    void *m_data;
    bool m_allocated_by_malloc = false;

public:
    CPUBuffer(Stride stride, void *data, bool allocated_by_malloc = false) :
        m_stride(stride),
        m_data(data),
        m_allocated_by_malloc(allocated_by_malloc) {}

    virtual ~CPUBuffer() override {
        if (m_allocated_by_malloc) {
            free(m_data);
        }
    }

    template <typename T>
    static auto create_buffer(Shape shape) -> BufferPtr {
        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        return std::make_shared<CPUBuffer>(stride, malloc(size), true);
    }

    template <typename T>
    static auto create_buffer_view(CPUBuffer &parent, Shape shape) -> BufferPtr {
        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        POWERSERVE_ASSERT(parent.m_data != nullptr, "paraent buffer is nullptr");
        auto b    = std::make_shared<CPUBuffer>(stride, nullptr, false);
        b->m_data = parent.m_data;
        return b;
    }
};

} // namespace powerserve

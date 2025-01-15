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

#include "ggml.hpp"

#include "backend/cpu_buffer.hpp"
#include "core/data_type.hpp"
#include "ggml.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace powerserve::ggml {

void GGMLBackend::plan(std::vector<std::shared_ptr<OpNode>> &ops) {
    size_t max_work_size = 0;
    for (auto op : ops) {
        size_t cur = 0;

        const int n_tasks = get_n_tasks(op);

        switch (op->op) {
        // custom ops
        case OpType::SILU_HADAMARD:
        case OpType::ADD_CACHE:
        case OpType::TRANSPOSE:
        case OpType::PRINT:
        case OpType::VIEW:
        case OpType::COPY: {
        } break;

        case OpType::PERMUTE:
        case OpType::CONT:
        case OpType::GET_MASK:
        case OpType::GET_EMBEDDING: {
            max_work_size = 0;
        } break;

        case OpType::ADD: {
            auto a = op->prev[0]->tensor();
            if (a->is_quantized()) {
                cur = ggml_type_size(GGML_TYPE_F32) * a->m_shape[0] * n_tasks;
            }
        } break;

        case OpType::MAT_MUL: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();

            const enum ggml_type vec_dot_type = get_vec_dot_type(x);
            if (ggml::convert_datatype_to_ggml(weight->m_dtype) != vec_dot_type) {
                cur = ggml_row_size(vec_dot_type, weight->n_elements());
            }
        } break;

        case OpType::SOFTMAX_EXT:
        case OpType::SOFTMAX:
        case OpType::ROPE: {
            auto dst = op->next[0]->tensor();
            cur      = ggml_type_size(GGML_TYPE_F32) * dst->m_shape[0] * n_tasks;
        } break;

        case OpType::RMS_NORM: {
        } break;

#if defined(POWERSERVE_WITH_QNN)
        case OpType::QNN_FORWARD: {
        } break;
        case OpType::QNN_FORWARD_VL: {
        } break;
#endif

        default:
            POWERSERVE_ABORT("unsupported op type: {}", static_cast<int>(op->op));
        }

        max_work_size = std::max(max_work_size, cur);
    }

    setup_work_data(max_work_size);
}

void GGMLBackend::setup_work_data(size_t work_size) {
    if (work_size <= m_wdata.size()) {
        return;
    }
    if (work_size > 0) {
        work_size += get_cache_line_size() * num_threads;
    }

    m_wdata.resize(work_size);
    m_params.wdata = m_wdata.data();
    m_params.wsize = m_wdata.size();
}

void GGMLBackend::reset_kv_batch_size(const size_t batch_size) const {
    m_kv->reset_batch_size(batch_size);
}

void GGMLBackend::silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const {
    POWERSERVE_ASSERT(is_contiguous(out, 0));
    POWERSERVE_ASSERT(is_contiguous(hb, 0));
    POWERSERVE_ASSERT(is_contiguous(hb2, 0));
    float *out_data = static_cast<float *>(out->get<CPUBuffer>().m_data);
    float *hb_data  = static_cast<float *>(hb->get<CPUBuffer>().m_data);
    float *hb2_data = static_cast<float *>(hb2->get<CPUBuffer>().m_data);

    for (size_t j = 0; j < hb->n_elements(); j++) {
        float val = hb_data[j];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2_data[j];
        out_data[j] = val;
    }
}

void GGMLBackend::print(const Tensor *x, size_t size) const {
    POWERSERVE_UNUSED(size);
    POWERSERVE_ASSERT(x->m_dtype == DataType::FP32);
    auto shape  = x->m_shape;
    auto stride = x->get<CPUBuffer>().m_stride;
    printf("\n{%ld, %ld, %ld, %ld}\n", shape[3], shape[2], shape[1], shape[0]);
    printf("\n{%ld, %ld, %ld, %ld}\n", stride[3], stride[2], stride[1], stride[0]);
    for (size_t i3 = 0; i3 < shape[3]; i3++) {
        for (size_t i2 = 0; i2 < shape[2]; i2++) {
            for (size_t i1 = 0; i1 < shape[1]; i1++) {
                for (size_t i0 = 0; i0 < shape[0]; i0++) {
                    float *ptr = (float *)((char *)x->get<CPUBuffer>().m_data + i3 * stride[3] + i2 * stride[2] +
                                           i1 * stride[1] + i0 * stride[0]);
                    // printf("[%ld][%ld][%ld][%ld] = %.6f\n", i3, i2, i1, i0, (double)*ptr);
                    printf("%.6f\n", (double)*ptr);
                }
            }
        }
    }
    exit(0);
}

void GGMLBackend::add_cache(const Tensor *k, const Tensor *v, size_t L, const std::vector<int> &pos, size_t head_id) {
    fmt::println("This function is deprecated!");
    POWERSERVE_UNUSED(head_id);

    auto kv_dim       = m_kv->m_kv_dim;
    auto batch_size   = pos.size();
    auto cur_position = m_kv->kv_cache->position;
    POWERSERVE_ASSERT(batch_size == m_kv->m_batch_size);

    float *src_k  = static_cast<float *>(k->get<CPUBuffer>().m_data); // (kv_dim, batch_size, 1, 1)
    float *src_v  = static_cast<float *>(v->get<CPUBuffer>().m_data); // (kv_dim, batch_size, 1, 1)
    float *dst_kb = m_kv->chunk.key_buffer[L].data() + kv_dim * cur_position;
    float *dst_vb = m_kv->chunk.value_buffer[L].data() + kv_dim * cur_position;
    memcpy(dst_kb, src_k, kv_dim * batch_size * sizeof(float));
    memcpy(dst_vb, src_v, kv_dim * batch_size * sizeof(float));
}

void GGMLBackend::transpose(const Tensor *out, const Tensor *x) const {
    Stride stride{x->get<CPUBuffer>().m_stride};
    stride[0] = x->get<CPUBuffer>().m_stride[1];
    stride[1] = x->get<CPUBuffer>().m_stride[0];

    out->get<CPUBuffer>().m_data   = x->get<CPUBuffer>().m_data;
    out->get<CPUBuffer>().m_stride = stride;
}

void GGMLBackend::setup_threadpool() {
    m_thread_pool = std::make_unique<ThreadPool>(m_thread_config);
}

void GGMLBackend::reset_threadpool() {
    m_thread_pool.reset();
}

} // namespace powerserve::ggml

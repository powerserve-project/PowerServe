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

#include "ggml-quants.h"
#include "ggml.hpp"

namespace powerserve::ggml {

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        params.thread_pool = (void *)m_thread_pool.get();
        params.barrier_fn  = [](void *opaque) {
            auto thread_pool = (ThreadPool *)opaque;
            thread_pool->barrier();
        };
        params.current_chunk = (atomic_int *)&m_current_chunk;

        powerserve_compute_forward_mul_mat(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
}

void GGMLBackend::rmsnorm(const Tensor *out, const Tensor *x, const Tensor *weight, float eps) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(weight);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rms_norm(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), eps);
    });
}

void GGMLBackend::softmax(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_soft_max(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::rope(
    Tensor *out, const Tensor *src, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(src);
    auto src1_tensor = std::make_unique<ggml_tensor>();
    {
        src1_tensor->data  = (void *)pos.data();
        src1_tensor->type  = GGML_TYPE_I32;
        src1_tensor->ne[0] = pos.size();
        src1_tensor->ne[1] = src1_tensor->ne[2] = src1_tensor->ne[3] = 1;
        src1_tensor->nb[0]                                           = sizeof(int32_t);
        src1_tensor->nb[1] = src1_tensor->nb[2] = src1_tensor->nb[3] = pos.size() * sizeof(int32_t);
    }

    rope_compute_params rope_params = {
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow,
        .mode        = rope_cfg.rope_type,
    };

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rope(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), nullptr, &rope_params
        );
    });
}

void GGMLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_add(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });
}

void GGMLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    stride[axes[0]] = x->get<CPUBuffer>().m_stride[0];
    stride[axes[1]] = x->get<CPUBuffer>().m_stride[1];
    stride[axes[2]] = x->get<CPUBuffer>().m_stride[2];
    stride[axes[3]] = x->get<CPUBuffer>().m_stride[3];

    out->get<CPUBuffer>().m_stride = stride;
}

void GGMLBackend::cont(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::copy(const Tensor *dst, const Tensor *src) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias)
    const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(mask);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_softmax_ext(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), scale, max_bias
        );
    });
}

void GGMLBackend::get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    auto embd_tb = static_cast<char *>(weight->get<CPUBuffer>().m_data);
    auto dst_tb  = static_cast<float *>(dst->get<CPUBuffer>().m_data);

    auto dim        = dst->m_shape[0];
    auto batch_size = tokens.size();
    POWERSERVE_ASSERT(batch_size == dst->m_shape[1]);
    auto weight_strip = weight->get<CPUBuffer>().m_stride;

    for (size_t i = 0; i < batch_size; i++) {
        auto token = tokens[i];
        auto src   = embd_tb + weight_strip[1] * token;
        POWERSERVE_ASSERT(src < embd_tb + weight_strip[2]);
        switch (weight->m_dtype) {
        case DataType::FP32: {
            memcpy(dst_tb + i * dim, src, dim * sizeof(float));
        } break;

        case DataType::GGML_Q4_0: {
            dequantize_row_q4_0((block_q4_0 *)src, dst_tb + i * dim, dim);
        } break;

        case DataType::GGML_Q8_0: {
            dequantize_row_q8_0((block_q8_0 *)src, dst_tb + i * dim, dim);
        } break;

        default:
            POWERSERVE_ASSERT(false);
        }
    }
}

bool GGMLBackend::is_contiguous(const Tensor *tensor, int n) const {
    POWERSERVE_ASSERT(n >= 0 && n <= 2);
    if (n == 0) {
        return ggml_is_contiguous_0(convert_to_ggml(tensor).get());
    } else if (n == 1) {
        return ggml_is_contiguous_1(convert_to_ggml(tensor).get());
    } else if (n == 2) {
        return ggml_is_contiguous_2(convert_to_ggml(tensor).get());
    }
    return false;
}

int GGMLBackend::get_n_tasks(std::shared_ptr<OpNode> op) {
    int n_tasks = 1;

    switch (op->op) {
    // custom ops
    case OpType::SILU_HADAMARD:
    case OpType::ADD_CACHE:
    case OpType::PRINT:
    case OpType::VIEW:
    case OpType::TRANSPOSE:
    case OpType::COPY: {
        n_tasks = 1;
    } break;

    // ggml wrapper ops
    case OpType::PERMUTE:
    case OpType::GET_MASK:
    case OpType::GET_EMBEDDING: {
        n_tasks = 1;
    } break;

    case OpType::ROPE:
    case OpType::RMS_NORM:
    case OpType::CONT:
    case OpType::MAT_MUL:
    case OpType::ADD: {
        n_tasks = num_threads;
    } break;

    case OpType::SOFTMAX_EXT:
    case OpType::SOFTMAX: {
        n_tasks = std::min((int64_t)num_threads, op->prev[0]->tensor()->nrows());
    } break;

#if defined(POWERSERVE_WITH_QNN)
    case OpType::QNN_FORWARD: {
        n_tasks = 1;
    } break;
    case OpType::QNN_FORWARD_VL: {
        n_tasks = 1;
    } break;
#endif

    default: {
        fmt::println("op not implemented: {}", int(op->op));
        POWERSERVE_ASSERT(false);
    }
    }

    return n_tasks;
}

ggml_type GGMLBackend::get_vec_dot_type(const Tensor *tensor) {
    auto t = convert_to_ggml(tensor);
    return powerserve_get_vec_dot_type(t.get());
}

} // namespace powerserve::ggml

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

#include "backend/backend.hpp"
#include "backend/cpu_buffer.hpp"
#include "backend/ggml/ggml_kv_cache.hpp"
#include "core/config.hpp"
#include "core/data_type.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/thread_pool.hpp"
#include "ggml.h"
#include "graph/node.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace powerserve::ggml {

static ggml_type convert_datatype_to_ggml(DataType dtp) {
    switch (dtp) {
    case DataType::FP32:
        return GGML_TYPE_F32;
    case DataType::FP16:
        return GGML_TYPE_F16;
    case DataType::GGML_Q4_0:
        return GGML_TYPE_Q4_0;
    case DataType::GGML_Q8_0:
        return GGML_TYPE_Q8_0;
    case DataType::INT32:
        return GGML_TYPE_I32;
    case DataType::INT64:
        return GGML_TYPE_I64;
    default:
        POWERSERVE_ABORT("unsupported data type: {}", static_cast<int>(dtp));
    }
}

static DataType convert_datatype_from_ggml(ggml_type tp) {
    switch (tp) {
    case GGML_TYPE_F32:
        return DataType::FP32;
    case GGML_TYPE_F16:
        return DataType::FP16;
    case GGML_TYPE_Q4_0:
        return DataType::GGML_Q4_0;
    case GGML_TYPE_Q8_0:
        return DataType::GGML_Q8_0;
    case GGML_TYPE_I32:
        return DataType::INT32;
    case GGML_TYPE_I64:
        return DataType::INT64;
    default:
        POWERSERVE_ABORT("unsupported ggml data type: {}", static_cast<int>(tp));
    }
}

static Tensor convert_from_ggml(ggml_tensor *t) {
    POWERSERVE_ASSERT(t != nullptr);
    Shape shape;
    Stride stride;
    for (size_t i = 0; i < max_n_dims; i++) {
        shape[i]  = t->ne[i];
        stride[i] = t->nb[i];
    }
    Tensor tensor(convert_datatype_from_ggml(t->type), shape);
    tensor.m_data = std::make_shared<CPUBuffer>(stride, t->data);
    return tensor;
}

static std::unique_ptr<ggml_tensor> convert_to_ggml(const Tensor *tensor) {
    auto gt  = std::make_unique<ggml_tensor>();
    gt->data = tensor->get<CPUBuffer>().m_data;
    gt->type = convert_datatype_to_ggml(tensor->m_dtype);
    for (size_t i = 0; i < max_n_dims; i++) {
        gt->ne[i] = tensor->m_shape[i];
        gt->nb[i] = tensor->get<CPUBuffer>().m_stride[i];
    }
    return gt;
}

// debug functions
static void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    {
        POWERSERVE_LOG_INFO("version     : {:10}", gguf_get_version(gguf_ctx));
        POWERSERVE_LOG_INFO("n_kv        : {:10}", gguf_get_n_kv(gguf_ctx));
        POWERSERVE_LOG_INFO("n_tensors   : {:10}", gguf_get_n_tensors(gguf_ctx));
        POWERSERVE_LOG_INFO("alignment   : {:10}", gguf_get_alignment(gguf_ctx));
        POWERSERVE_LOG_INFO("meta size   : {:10}", gguf_get_meta_size(gguf_ctx));
        POWERSERVE_LOG_INFO("data offset : {:10}", gguf_get_data_offset(gguf_ctx));
    }

    {
        for (auto i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            auto key      = gguf_get_key(gguf_ctx, i);
            auto v_type   = gguf_get_kv_type(gguf_ctx, i);
            auto type_str = gguf_type_name(v_type);
            POWERSERVE_LOG_INFO("{:40}: {:4}", key, type_str);
        }
    }

    {
        for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
            auto name   = gguf_get_tensor_name(gguf_ctx, i);
            auto t_type = gguf_get_tensor_type(gguf_ctx, i);
            POWERSERVE_LOG_INFO(
                "{:40}: {:6}: {:10}", name, ggml_type_name(t_type), gguf_get_tensor_offset(gguf_ctx, i)
            );
        }
    }

    {
        POWERSERVE_LOG_INFO("GGML used mem        : {:10}", ggml_used_mem(ggml_ctx));
        POWERSERVE_LOG_INFO("GGML no alloc        : {:10}", ggml_get_no_alloc(ggml_ctx));
        POWERSERVE_LOG_INFO("GGML mem buffer      : {:10}", ggml_get_mem_buffer(ggml_ctx));
        POWERSERVE_LOG_INFO("GGML mem size        : {:10}", ggml_get_mem_size(ggml_ctx));
        POWERSERVE_LOG_INFO("GGML max tensor size : {:10}", ggml_get_max_tensor_size(ggml_ctx));
    }
}

static void debug_tensors_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
        auto t = ggml_get_tensor(ggml_ctx, gguf_get_tensor_name(gguf_ctx, i));

        POWERSERVE_LOG_DEBUG(
            "{:40}|{:>5}|({:6},{:6},{:1},{:1})|{:10}|{:4}|{:4}|{:10}",
            ggml_get_name(t),
            ggml_type_name(t->type),
            t->ne[0],
            t->ne[1],
            t->ne[2],
            t->ne[3],
            ggml_get_data(t),
            ggml_type_size(t->type),
            ggml_blck_size(t->type),
            ggml_row_size(t->type, ggml_nelements(t)) // ne * ggml_type_size / ggml_blk_size (bytes)
        );
    }
}

static void debug_system_info(void) {
    std::string s{};

    s += "AVX = " + std::to_string(ggml_cpu_has_avx()) + " | ";
    s += "AVX_VNNI = " + std::to_string(ggml_cpu_has_avx_vnni()) + " | ";
    s += "AVX2 = " + std::to_string(ggml_cpu_has_avx2()) + " | ";
    s += "AVX512 = " + std::to_string(ggml_cpu_has_avx512()) + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "AVX512_BF16 = " + std::to_string(ggml_cpu_has_avx512_bf16()) + " | ";
    s += "FMA = " + std::to_string(ggml_cpu_has_fma()) + " | ";
    s += "NEON = " + std::to_string(ggml_cpu_has_neon()) + " | ";
    s += "SVE = " + std::to_string(ggml_cpu_has_sve()) + " | ";
    s += "ARM_FMA = " + std::to_string(ggml_cpu_has_arm_fma()) + " | ";
    s += "F16C = " + std::to_string(ggml_cpu_has_f16c()) + " | ";
    s += "FP16_VA = " + std::to_string(ggml_cpu_has_fp16_va()) + " | ";
    s += "RISCV_VECT = " + std::to_string(ggml_cpu_has_riscv_v()) + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = " + std::to_string(ggml_cpu_has_blas()) + " | ";
    s += "SSE3 = " + std::to_string(ggml_cpu_has_sse3()) + " | ";
    s += "SSSE3 = " + std::to_string(ggml_cpu_has_ssse3()) + " | ";
    s += "VSX = " + std::to_string(ggml_cpu_has_vsx()) + " | ";
    s += "MATMUL_INT8 = " + std::to_string(ggml_cpu_has_matmul_int8()) + " | ";
    s += "LLAMAFILE = " + std::to_string(ggml_cpu_has_llamafile()) + " | ";

    POWERSERVE_LOG_INFO("system info: {}", s);
}

// **Note**: Backend receives Tensor not TensorNode
struct GGMLBackend : Backend {
public:
    op_compute_params m_params;
    std::vector<char> m_wdata;
    std::unique_ptr<GGMLKV> m_kv;
    int num_threads;

public:
    explicit GGMLBackend(const ModelConfig::LLMConfig &config, const HyperParams &hparams) :
        num_threads(hparams.n_threads) {
        m_params = {
            .ith           = 0,
            .nth           = 1,
            .wsize         = 0,
            .wdata         = nullptr,
            .thread_pool   = nullptr,
            .barrier_fn    = nullptr,
            .current_chunk = nullptr,
        };

        for (int i = 0; i < num_threads; i++) {
            m_thread_config.emplace_back(ThreadConfig{});
        }
        m_kv = std::make_unique<GGMLKV>(config);
    }

    ~GGMLBackend() override = default;

public:
    // ggml wrapper ops
    void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const;
    void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight, float eps) const;
    void rope(
        Tensor *out, const Tensor *src, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &rope_cfg
    ) const;
    void softmax(const Tensor *out, const Tensor *x) const;
    void permute(const Tensor *out, const Tensor *x, Shape axes) const;
    void cont(const Tensor *out, const Tensor *x) const;
    void softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias) const;

    bool is_contiguous(const Tensor *tensor, int n) const;
    int get_n_tasks(std::shared_ptr<OpNode> op);
    enum ggml_type get_vec_dot_type(const Tensor *tensor);

public:
    void silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const;
    void copy(const Tensor *dst, const Tensor *src) const;
    void print(const Tensor *x, size_t size) const;
    void reset_kv_batch_size(const size_t batch_size) const;
    void add_cache(const Tensor *k, const Tensor *v, size_t L, const std::vector<int> &pos, size_t head_id);
    void transpose(const Tensor *out, const Tensor *x) const;

public:
    void plan(std::vector<std::shared_ptr<OpNode>> &ops);
    void setup_work_data(size_t work_size);
    void setup_threadpool();
    void reset_threadpool();

private:
    std::vector<ThreadConfig> m_thread_config;
    std::unique_ptr<ThreadPool> m_thread_pool;
    std::atomic<int> m_current_chunk = 0;
};

} // namespace powerserve::ggml

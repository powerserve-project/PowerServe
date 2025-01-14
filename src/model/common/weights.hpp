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

#include "backend/ggml/ggml.hpp"

#include <cstdio>

namespace powerserve {

// Note: Each model can add its local tensors
struct LayerWeights {
public:
    Tensor attn_norm; // "blk.$.attn_norm.weight" (layer, dim)
    Tensor ffn_norm;  // "blk.$.ffn_norm.weight" (layer, dim)
    // dim == n_heads * head_size
    Tensor attn_q;      // "blk.$.attn_q.weight" (layer, dim, n_heads * head_size)
    Tensor attn_k;      // "blk.$.attn_k.weight" (layer, dim, n_kv_heads * head_size)
    Tensor attn_v;      // "blk.$.attn_v.weight" (layer, dim, n_kv_heads * head_size)
    Tensor attn_output; // "blk.$.attn_output.weight" (layer, n_heads * head_size, dim)

    Tensor ffn_gate; // "blk.$.ffn_gate.weight" (layer, dim, hidden_dim)
    Tensor ffn_up;   // "blk.$.ffn_up.weight" (layer, dim, hidden_dim)
    Tensor ffn_down; // "blk.$.ffn_down.weight" (layer, hidden_dim, dim)

    Tensor attn_q_bias;
    Tensor attn_k_bias;
    Tensor attn_v_bias;

    virtual ~LayerWeights() = default;

protected:
    static Tensor get_tensor(ggml_context *ctx, uint32_t layer, const char *name) {
        std::string tensor_name = fmt::format("blk.{}.{}", layer, name);
        ggml_tensor *t          = ggml_get_tensor(ctx, tensor_name.c_str());
        if (t == nullptr) {
            throw std::runtime_error(fmt::format("Failed to get tensor: {}", tensor_name));
        }
        return ggml::convert_from_ggml(t);
    }
};

struct Weight {
public:
    Tensor token_embedding_table; // "token_embd.weight" (vocab_size, dim)
    Tensor output_weight;         // "output.weight" (vocab_size, dim)
    Tensor rms_final_weight;      // "output_norm.weight" (dim,)

    std::vector<LayerWeights> lw;

public:
    Weight(ggml_context *ctx, bool lazy_load) {
        token_embedding_table = ggml::convert_from_ggml(ggml_get_tensor(ctx, "token_embd.weight"));
        if (!lazy_load) {
            auto ow_name     = ggml_get_tensor(ctx, "output.weight") == nullptr ? "token_embd.weight" : "output.weight";
            output_weight    = ggml::convert_from_ggml(ggml_get_tensor(ctx, ow_name));
            rms_final_weight = ggml::convert_from_ggml(ggml_get_tensor(ctx, "output_norm.weight"));
        }
    }

    virtual ~Weight() = default;
};

} // namespace powerserve

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

#include "norm_attention.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace powerserve {

TensorNode *NormAttention::build(
    Graph &g,
    TensorNode *x, // {embd_dim, bs, 1, 1}
    int64_t L,
    const TensorNode *k_cache, // {seq_len * kv_dim, 1, 1, 1}
    const TensorNode *v_cache,
    const std::vector<int> &pos,
    const CausalAttentionMask &mask,
    bool is_need_bias
) {
    auto batch_size = pos.size();
    auto head_size  = m_config.head_size;
    POWERSERVE_ASSERT(head_size == (size_t)m_config.rope_config.n_dims);
    auto n_head    = m_config.n_heads;
    auto n_head_kv = m_config.n_kv_heads;
    auto n_ctx     = m_config.seq_len;
    size_t kv_gqa  = head_size * n_head_kv;
    size_t cur_pos = pos[0];

    auto att_norm_w = g.add_tensor(m_weights->lw[L].attn_norm);     // (embd_dim, 1, 1, 1)
    auto att_norm_o = g.rms_norm(x, att_norm_w, m_config.norm_eps); // (embd_dim, bs, 1, 1)

    // QKV
    auto q_w = g.add_tensor(m_weights->lw[L].attn_q); // (embd_dim, embd_dim, 1, 1)
    auto q   = g.mat_mul(q_w, att_norm_o);            // (embd_dim, bs, 1, 1)
    if (is_need_bias) {
        auto q_b = g.add_tensor(m_weights->lw[L].attn_q_bias); // (embd_dim, 1, 1, 1)
        q        = g.add(q, q_b);
    }
    // embd_dim == n_heads * head_size
    // kv_dim == n_kv_heads * head_size
    auto k_w = g.add_tensor(m_weights->lw[L].attn_k); // (embd_dim, kv_dim, 1, 1)
    auto k   = g.mat_mul(k_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    if (is_need_bias) {
        auto k_b = g.add_tensor(m_weights->lw[L].attn_k_bias); // (kv_dim, 1, 1, 1)
        k        = g.add(k, k_b);
    }

    auto v_w = g.add_tensor(m_weights->lw[L].attn_v); // (embd_dim, kv_dim, 1, 1)
    auto v   = g.mat_mul(v_w, att_norm_o);            // (kv_dim, batch_size, 1, 1)
    if (is_need_bias) {
        auto v_b = g.add_tensor(m_weights->lw[L].attn_v_bias); // (kv_dim, 1, 1, 1)
        v        = g.add(v, v_b);
    }

    // (head_size, n_heads, bs, 1)
    auto q_view = g.view_tensor(q, {head_size, n_head, q->m_shape[1], q->m_shape[2]});
    // (head_size, n_kv_heads, bs, 1)
    auto k_view = g.view_tensor(k, {head_size, n_head_kv, k->m_shape[1], k->m_shape[2]});
    auto rope_q = g.rope(q_view, pos, m_config.rope_config); // (head_size, n_heads, bs, 1)
    auto rope_k = g.rope(k_view, pos, m_config.rope_config); // (head_size, n_kv_heads, bs, 1)

    // store kv
    {
        k                 = rope_k;
        v                 = g.transpose(v);
        auto k_cache_view = g.view(
            k_cache,
            {batch_size * kv_gqa, 1, 1, 1},
            {k_cache->element_size(),
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa,
             k_cache->element_size() * batch_size * kv_gqa},
            k_cache->row_size(kv_gqa) * cur_pos
        );
        g.copy(k_cache_view, k);

        auto v_cache_view = g.view(
            v_cache,
            {batch_size, kv_gqa, 1, 1},
            {
                v_cache->element_size(),
                n_ctx * v_cache->element_size(),
                n_ctx * v_cache->element_size() * kv_gqa,
                n_ctx * v_cache->element_size() * kv_gqa,
            },
            v_cache->element_size() * cur_pos
        );
        g.copy(v_cache_view, v);
    }

    TensorNode *att_scores = nullptr;
    {
        // size_t n_kv = ((pos.back() / 32) + 1) * 32;
        // size_t batch_32 = ((batch_size / 32) + 1) * 32;
        size_t n_kv     = pos.back() + 1;
        size_t batch_32 = batch_size;

        // (head_size, bs, n_heads, 1)
        q = g.permute(rope_q, {0, 2, 1, 3});
        // {head_size, cur_postion, n_head_kv, 1}
        k = g.view(
            k_cache,
            {head_size, n_kv, n_head_kv, 1},
            {
                k_cache->element_size(),
                k_cache->row_size(kv_gqa),
                k_cache->row_size(head_size),
                k_cache->row_size(head_size) * n_head_kv,
            }
        );

        // {bs, cur_postion, n_head_kv, 1}
        auto kq                = g.mat_mul(k, q);
        auto f_attention_scale = 0.0f;
        float kq_scale         = f_attention_scale == 0.0f ? 1.0f / sqrtf(float(head_size)) : f_attention_scale;
        float f_max_alibi_bias = 0.000000;
        auto kq_mask           = g.get_mask(mask, {n_kv, batch_32, 1, 1}, pos);
        kq                     = g.softmax_ext(kq, kq_mask, kq_scale, f_max_alibi_bias);

        // split cached v into n_head heads
        // {cur_postion, head_size, n_head_kv, 1};
        v = g.view(
            v_cache,
            {n_kv, head_size, n_head_kv, 1},
            {v_cache->element_size(),
             v_cache->element_size() * n_ctx,
             v_cache->element_size() * n_ctx * head_size,
             v_cache->element_size() * n_ctx * head_size * n_head_kv}
        );
        // {head_size, cur_postion, n_head_kv, 1};
        auto kqv = g.mat_mul(v, kq);
        // {head_size, n_head_kv, cur_postion, 1};
        auto kqv_merged = g.permute(kqv, {0, 2, 1, 3});
        //  {embed_dim, bs, 1, 1};
        att_scores = g.cont(kqv_merged, {head_size * n_head, batch_size, 1, 1});
    }

    auto attn_output_w = g.add_tensor(m_weights->lw[L].attn_output);
    auto attn_o        = g.mat_mul(attn_output_w, att_scores); // (embd_dim, bs, 1, 1)

    // residual connection
    auto res_conn = g.add(x, attn_o);
    return res_conn;
}

} // namespace powerserve

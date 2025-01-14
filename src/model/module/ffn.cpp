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

#include "model/module/ffn.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

namespace powerserve {

TensorNode *FFN::build(Graph &g, TensorNode *attn_o, int64_t L) {
    auto ffn_norm_w = g.add_tensor(m_weights->lw[L].ffn_norm);
    auto ffn_norm_o = g.rms_norm(attn_o, ffn_norm_w, m_config.norm_eps);

    auto gate_w = g.add_tensor(m_weights->lw[L].ffn_gate);
    auto gate_o = g.mat_mul(gate_w, ffn_norm_o);

    auto up_w = g.add_tensor(m_weights->lw[L].ffn_up);
    auto up_o = g.mat_mul(up_w, ffn_norm_o);

    // {hidden_dim, bs, 1, 1}
    auto silu = g.silu_hadamard(gate_o, up_o);

    auto down_w = g.add_tensor(m_weights->lw[L].ffn_down);
    auto down_o = g.mat_mul(down_w, silu);

    // {embed_dim, bs, 1, 1}
    auto res_conn = g.add(attn_o, down_o);

    return res_conn;
}

} // namespace powerserve

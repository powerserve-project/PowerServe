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

#include "graph/node.hpp"

namespace powerserve {

struct Graph {
public:
    std::vector<std::shared_ptr<TensorNode>> tensors;
    std::vector<std::shared_ptr<OpNode>> ops;
    std::string m_model_id;

    Graph(std::string model_id) : m_model_id(model_id) {}

public:
    auto add_tensor(const Tensor &tensor) -> TensorNode *;
    auto new_tensor(DataType dtype, const Shape &shape) -> TensorNode *;
    auto new_op(OpType type) -> OpNode *;
    auto dup_tensor(TensorNode *tensor) -> TensorNode *;
    auto view_tensor(const TensorNode *tensor, Shape shape) -> TensorViewNode *;

public:
    auto get_embedding(TensorNode *weight, const std::vector<int> &tokens) -> TensorNode *;
    auto add(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto mat_mul(TensorNode *a, TensorNode *b) -> TensorNode *;
    auto rms_norm(TensorNode *x, TensorNode *weight, float eps) -> TensorNode *;
    auto silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode *;
    void copy(TensorNode *dst, TensorNode *src);
    auto make_contiguous(TensorNode *x) -> TensorNode *;

#if defined(POWERSERVE_WITH_QNN)
    auto qnn_forward(
        TensorNode *x, std::vector<int> pos, const CausalAttentionMask &mask, size_t vocab_size, bool lm_head
    ) -> TensorNode *;
    auto qnn_forward_vl(
        TensorNode *x,
        std::vector<int> pos,
        const CausalAttentionMask &mask,
        size_t vocab_size,
        bool lm_head,
        std::vector<std::vector<float>> &pixel_values_list,
        std::vector<std::pair<int, size_t>> &img_infos
    ) -> TensorNode *;
#endif

    auto rope(TensorNode *src, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &params)
        -> TensorNode *;

    auto softmax(TensorNode *x) -> TensorNode *;
    auto softmax_ext(TensorNode *x, TensorNode *mask, float scale, float max_bias) -> TensorNode *;
    void print(TensorNode *x, size_t size);

    void add_cache(TensorNode *k, TensorNode *v, size_t L, const std::vector<int> &pos, size_t head_id);
    auto permute(TensorNode *x, Shape axes) -> TensorViewNode *;
    auto cont(TensorNode *x, Shape shape) -> TensorNode *;
    auto view(const TensorNode *x, Shape shape, Shape stride, size_t offset = 0) -> TensorViewNode *;
    auto get_mask(const CausalAttentionMask &mask, Shape shape, const std::vector<int> &pos) -> TensorNode *;
    auto transpose(TensorNode *x) -> TensorViewNode *;
};

} // namespace powerserve

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

#include "core/config.hpp"
#include "core/tensor.hpp"
#include "model/module/attention_mask.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace powerserve {

// Base class for op parameters
struct OpParams {
    virtual ~OpParams() = default;
};

// This wrapper decouples the inheritance from the parameter structs
// So that parameter structs can keep its default constructors
template <typename T>
struct OpParamWrapper : OpParams {
    T value;

    explicit OpParamWrapper(const T &value) : value(value) {}
};

struct GetEmbeddingParams {
    std::vector<int> tokens;
};

struct RMSNormParams {
    float eps;
};

struct RopeParams {
    std::vector<int> pos;
    ModelConfig::LLMConfig::RopeConfig rope_cfg;
};

struct AddCacheParams {
    size_t L;
    std::vector<int> pos;
    size_t head_id;
};

struct CopyParams {};

#if defined(POWERSERVE_WITH_QNN)
struct QNNForwardParams : OpParams {
    const CausalAttentionMask mask;
    std::vector<int> pos;

    explicit QNNForwardParams(std::vector<int> pos, const CausalAttentionMask &mask) : mask(mask), pos(pos) {}

    ~QNNForwardParams() override = default;
};

struct QNNForwardVLParams : OpParams {
    const CausalAttentionMask mask;
    std::vector<int> pos;

    std::vector<std::vector<float>> &pixel_values_list;
    std::vector<std::pair<int, size_t>> &img_infos;

    explicit QNNForwardVLParams(
        std::vector<int> pos,
        const CausalAttentionMask &mask,
        std::vector<std::vector<float>> &pixel_values_list,
        std::vector<std::pair<int, size_t>> &img_infos
    ) :
        mask(mask),
        pos(pos),
        pixel_values_list(pixel_values_list),
        img_infos(img_infos) {}

    ~QNNForwardVLParams() override = default;
};
#endif

struct PrintParams {
    size_t size = 0;
};

struct PermuteParams {
    Shape axes;
};

struct ContParams {};

struct ViewParams {
    Shape stride;
    size_t offset;
};

struct SoftmaxExtParams {
    float scale;
    float max_bias;
};

struct GetMaskParams {
    const CausalAttentionMask &mask;
    const std::vector<int> &pos;
};

} // namespace powerserve

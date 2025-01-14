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

#include "attention_mask.hpp"

namespace powerserve {

AttentionMask::AttentionMask(size_t size) : size(size) {}

AttentionMaskView::AttentionMaskView(const AttentionMask &mask, size_t offset, size_t size) :
    mask(mask),
    offset(offset),
    size(size) {}

bool AttentionMaskView::not_masked(size_t i, size_t j) const {
    return mask.not_masked(offset + i, offset + j);
}

CausalAttentionMask::CausalAttentionMask(size_t size) : AttentionMask(size) {}

CausalAttentionMask::CausalAttentionMask(size_t size, const std::vector<std::vector<bool>> &batch_mask) :
    AttentionMask(size),
    mask(batch_mask) {}

bool CausalAttentionMask::not_masked(size_t i, size_t j) const {
    if (!mask.empty()) {
        return mask[i][j];
    }
    return i >= j;
}

} // namespace powerserve

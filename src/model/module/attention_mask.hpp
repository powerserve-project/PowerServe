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

#include <cstddef>
#include <vector>

namespace powerserve {

struct AttentionMask {
    size_t size = 0;

    AttentionMask(size_t size);

    virtual ~AttentionMask() = default;

    virtual bool not_masked(size_t i, size_t j) const = 0;
};

struct AttentionMaskView {
    const AttentionMask &mask;
    size_t offset = 0;

    size_t size = 0;

    AttentionMaskView(const AttentionMask &mask, size_t offset, size_t size);

    bool not_masked(size_t i, size_t j) const;
};

struct CausalAttentionMask : AttentionMask {
    std::vector<std::vector<bool>> mask;

    CausalAttentionMask(size_t size);
    CausalAttentionMask(size_t size, const std::vector<std::vector<bool>> &mask);
    virtual ~CausalAttentionMask() override = default;
    virtual bool not_masked(size_t i, size_t j) const override;
};

} // namespace powerserve

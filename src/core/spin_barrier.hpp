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

#include "core/spin_barrier.h"

#include <atomic>

#define CHECK_STRUCT_SIZE_AND_ALIGNMENT(a, b)                                                                          \
    static_assert(sizeof(a) == sizeof(b), "Sizes of " #a " and " #b " must equal");                                    \
    static_assert(alignof(a) == alignof(b), "Alignments of " #a " and " #b " must equal")

namespace powerserve {

struct SpinBarrier {
    void init(size_t new_width);
    void wait();

private:
    size_t width = 0;
    std::atomic<size_t> count;
};

CHECK_STRUCT_SIZE_AND_ALIGNMENT(SpinBarrier, spin_barrier);

} // namespace powerserve

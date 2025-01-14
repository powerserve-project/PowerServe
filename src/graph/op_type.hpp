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

namespace powerserve {

enum class OpType {
    NONE = 0,

    ADD,
    MAT_MUL,
    RMS_NORM,
    SILU_HADAMARD,
    ROPE,
    SOFTMAX,
    COPY,

#if defined(POWERSERVE_WITH_QNN)
    QNN_FORWARD,
    QNN_FORWARD_VL,
#endif

    PRINT,
    GET_EMBEDDING,
    ADD_CACHE,
    PERMUTE,
    CONT,
    VIEW,
    SOFTMAX_EXT,
    GET_MASK,
    TRANSPOSE,
    INSERT_IMG_EMBEDDIGN,
};

} // namespace powerserve

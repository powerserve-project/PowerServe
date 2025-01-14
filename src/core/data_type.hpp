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

#include "core/logger.hpp"
#include "ggml.h"

#include <cstddef>

namespace powerserve {

enum class DataType {
    UNKNOWN,

    FP32,
    FP16,
    INT32,
    INT64,
    GGML_Q4_0,
    GGML_Q8_0,

    COUNT,
};

static size_t get_type_size(DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return sizeof(float);
    case DataType::FP16:
        return ggml_type_size(GGML_TYPE_F16);
    case DataType::INT32:
        return sizeof(int32_t);
    case DataType::INT64:
        return sizeof(int64_t);
    case DataType::GGML_Q4_0:
        return ggml_type_size(GGML_TYPE_Q4_0);
    case DataType::GGML_Q8_0:
        return ggml_type_size(GGML_TYPE_Q8_0);
    default:
        POWERSERVE_ASSERT(false);
    }
}

static size_t get_block_size(DataType dtype) {
    switch (dtype) {
    case DataType::FP32:
        return ggml_blck_size(GGML_TYPE_F32);
    case DataType::FP16:
        return ggml_blck_size(GGML_TYPE_F16);
    case DataType::INT32:
        return ggml_blck_size(GGML_TYPE_I32);
    case DataType::INT64:
        return ggml_blck_size(GGML_TYPE_I64);
    case DataType::GGML_Q4_0:
        return ggml_blck_size(GGML_TYPE_Q4_0);
    case DataType::GGML_Q8_0:
        return ggml_blck_size(GGML_TYPE_Q8_0);
    default:
        POWERSERVE_ASSERT(false);
    }
}

} // namespace powerserve

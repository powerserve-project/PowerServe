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

#ifndef ASCENDC_KERNELS_H
#define ASCENDC_KERNELS_H

#include "aclrtlaunch_ascendc_get_row_f32.h"
#include "aclrtlaunch_ascendc_get_row_f16.h"
#include "aclrtlaunch_ascendc_get_row_q8_0.h"
#include "aclrtlaunch_ascendc_get_row_q4_0.h"

#include "aclrtlaunch_ascendc_quantize_f32_q8_0.h"
#include "aclrtlaunch_ascendc_quantize_f16_q8_0.h"
#include "aclrtlaunch_ascendc_quantize_f16_to_q4_0.h"
#include "aclrtlaunch_ascendc_quantize_f32_to_q4_0.h"

#include "aclrtlaunch_ascendc_dup_by_rows_fp16.h"
#include "aclrtlaunch_ascendc_dup_by_rows_fp32.h"
#include "aclrtlaunch_ascendc_dup_by_rows_fp32_to_fp16.h"
#include "aclrtlaunch_ascendc_dup_by_rows_fp16_to_fp32.h"

#endif  // ASCENDC_KERNELS_H

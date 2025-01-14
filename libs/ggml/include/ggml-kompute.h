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

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_vk_device {
    int index;
    int type; // same as VkPhysicalDeviceType
    size_t heapSize;
    const char * name;
    const char * vendor;
    int subgroupSize;
    uint64_t bufferAlignment;
    uint64_t maxAlloc;
};

struct ggml_vk_device * ggml_vk_available_devices(size_t memoryRequired, size_t * count);
bool ggml_vk_get_device(struct ggml_vk_device * device, size_t memoryRequired, const char * name);
bool ggml_vk_has_vulkan(void);
bool ggml_vk_has_device(void);
struct ggml_vk_device ggml_vk_current_device(void);

//
// backend API
//

// forward declaration
typedef struct ggml_backend * ggml_backend_t;

GGML_API ggml_backend_t ggml_backend_kompute_init(int device);

GGML_API bool ggml_backend_is_kompute(ggml_backend_t backend);

GGML_API ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(int device);

#ifdef __cplusplus
}
#endif

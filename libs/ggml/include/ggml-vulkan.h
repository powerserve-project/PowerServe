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

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_VK_NAME "Vulkan"
#define GGML_VK_MAX_DEVICES 16

GGML_API void ggml_vk_instance_init(void);

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num);

GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);
GGML_API GGML_CALL int  ggml_backend_vk_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif

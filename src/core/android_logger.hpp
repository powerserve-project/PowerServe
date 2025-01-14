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

#include <android/log.h>

namespace powerserve {

#define POWERSERVE_LOG_DEBUG(...)                                                                                      \
    __android_log_write(ANDROID_LOG_DEBUG, "PowerServe", fmt::format("" __VA_ARGS__).c_str())

#define POWERSERVE_LOG_INFO(...)                                                                                       \
    __android_log_write(ANDROID_LOG_INFO, "PowerServe", fmt::format("" __VA_ARGS__).c_str())

#define POWERSERVE_LOG_WARN(...)                                                                                       \
    __android_log_write(ANDROID_LOG_WARN, "PowerServe", fmt::format("" __VA_ARGS__).c_str())

#define POWERSERVE_LOG_ERROR(...)                                                                                      \
    __android_log_write(ANDROID_LOG_ERROR, "PowerServe", fmt::format("" __VA_ARGS__).c_str())

} // namespace powerserve

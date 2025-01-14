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

#include <string>

namespace powerserve {

template <typename T>
T getenv(const std::string &name, const T &default_value) {
    auto env = ::getenv(name.c_str());
    if (env) {
        if constexpr (std::is_integral_v<T>) {
            return atoll(env);
        } else if constexpr (std::is_floating_point_v<T>) {
            return atof(env);
        } else {
            return std::string(env);
        }
    } else {
        return default_value;
    }
}

} // namespace powerserve

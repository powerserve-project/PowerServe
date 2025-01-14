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

#include "llama-vocab.h"

#include <array>
#include <filesystem>

namespace powerserve {

using Path  = std::filesystem::path;
using Token = llama_vocab::id;

static constexpr size_t max_n_dims = 4;
using Shape                        = std::array<size_t, max_n_dims>;
using Stride                       = std::array<size_t, max_n_dims>;

struct Noncopyable {
    Noncopyable(const Noncopyable &)    = delete;
    auto operator=(const Noncopyable &) = delete;

protected:
    Noncopyable()  = default;
    ~Noncopyable() = default;
};

} // namespace powerserve

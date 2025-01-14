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

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct range_nfd {
    uint32_t first;
    uint32_t last;
    uint32_t nfd;
};

static const uint32_t MAX_CODEPOINTS = 0x110000;

extern const std::vector<std::pair<uint32_t, uint16_t>> unicode_ranges_flags;
extern const std::unordered_set<uint32_t> unicode_set_whitespace;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_lowercase;
extern const std::unordered_map<uint32_t, uint32_t> unicode_map_uppercase;
extern const std::vector<range_nfd> unicode_ranges_nfd;

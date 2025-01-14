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

#include <chrono>

namespace powerserve {

auto timestamp_ns() -> int64_t;
auto timestamp_us() -> int64_t;
auto timestamp_ms() -> int64_t;

struct Timer {
    Timer();

    // Return elapsed time since last tick.
    auto elapsed_time_ns() const -> int64_t;
    auto elapsed_time_us() const -> int64_t;
    auto elapsed_time_ms() const -> int64_t;

    // Return elapsed time since last tick and set new tick.
    auto tick_ns() -> int64_t;
    auto tick_us() -> int64_t;
    auto tick_ms() -> int64_t;

    void reset();

private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point last_time_point;

    auto tick_impl(Clock::time_point *out_time_point) const -> int64_t;
};

} // namespace powerserve

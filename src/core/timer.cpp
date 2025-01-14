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

#include "core/timer.hpp"

namespace {

constexpr auto div_round(int64_t a, int64_t b) -> int64_t {
    return (a + b - 1) / b;
}

} // namespace

namespace powerserve {

auto timestamp_ns() -> int64_t {
    auto timepoint = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(timepoint).count();
}

auto timestamp_us() -> int64_t {
    return div_round(timestamp_ns(), 1000);
}

auto timestamp_ms() -> int64_t {
    return div_round(timestamp_ns(), 1000000);
}

Timer::Timer() {
    reset();
}

auto Timer::elapsed_time_ns() const -> int64_t {
    return tick_impl(nullptr);
}

auto Timer::elapsed_time_us() const -> int64_t {
    return div_round(elapsed_time_ns(), 1000);
}

auto Timer::elapsed_time_ms() const -> int64_t {
    return div_round(elapsed_time_ns(), 1000000);
}

auto Timer::tick_ns() -> int64_t {
    return tick_impl(&last_time_point);
}

auto Timer::tick_us() -> int64_t {
    return div_round(tick_ns(), 1000);
}

auto Timer::tick_ms() -> int64_t {
    return div_round(tick_ns(), 1000000);
}

void Timer::reset() {
    tick_ns();
}

auto Timer::tick_impl(Clock::time_point *out_time_point) const -> int64_t {
    auto current_time_point = Clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - last_time_point);
    if (out_time_point) {
        *out_time_point = current_time_point;
    }
    return elapsed_time.count();
}

} // namespace powerserve

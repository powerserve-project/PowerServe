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

#include "fmt/core.h"
#include "fmt/format.h"

#include <chrono>
#include <cstddef>

namespace powerserve {

struct CPUPerfResult {
    size_t total_user_time;
    size_t total_user_low_time;
    size_t total_system_time;
    size_t total_idle_time;
    size_t total_irq_time;
    size_t total_softirq_time;

    CPUPerfResult &operator-=(const CPUPerfResult &other) {
        total_user_time -= other.total_user_time;
        total_user_low_time -= other.total_user_low_time;
        total_system_time -= other.total_system_time;
        total_idle_time -= other.total_idle_time;
        total_irq_time -= other.total_irq_time;
        total_softirq_time -= other.total_softirq_time;
        return *this;
    }

    CPUPerfResult operator-(const CPUPerfResult &other) const {
        CPUPerfResult ret = *this;
        ret -= other;
        return ret;
    }

    std::string to_string() const {
        return fmt::format(
            "[CPU times(us)] user: {}; user_low: {}, system: {}, idle: {}, irq: {}, softirq: {}",
            total_user_time,
            total_user_low_time,
            total_system_time,
            total_idle_time,
            total_irq_time,
            total_softirq_time
        );
    }
};

struct IOPerfResult {
    // In bytes
    size_t total_bytes_read;
    // In bytes
    size_t total_bytes_write;

    IOPerfResult &operator-=(const IOPerfResult &other) {
        total_bytes_read -= other.total_bytes_read;
        total_bytes_write -= other.total_bytes_write;
        return *this;
    }

    IOPerfResult operator-(const IOPerfResult &other) const {
        IOPerfResult ret = *this;
        ret -= other;
        return ret;
    }

    std::string to_string() const {
        return fmt::format(
            "[I/O throughput(MB)] total read: {}, total write: {}",
            total_bytes_read / 1024 / 1024,
            total_bytes_write / 1024 / 1024
        );
    }
};

struct MemPerfResult {
    // In bytes
    size_t virtual_memory_size = 0;
    // In bytes
    size_t resident_set_size = 0;

    std::string to_string() const {
        return fmt::format(
            "[Memory(MB)] VMS: {}, RSS: {}", virtual_memory_size / 1024 / 1024, resident_set_size / 1024 / 1024
        );
    }
};

CPUPerfResult perf_get_cpu_result();

IOPerfResult perf_get_io_result();

MemPerfResult perf_get_mem_result();

} // namespace powerserve

template <>
struct fmt::formatter<powerserve::CPUPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const powerserve::CPUPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};

template <>
struct fmt::formatter<powerserve::IOPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const powerserve::IOPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};

template <>
struct fmt::formatter<powerserve::MemPerfResult> : fmt::formatter<std::string> {
    template <typename FormatContext>
    auto format(const powerserve::MemPerfResult &data, FormatContext &ctx) const {
        return formatter<std::string>::format(data.to_string(), ctx);
    }
};

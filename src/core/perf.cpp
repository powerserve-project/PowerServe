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

#include "perf.hpp"

#include "logger.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>

namespace powerserve {

CPUPerfResult perf_get_cpu_result() {
    constexpr std::string_view stat_file = "/proc/stat";
    std::ifstream file(stat_file.data());
    if (!file.good()) [[unlikely]] {
        POWERSERVE_LOG_ERROR("failed reading file: {}", stat_file);
        return {};
    }

    CPUPerfResult ret;
    /// The first line of "/proc/stat"
    std::string line;
    if (std::getline(file, line)) {
        sscanf(
            line.c_str(),
            "cpu %zu %zu %zu %zu %zu %zu",
            &ret.total_user_time,
            &ret.total_user_low_time,
            &ret.total_system_time,
            &ret.total_idle_time,
            &ret.total_irq_time,
            &ret.total_softirq_time
        );
        return ret;
    }
    POWERSERVE_LOG_ERROR("failed reading stat");
    return {};
}

IOPerfResult perf_get_io_result() {
    constexpr std::string_view stat_file = "/proc/self/io";
    std::ifstream file(stat_file.data());
    if (!file.good()) [[unlikely]] {
        POWERSERVE_LOG_ERROR("failed reading file: {}", stat_file);
        return {};
    }

    IOPerfResult ret;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("read_bytes") != std::string::npos) {
            sscanf(line.c_str(), "read_bytes: %zu", &ret.total_bytes_read);
        }
        if (line.find("write_bytes") != std::string::npos) {
            sscanf(line.c_str(), "write_bytes: %zu", &ret.total_bytes_write);
        }
    }
    return ret;
}

MemPerfResult perf_get_mem_result() {
#if defined(__ANDROID__)
    std::ifstream file("/proc/self/statm");
    std::string line;
    std::getline(file, line);

    size_t pages, resident, shared, text, lib, data, dt;
    sscanf(line.c_str(), "%zu %zu %zu %zu %zu %zu %zu", &pages, &resident, &shared, &text, &lib, &data, &dt);

    const long page_size = sysconf(_SC_PAGESIZE);
    const size_t rss     = resident * page_size;
    const size_t vms     = pages * page_size;

    return {vms, rss};
#elif defined(__linux__)
    std::ifstream file("/proc/self/status");
    std::string line;

    MemPerfResult ret;
    while (std::getline(file, line)) {
        if (line.find("VmRSS") != std::string::npos) {
            sscanf(line.c_str(), "VmRSS: %zu", &ret.resident_set_size);
            ret.resident_set_size *= 1024;
        }
        if (line.find("VmSize") != std::string::npos) {
            sscanf(line.c_str(), "VmSize: %zu", &ret.virtual_memory_size);
            ret.virtual_memory_size *= 1024;
        }
    }
    return ret;
#else
    POWERSERVE_LOG_ERROR("unimplement platform for perf");
    return {};
#endif
}

} // namespace powerserve

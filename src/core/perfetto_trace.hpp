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

#include "core/defines.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"

namespace powerserve {

struct PerfettoTrace final : Noncopyable {
    // This is a singleton class.
    static auto instance() -> PerfettoTrace & {
        static PerfettoTrace instance;
        return instance;
    }

    // Default buffer size is 32 MiB
    void start_tracing(size_t buffer_size_kb = 32 * 1024);
    void stop_tracing(const Path &output_path = "./perfetto.data");
    void enable();
    void disable();

    // NOTE: name must be a static string
    static void begin(const char *name) {
#if defined(POWERSERVE_WITH_PERFETTO)
        if (instance().enabled.load(std::memory_order_relaxed)) {
            instance().begin_event(name, true);
        }
#else
        POWERSERVE_UNUSED(name);
#endif
    }

    template <typename... Args>
    static void begin(const char *name_fmt, Args &&...args) {
#if defined(POWERSERVE_WITH_PERFETTO)
        if (instance().enabled.load(std::memory_order_relaxed)) {
            auto name = fmt::vformat(name_fmt, fmt::make_format_args(args...));
            instance().begin_event(name.c_str(), false);
        }
#else
        POWERSERVE_UNUSED(name_fmt);
        (POWERSERVE_UNUSED(args), ...);
#endif
    }

    static void end() {
#if defined(POWERSERVE_WITH_PERFETTO)
        if (instance().enabled.load(std::memory_order_relaxed)) {
            instance().end_event();
        }
#endif
    }

    struct CounterTimePoint {
        uint64_t timestamp;

        // NOTE: name must be a static string
        void set_value(const char *name, double value);
        uint64_t elapsed_ns() const;
    };

    // Value can be set later
    static auto counter() -> CounterTimePoint;

    static void counter(const char *name, double value) {
#if defined(POWERSERVE_WITH_PERFETTO)
        counter().set_value(name, value);
#else
        POWERSERVE_UNUSED(name);
        POWERSERVE_UNUSED(value);
#endif
    }

private:
    std::atomic<bool> enabled = false;

    void begin_event(const char *name, bool static_name);
    void end_event();
};

} // namespace powerserve

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

#ifdef POWERSERVE_ANDROID_LOG
#include "android_logger.hpp"
#endif // POWERSERVE_ANDROID_LOG

#include "exception.hpp"
#include "fmt/base.h"
#include "fmt/ranges.h"
#include "fmt/std.h"
#include "perf.hpp"

#include <cstdio>
#include <string>
#include <string_view>

#ifndef POWERSERVE_LOG_DEBUG
#define POWERSERVE_LOG_DEBUG(...) fmt::println(stdout, "[DEBUG] " __VA_ARGS__)
#endif // POWERSERVE_LOG_DEBUG

#ifndef POWERSERVE_LOG_INFO
#define POWERSERVE_LOG_INFO(...) fmt::println(stdout, "[INFO ] " __VA_ARGS__)
#endif // POWERSERVE_LOG_INFO

#ifndef POWERSERVE_LOG_WARN
#define POWERSERVE_LOG_WARN(...) fmt::println(stderr, "[WARN ] " __VA_ARGS__)
#endif // POWERSERVE_LOG_WARN

#ifndef POWERSERVE_LOG_ERROR
#define POWERSERVE_LOG_ERROR(...) fmt::println(stderr, "[ERROR] " __VA_ARGS__)
#endif // POWERSERVE_LOG_ERROR

#ifndef POWERSERVE_LOG_EMPTY_LINE
#define POWERSERVE_LOG_EMPTY_LINE()                                                                                    \
    {                                                                                                                  \
        fflush(stdout);                                                                                                \
        fflush(stderr);                                                                                                \
        fmt::println(stdout, "");                                                                                      \
    }
#endif // POWERSERVE_LOG_EMPTY_LINE

#ifndef POWERSERVE_ABORT
#define POWERSERVE_ABORT(...)                                                                                          \
    do {                                                                                                               \
        fflush(stdout);                                                                                                \
        fflush(stderr);                                                                                                \
        POWERSERVE_LOG_ERROR("{}:{}: {}: Abort", __FILE__, __LINE__, __func__);                                        \
        POWERSERVE_LOG_ERROR("" __VA_ARGS__);                                                                          \
        POWERSERVE_LOG_ERROR("System error: {}", ::powerserve::get_system_error());                                    \
        abort();                                                                                                       \
    } while (0)
#endif // POWERSERVE_ABORT

#if defined(POWERSERVE_NO_ASSERT)
#define POWERSERVE_ASSERT(expr) POWERSERVE_UNUSED(expr)
#elif !defined(POWERSERVE_ASSERT)
#define POWERSERVE_ASSERT(expr, ...)                                                                                   \
    do {                                                                                                               \
        if (!(expr)) [[unlikely]] {                                                                                    \
            fflush(stdout);                                                                                            \
            fflush(stderr);                                                                                            \
            POWERSERVE_LOG_ERROR("{}:{}: {}: Assertion failed: {}", __FILE__, __LINE__, __func__, #expr);              \
            POWERSERVE_LOG_ERROR("System error: {}", ::powerserve::get_system_error());                                \
            POWERSERVE_LOG_ERROR("" __VA_ARGS__);                                                                      \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // POWERSERVE_ASSERT

namespace powerserve {

inline void print_timestamp() {
    POWERSERVE_LOG_INFO("Compiled on: {} at {}", __DATE__, __TIME__);
}

inline std::string abbreviation(std::string text, size_t limit) {
    auto len = text.length();
    if (len > limit) {
        return fmt::format("{}...[omit {} chars]", text.substr(0, limit), len - limit);
    }
    return text;
}

// trim whitespace from the beginning and end of a string
inline std::string trim(const std::string &str) {
    size_t start = 0;
    size_t end   = str.size();
    while (start < end && isspace(str[start])) {
        start += 1;
    }
    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

} // namespace powerserve

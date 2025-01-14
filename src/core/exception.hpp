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
#include "fmt/std.h"

#include <cerrno>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>

#if defined(__ANDROID__)
#include <cstdio>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unwind.h>
#elif defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
#include <sys/wait.h>
#else
#include <unistd.h>
#endif

namespace powerserve {

/*!
 * @brief Query the error of OS and format it as a string.
 */
inline std::string get_system_error() {
#ifdef _WIN32
    // TODO: include windows headers
    const DWORD error = getLastError();
    LPSTR buf;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&buf,
        0,
        NULL
    );
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
#else
    return {strerror(errno)};
#endif
}

constexpr int MAX_NUM_BACKTRACE_STACK = 100;
#if defined(__ANDROID__)
struct backtrace_state {
    void **current;
    void **end;
};

static _Unwind_Reason_Code unwind_callback(struct _Unwind_Context *context, void *arg) {
    struct backtrace_state *state = static_cast<struct backtrace_state *>(arg);
    uintptr_t pc                  = _Unwind_GetIP(context);
    if (pc) {
        if (state->current == state->end) {
            return _URC_END_OF_STACK;
        } else {
            *state->current++ = (void *)pc;
        }
    }
    return _URC_NO_REASON;
}

inline void print_backtrace_symbols() {
    void *trace[MAX_NUM_BACKTRACE_STACK];

    struct backtrace_state state = {trace, trace + MAX_NUM_BACKTRACE_STACK};
    _Unwind_Backtrace(unwind_callback, &state);

    const int count = state.current - trace;
    for (int i = 0; i < count; ++i) {
        const void *addr   = trace[i];
        const char *symbol = "";

        Dl_info info;
        if (dladdr(addr, &info) && info.dli_sname) {
            symbol = info.dli_sname;
        }

        fprintf(stderr, "%d: %p %s\n", i, addr, symbol);
    }
}
#elif defined(__linux__) && defined(__GLIBC__)
inline void print_backtrace_symbols() {
    void *trace[MAX_NUM_BACKTRACE_STACK];
    const int num_stack = backtrace(trace, sizeof(trace) / sizeof(void *));
    backtrace_symbols_fd(trace, num_stack, STDERR_FILENO);
}
#endif

#if (defined(__linux__) || defined(__ANDROID__)) && !defined(__OHOS__)
inline void print_backtrace() {
    char attach[32];
    snprintf(attach, sizeof(attach), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        // try gdb
        execlp(
            "gdb",
            "gdb",
            "--batch",
            "-ex",
            "set style enabled on",
            "-ex",
            attach,
            "-ex",
            "bt -frame-info source-and-location",
            "-ex",
            "detach",
            "-ex",
            "quit",
            (char *)NULL
        );
        // try lldb
        execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", attach, (char *)NULL);
        exit(EXIT_FAILURE);
    } else {
        int wstatus;
        waitpid(pid, &wstatus, 0);
        if (WIFEXITED(wstatus)) {
            if (WEXITSTATUS(wstatus) == EXIT_FAILURE) {
                // gdb failed, fallback to backtrace_symbols
                print_backtrace_symbols();
            }
        }
    }
}
#else
static void print_backtrace() {
    // platform not supported
}
#endif

#define POWERSERVE_EXP_ASSERT(expr, Exception, ...)                                                                    \
    do {                                                                                                               \
        if (!(expr)) [[unlikely]] {                                                                                    \
            throw Exception(__VA_ARGS__);                                                                              \
        }                                                                                                              \
    } while (false)

class BasicException : public std::exception {
protected:
    /// Container for detailed exception message
    std::string m_content;

public:
    BasicException(const std::string_view tag, const std::string_view message) :
        m_content(fmt::format("[Exception][{}] BasicException: {}", tag, message)) {}

    ~BasicException() noexcept = default;

public:
    const char *what() const noexcept override {
        return m_content.c_str();
    }
};

#define POWERSERVE_ASSERT_BASIC(expr, tag, ...)                                                                        \
    POWERSERVE_EXP_ASSERT(expr, ::powerserve::BasicException, tag, fmt::format("" __VA_ARGS__))

///
/// @brief Exception caused by logically impossible events (e.g. unknown switch cases)
/// @note  When receiving this exception, it may be quite difficult for application to
/// figure out the cause of exception. A totally reboot may be useful. Or you can just
/// give feedback to developers.
/// @note  Developers should avoid throw this exception as possible.
///
class AbortException final : public BasicException {
public:
    AbortException(
        const std::string_view tag,
        const std::string_view message,
        const std::string_view file_name,
        const std::string_view line,
        const std::string_view func_name
    ) :
        BasicException(tag, "AbortException") {
        m_content += fmt::format("\n[Exception][{}] Source {}: {}: {}", tag, file_name, line, func_name);
        m_content += fmt::format("\n[Exception][{}] AbortException: {}", tag, message);
        m_content += fmt::format("\n[Exception][{}] System error: {}", tag, get_system_error());
    }

    ~AbortException() noexcept = default;
};

#define POWERSERVE_ASSERT_ABORT(expr, tag, ...)                                                                        \
    POWERSERVE_EXP_ASSERT(expr, ::powerserve::AbortException, tag, fmt::format("" __VA_ARGS__))

///
/// @brief Exception caused by invalid config arguments
/// @note  It may be of no meaning to handle it when using PowerServe as CLI. However,
/// when integrating this framework into some application or using the server, user may
/// input invalid config entry (e.g. invalid model path, invalid sampler params), we need
/// to inform them with detailed information rather than closing the whole pragram directly.
/// Users can rapidly fix their input accordingly.
///
class ConfigException final : public BasicException {
public:
    ConfigException(const std::string_view tag, const std::string_view message) :
        BasicException(tag, "ConfigException") {
        m_content += fmt::format("\n[Exception][{}] ConfigException: {}", tag, message);
    }

    ~ConfigException() noexcept = default;
};

#define POWERSERVE_ASSERT_CONFIG(expr, tag, ...)                                                                       \
    POWERSERVE_EXP_ASSERT(expr, ::powerserve::ConfigException, tag, fmt::format("" __VA_ARGS__))

///
/// @brief Exception caused by execution environment (e.g. OS, filesystem, SDK runtime)
/// @note  Most of the time, we cannot handle this exception and have to abort the program.
/// However, in some scenary, there may be some application laid on this framework. Therefore,
/// we still need to inform these apps with detailed information to aid fixing the error manually
/// (e.g. assign more priviledge) instead of printing meaningless assert.
///
class EnvironmentException final : public BasicException {
public:
    EnvironmentException(const std::string_view tag, const std::string_view message) :
        BasicException(tag, "EnvironmentException") {
        m_content += fmt::format("\n[Exception][{}] EnvironmentException: {}", tag, message);
        m_content += fmt::format("\n[Exception][{}] System error: {}", tag, get_system_error());
    }

    ~EnvironmentException() noexcept = default;
};

#define POWERSERVE_ASSERT_ENV(expr, tag, ...)                                                                          \
    POWERSERVE_EXP_ASSERT(expr, ::powerserve::EnvironmentException, tag, fmt::format("" __VA_ARGS__))

///
/// @brief Exception caused by non-system module (tokenizer, sampler)
/// @note  LLM algorithm varys. It may introduce some unexpected error when integrating
/// this system with these new things. Users can consider to use another counterpart.
///
class ModuleException final : public BasicException {
public:
    ModuleException(const std::string_view tag, const std::string module_name, const std::string_view message) :
        BasicException(tag, "ModuleException") {
        m_content += fmt::format("\n[Exception][{}] Module {}:", tag, module_name);
        m_content += fmt::format("\n[Exception][{}] ModuleException: {}", tag, message);
    }

    ~ModuleException() noexcept = default;
};

#define POWERSERVE_ASSERT_MODULE(expr, tag, module, ...)                                                               \
    POWERSERVE_EXP_ASSERT(expr, ::powerserve::ModuleException, tag, module, fmt::format("" __VA_ARGS__))

#ifdef POWERSERVE_EXCEPTION_ABORT
#define POWERSERVE_ABORT(...)        throw AbortException(__func__, fmt::format("" __VA_ARGS__), __FILE__, __LINE__, __func__)
#define POWERSERVE_ASSERT(expr, ...) throw AssertException(__func__, fmt::format("Assert(" #expr ") " __VA_ARGS__))
#endif // POWERSERVE_EXCEPTION_ABORT

} // namespace powerserve

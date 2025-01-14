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

#include "core/perfetto_trace.hpp"

#if defined(POWERSERVE_WITH_PERFETTO)

#include "perfetto.h"

constexpr const char *event_category   = "event";
constexpr const char *counter_category = "counter";

// clang-format off
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(event_category),
    perfetto::Category(counter_category),
);
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
// clang-format on

namespace {

// We do not want to include perfetto.h in perfetto_trace.hpp
// Therefore tracing object is defined in C++ source
static std::unique_ptr<perfetto::TracingSession> tracing;

} // namespace

#endif

namespace powerserve {

void PerfettoTrace::start_tracing(size_t buffer_size_kb) {
#if defined(POWERSERVE_WITH_PERFETTO)
    POWERSERVE_ASSERT(!tracing);

    perfetto::TracingInitArgs args;
    args.backends            = perfetto::kInProcessBackend;
    args.use_monotonic_clock = true;
    perfetto::Tracing::Initialize(args);
    POWERSERVE_ASSERT(perfetto::TrackEvent::Register());

    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(buffer_size_kb);
    auto *ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");

    tracing = perfetto::Tracing::NewTrace();
    tracing->Setup(cfg);
    tracing->StartBlocking();

    enabled = true;
#else
    POWERSERVE_UNUSED(buffer_size_kb);
#endif
}

void PerfettoTrace::stop_tracing(const Path &output_path) {
#if defined(POWERSERVE_WITH_PERFETTO)
    POWERSERVE_ASSERT(tracing);

    fmt::println("Saving Perfetto trace data to {:?}...", output_path);

    perfetto::TrackEvent::Flush();
    tracing->StopBlocking();
    std::vector<char> trace_data(tracing->ReadTraceBlocking());

    std::ofstream output_file;
    output_file.open(output_path, std::ios::out | std::ios::binary);

    if (!output_file) {
        fmt::println(stderr, "Cannot open {:?}.", output_path);
        abort();
    }

    output_file.write(trace_data.data(), trace_data.size());
    output_file.close();

    tracing.reset();
    perfetto::Tracing::Shutdown();
    enabled = false;
#else
    POWERSERVE_UNUSED(output_path);
#endif
}

void PerfettoTrace::enable() {
#if defined(POWERSERVE_WITH_PERFETTO)
    enabled = true;
#endif
}

void PerfettoTrace::disable() {
#if defined(POWERSERVE_WITH_PERFETTO)
    enabled = false;
#endif
}

auto PerfettoTrace::counter() -> CounterTimePoint {
#if defined(POWERSERVE_WITH_PERFETTO)
    if (instance().enabled.load(std::memory_order_relaxed)) {
        return CounterTimePoint{
            .timestamp = perfetto::TrackEvent::GetTraceTimeNs(),
        };
    } else {
        return CounterTimePoint(); // Return a dummy object
    }
#else
    return CounterTimePoint();
#endif
}

void PerfettoTrace::CounterTimePoint::set_value(const char *name, double value) {
#if defined(POWERSERVE_WITH_PERFETTO)
    if (instance().enabled.load(std::memory_order_relaxed)) {
        TRACE_COUNTER(counter_category, perfetto::StaticString{name}, timestamp, value);
    }
#else
    POWERSERVE_UNUSED(name);
    POWERSERVE_UNUSED(value);
#endif
}

uint64_t PerfettoTrace::CounterTimePoint::elapsed_ns() const {
#if defined(POWERSERVE_WITH_PERFETTO)
    return perfetto::TrackEvent::GetTraceTimeNs() - timestamp;
#else
    return 1; // Prevent division by zero
#endif
}

void PerfettoTrace::begin_event(const char *name, bool static_name) {
#if defined(POWERSERVE_WITH_PERFETTO)
    if (static_name) {
        TRACE_EVENT_BEGIN(event_category, perfetto::StaticString{name});
    } else {
        TRACE_EVENT_BEGIN(event_category, perfetto::DynamicString{name});
    }
#else
    POWERSERVE_UNUSED(name);
    POWERSERVE_UNUSED(static_name);
#endif
}

void PerfettoTrace::end_event() {
#if defined(POWERSERVE_WITH_PERFETTO)
    TRACE_EVENT_END(event_category);
#endif
}

} // namespace powerserve

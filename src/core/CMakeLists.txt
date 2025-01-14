target_sources(powerserve PRIVATE
    "config.cpp"
    "perf.cpp"
    "perfetto_trace.cpp"
    "thread_pool.cpp"
    "timer.cpp"
    "spin_barrier.cpp"
)

if (POWERSERVE_WITH_PERFETTO)
    target_compile_definitions(powerserve PUBLIC POWERSERVE_WITH_PERFETTO)
endif()

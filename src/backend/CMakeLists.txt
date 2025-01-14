target_sources(powerserve PRIVATE
    "platform.cpp"
)

add_subdirectory(ggml)

if (POWERSERVE_WITH_QNN)
    add_subdirectory(qnn)
endif()

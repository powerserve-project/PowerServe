add_library(powerserve STATIC)
target_include_directories(powerserve PUBLIC .)

target_link_libraries(powerserve PUBLIC
    fmt
    ggml
    llama_tokenizer
    nlohmann_json::nlohmann_json
)

if (NOT OHOS)
    message(STATUS "Introduce uv a")
    target_link_libraries(powerserve PUBLIC uv_a)
endif()

if (POWERSERVE_WITH_PERFETTO)
    target_link_libraries(powerserve PRIVATE perfetto)
endif()

if (POWERSERVE_ANDROID_LOG)
    add_compile_definitions(POWERSERVE_ANDROID_LOG)
    target_link_libraries(powerserve PUBLIC log)
endif ()

if (NOT MSVC)
    target_compile_options(powerserve PRIVATE
        -Wall
        -Wextra

        -Wno-unused-function
    )

    if (POWERSERVE_ENABLE_WERROR)
        target_compile_options(powerserve PRIVATE -Werror)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(powerserve PRIVATE -Wno-unknown-pragmas)
endif()

add_subdirectory(storage)
add_subdirectory(backend)
add_subdirectory(core)
add_subdirectory(executor)
add_subdirectory(graph)
add_subdirectory(model)
add_subdirectory(sampler)
if (POWERSERVE_WITH_QNN)
    add_subdirectory(speculative)
endif()
add_subdirectory(tokenizer)

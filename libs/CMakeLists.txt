# CLI11
add_subdirectory(cli11)

# fmt
add_subdirectory(fmt)

# ggml
set(GGML_OPENMP OFF)
add_subdirectory(ggml)

if (${CMAKE_C_COMPILER_ID} MATCHES "Clang")
    target_compile_options(ggml PRIVATE -Wno-double-promotion -Wno-unreachable-code-break)
endif()

# json
add_subdirectory(json)

if (NOT OHOS)
    # libuv
    set(LIBUV_BUILD_SHARED OFF CACHE BOOL "Build shared lib")
    add_subdirectory(libuv)

    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(uv_a PRIVATE -Wno-missing-prototypes -Wno-unreachable-code-return -Wno-shadow)
    elseif (CMAKE_C_COMPILER_ID MATCHES "GNU")
        target_compile_options(uv_a PRIVATE -Wno-cast-qual -Wno-missing-prototypes -Wno-shadow)
    endif()
endif()

# llama_tokenizer
add_subdirectory(llama_tokenizer)

# QNN headers
add_subdirectory(qnn_headers)

# xtl(Required by xtensor)
add_subdirectory(xtl)

# xtensor
add_subdirectory(xtensor)

add_subdirectory(stb_headers)

# http lib
add_subdirectory(cpp-httplib)

# concurrent queue
add_subdirectory(concurrentqueue)

# Perfetto
if (POWERSERVE_WITH_PERFETTO)
    add_library(perfetto STATIC perfetto/sdk/perfetto.cc)
    target_include_directories(perfetto PUBLIC perfetto/sdk)
    if (ANDROID)
        target_link_libraries(perfetto PRIVATE -llog)
    endif()
endif()

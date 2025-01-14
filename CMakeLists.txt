cmake_minimum_required(VERSION 3.16)
project("PowerServe" C CXX)

option(POWERSERVE_ENABLE_ASAN "Enable address sanitizer" OFF)
option(POWERSERVE_ENABLE_UBSAN "Enable undefined sanitizer" OFF)
option(POWERSERVE_ENABLE_LTO "Enable link-time optimization" OFF)
option(POWERSERVE_ENABLE_WERROR "Enable -Werror" OFF)

option(POWERSERVE_WITH_QNN "Enable QNN backend" OFF)
option(POWERSERVE_WITH_PERFETTO "Enable Perfetto tracing" OFF)
option(POWERSERVE_ANDROID_LOG "Replace the underlying interface of logger with android log interface" OFF)
option(POWERSERVE_EXCEPTION_ABORT "Replace underlying abort/assert with raising exception" OFF)
option(POWERSERVE_SERVER_MULTIMODEL "Cache all models used in the memory" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

# Remove -DNDEBUG from the default Release flags
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

# Remove -DNDEBUG from the default RelWithDebInfo flags
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")

set(BUILD_SHARED_LIBS OFF)

if (POWERSERVE_ENABLE_ASAN)
    if (ANDROID)
        add_compile_options(-fsanitize=hwaddress -fno-omit-frame-pointer)
        link_libraries(-fsanitize=hwaddress)
    else()
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()
endif()

if (POWERSERVE_ENABLE_UBSAN)
    if (ANDROID)
        add_compile_options(-fsanitize=undefined,bounds -static-libsan)
        link_libraries(-fsanitize=undefined,bounds -static-libsan)
    else()
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (POWERSERVE_ENABLE_LTO)
    if (${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        add_compile_options(-flto=thin)
    else()
        add_compile_options(-flto)
    endif()
endif()

if (OHOS)
    add_compile_options(-Wno-unused-command-line-argument)
endif()

add_custom_target(
    MAKEDIR_OUT
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/out
)

function(powerserve_add_artifact target_name)
    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                $<TARGET_FILE:${target_name}>
                ${CMAKE_BINARY_DIR}/out/powerserve-${target_name}
    )
endfunction()

add_subdirectory(libs)
add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(tools)

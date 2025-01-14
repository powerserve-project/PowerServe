set(POWERSERVE_BINARY_DIR  ${PROJECT_BINARY_DIR}/bin)
message(STATUS "PowerServe binary dir: ${POWERSERVE_BINARY_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${POWERSERVE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${POWERSERVE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${POWERSERVE_BINARY_DIR})

function(powerserve_add_binary target_name)
    powerserve_add_artifact(${target_name})
endfunction()

add_subdirectory(common)

add_subdirectory(run)
add_subdirectory(perplexity)
add_subdirectory(server)

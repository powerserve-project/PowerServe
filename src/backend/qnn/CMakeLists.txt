set(QNN_SDK $ENV{QNN_SDK_ROOT})

if ("${QNN_SDK}" STREQUAL "")
    message(FATAL_ERROR "Environment variable \"QNN_SDK_ROOT\" is not defined.")
endif()

message("QNN_SDK_ROOT=${QNN_SDK}")

target_sources(powerserve PRIVATE
    "causal_models.cpp"
    "graph_interface.cpp"
    "qnn_backend.cpp"
    "qnn.cpp"
)
target_include_directories(powerserve PUBLIC ${QNN_SDK}/include/QNN)
target_compile_definitions(powerserve PUBLIC POWERSERVE_WITH_QNN)
target_link_libraries(powerserve PUBLIC qnn_headers)

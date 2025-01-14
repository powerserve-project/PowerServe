add_executable(quant_mul_mat "quant_mul_mat.cpp")
target_link_libraries(quant_mul_mat PRIVATE powerserve ggml)

if (POWERSERVE_WITH_QNN)
    add_executable(qnn_test qnn_test.cpp)
    target_link_libraries(qnn_test PRIVATE powerserve CLI11::CLI11)
endif()

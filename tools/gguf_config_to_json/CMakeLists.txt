add_executable(config-generator "main.cpp")
target_link_libraries(config-generator PRIVATE powerserve ggml CLI11::CLI11)

powerserve_add_artifact(config-generator)

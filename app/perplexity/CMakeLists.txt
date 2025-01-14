add_executable(perplexity-test "main.cpp")
target_link_libraries(perplexity-test PRIVATE powerserve ggml app.common)

powerserve_add_artifact(perplexity-test)

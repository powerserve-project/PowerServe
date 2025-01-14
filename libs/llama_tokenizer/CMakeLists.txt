add_library(llama_tokenizer STATIC
    "llama-vocab.cpp"
    "unicode-data.cpp"
    "unicode.cpp"
)
target_link_libraries(llama_tokenizer PRIVATE ggml)
target_include_directories(llama_tokenizer PUBLIC .)

if (CMAKE_C_COMPILER_ID MATCHES "Clang")
    target_compile_options(llama_tokenizer PRIVATE -Wno-deprecated-declarations)
endif()

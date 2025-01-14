#!/bin/bash

# 如果没有./build_android目录，则创建
if [ ! -d "./build_android" ]; then
    cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -S . -B build_android
fi

# 构建项目
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

# 使用 scp 命令传输文件
scp -P 8022 -r ./build_android/bin/speculative u0_a342@192.168.60.173:/data/data/com.termux/files/home/CI/bin

scp -P 8022 -r ./build_android/bin/run u0_a342@192.168.60.173:/data/data/com.termux/files/home/CI/bin

sleep 1

# 在远程 ssh 中执行命令
ssh -p 8022 u0_a342@192.168.60.173 "export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./CI/bin/speculative --target-path ~/qnn_models/2_28/3_1/llama_3.1_8b_q4_0.gguf --draft-path ~/models/llama3.2-1b-instr/llama_3.2_1b_q4_0.gguf --vocab-path ~/models/llama3.1-8b-instr/vocab.gguf --target-config-path ~/qnn_models/2_28/3_1/model_config.json --draft-config-path ~/qnn_models/2_28/3_2/model_config.json --target-qnn-path ~/qnn_models/2_28/3_1 --draft-qnn-path ~/qnn_models/2_28/3_2"

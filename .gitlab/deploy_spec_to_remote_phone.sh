#!/bin/bash

# 如果没有./build_android目录，则创建
if [ ! -d "./build_android" ]; then
    cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -S . -B build_android
fi

# 构建项目
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

# 使用 scp 命令通过跳板机传输文件
scp -o "ProxyJump=jmp@202.120.40.80" -P 8022 -r ./build_android/bin/speculative u0_a334@192.168.1.134:/data/data/com.termux/files/home

scp -o "ProxyJump=jmp@202.120.40.80" -P 8022 -r ./build_android/bin/run u0_a334@192.168.1.134:/data/data/com.termux/files/home

sleep 1

# 在远程 ssh 中通过跳板机执行命令
ssh -o "ProxyJump=jmp@202.120.40.80" -p 8022 u0_a334@192.168.1.134 "export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./speculative --target-path 3_1/llama_3.1_8b_q4_0.gguf --draft-path libdir16_i/llama_3.2_1b_q4_0.gguf --vocab-path 3_1/Llama-3.1-Instruct-vocab.gguf --target-config-path 3_1/model_config.json --draft-config-path 3_2/model_config.json --target-qnn-path ./3_1 --draft-qnn-path ./3_2"

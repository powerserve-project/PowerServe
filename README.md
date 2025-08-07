# PowerServe
PowerServe is a high-speed and easy-use LLM serving framework for local deployment.

## Features
- [One-click compilation and deployment](./docs/end_to_end.md)
- NPU speculative inference support
- Achieves 40 tokens/s running Smallthinker on mobile devices
- Support Android and HarmonyOS NEXT

## Supported Models

Here's the list of models that PowerServe supports:

| Model Name | Hugging Face Link | Speculation Support(Draft model) | Soc Setting | Prefill Speed (tokens/s) | Decode Speed (tokens/s) | Speculative Decode Speed (tokens/s) |
|---|---|---|---|---|---|---|
| smallthinker-3b | [SmallThinker-3B](https://huggingface.co/PowerServe/SmallThinker-3B-PowerServe-QNN29-8G3) | Yes(smallthinker-0.5b) | 8G3 | 975.00 | 19.71 | 38.75 |
| llama-3.2-1b | [Llama-3.2-1B](https://huggingface.co/PowerServe/Llama-3.2-1B-PowerServe-QNN29-8G3) | No | 8G3 | 1876.58 | 58.99 | / |
| llama-3.1-8b | [Llama-3.1-8B](https://huggingface.co/PowerServe/Llama-3.1-8B-PowerServe-QNN29-8G3) | Yes(llama-3.2-1b) | 8G3 | 468.35 | 12.03 | 21.02 |
| qwen-2-0.5b | [Qwen-2-0.5B](https://huggingface.co/PowerServe/Qwen-2-0.5B-PowerServe-QNN29-8G3) | No | 8G3 | 3590.91 | 104.53 | / |
| qwen-2.5-3b | [Qwen-2.5-3B](https://huggingface.co/PowerServe/Qwen-2.5-3B-PowerServe-QNN29-8G3) | No | 8G3 | 906.98 | 21.01 | / |
| internlm-3-8b | [InternLM-3-8B](https://huggingface.co/PowerServe/InternLM-3-8B-PowerServe-QNN29-8G3) | No | 8G3 | TBC | TBC | / |
| deepseek-r1-llama-8b | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/PowerServe/DeepSeek-R1-Distill-Llama-8B-PowerServe-QNN29-8G3/tree/main) | Yes(llama-3.2-1b) | 8G3 | TBC | TBC | / |
| smallthinker-3b | [SmallThinker-3B](https://huggingface.co/PowerServe/SmallThinker-3B-PowerServe-QNN29-8G4) | Yes(smallthinker-0.5b) | 8G4(8Elite) | 1052.63 | 20.90 | 43.25 |
| llama-3.2-1b | [Llama-3.2-1B](https://huggingface.co/PowerServe/Llama-3.2-1B-PowerServe-QNN29-8G4) | No | 8G4(8Elite) | 1952.38 | 59.00 | / |
| llama-3.1-8b | [Llama-3.1-8B](https://huggingface.co/PowerServe/Llama-3.1-8B-PowerServe-QNN29-8G4) | Yes(llama-3.2-1b) | 8G4(8Elite) | 509.09 | 12.48 | 22.83 |
| qwen-2-0.5b | [Qwen-2-0.5B](https://huggingface.co/PowerServe/Qwen-2-0.5B-PowerServe-QNN29-8G4) | No | 8G4(8Elite) | 4027.30 | 109.49 | / |
| qwen-2.5-3b | [Qwen-2.5-3B](https://huggingface.co/PowerServe/Qwen-2.5-3B-PowerServe-QNN29-8G4) | No | 8G4(8Elite) | 981.69 | 22.19 | / |
| internlm-3-8b | [InternLM-3-8B](https://huggingface.co/PowerServe/InternLM-3-8B-PowerServe-QNN29-8G4) | No | 8G4(8Elite) | 314.80 | 7.62 | / |
| deepseek-r1-llama-8b | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/PowerServe/DeepSeek-R1-Distill-Llama-8B-PowerServe-QNN29-8G4/tree/main) | Yes(llama-3.2-1b) | 8G4(8Elite) | 336.37 | 10.21 | / |

We test these speeds with files in `./assets/prompts`as input prompt files. More tests on multiple datasets will be conducted in the future.

## News
- [2025/1/14] We release PowerServe 🎉
- [2025/4/22] Bug fixes, more debug tools, backward compatability on more Qualcomm hardwares and more QOL updates. Check the new features and improvements in the [release notes](#release-notes).

## Table of Contents

1. [End to end deployment](#end-to-end)
2. [Prerequisites](#prerequisites)
3. [Directory Structure](#directory-structure)
4. [Model Preparation](#model-preparation)
5. [Compile PowerServe](#compile-powerserve)
6. [Prepare PowerServe Workspace](#prepare-powerserve-workspace)
7. [Execution](#execution)
8. [Known Issues](#known-issues)
9. [Release Notes](#release-notes)

## End to End Deployment

We provide nearly one-click end to end deployment document(./docs/end_to_end.md), including model downloading, compiling, deploying, and running.

No matter what operating systems you are using, you can follow the instructions in the document to use Powerserve to run support models on your phone.

Details please refer to [End to End Deployment](./docs/end_to_end.md)


## Prerequisites

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

To deploy on aarch64 with Qualcomm NPU using QNN, [**NDK**](https://developer.android.google.cn/ndk/downloads) and [**QNN**](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html) are required to be installed.

```shell
export NDK=<path-to-ndk>
export QNN_SDK_ROOT=<path-to-QNN>
```
## directory-structure
```
powerserve
├── app
├── assets               # Prompt files.
├── CMakeLists.txt
├── docs
├── libs                 # External dependencies.
├── LICENSE
├── powerserve           # Python script to create work directory.
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── backend          # Backend implementations, include ggml and qnn.
│   ├── CMakeLists.txt
│   ├── core             # Core structures used across all levels of the runtime, like type definition, config, tensor and buffer.
│   ├── executor         # Tensor execution.
│   ├── graph            # Computing Graph.
│   ├── model            # Various model implementations.
│   ├── sampler          # Token sampler.
│   ├── speculative      # Speculative decoding.
│   ├── storage          # File loader.
│   └── tokenizer
├── tests
└── tools
    ├── add_license.py
    ├── CMakeLists.txt
    ├── convert_hf_to_gguf   # Convert huggingface to gguf, based on llama.cpp
    ├── cos_sim.py
    ├── end_to_end
    ├── extract_embd_from_vl
    ├── format.py
    ├── gen_flame_graph.sh
    ├── gguf_config_to_json  # Export config.json from gguf.
    ├── gguf_export.py
    ├── mmlu
    ├── mmmu_test
    ├── parameter_search
    ├── qnn_converter
    └── simple_qnn_test
```

## Model Preparation

For CPU-only execution, only `Models For CPU` is required. For NPU execution, both `Models For CPU` and `Models For NPU` is required.

Take llama3.1-8b-instruct model as example, the structure of model folder:
```shell
-- models                       # Level-1 dir, where server search different models and CLI search for runtime configurations
    -- hparams.json                 # Hyper params, containing #threads, #batch_size and sampler configurations.
    -- workspace.json               # The definition of model workspace structure, where main model and target model(if exist) is determined.
    -- bin                          # The binaries for execution
        -- powerserve-config-generator
        -- powerserve-perplexity-test
        -- powerserve-run
        -- powerserve-server
    -- qnn_libs                     # Dependent libraries of QNN
        -- libQNNSystem.so
        -- libQNNHtp.so
        -- libQNNHtpV79.so
        -- libQNNHtpV79Skel.so
        -- libQNNHtpV79Stub.so
    -- llama3.1-8b-instruct         # The model weights of GGUF and QNN
        -- model.json
        -- vocab.gguf               # The vocab table of model
        -- ggml                     # GGUF model binaries
            -- weights.gguf
        -- qnn                      # QNN model binaries
            -- kv
                -- *.raw
                -- ...
            -- config.json          # The information of QNN models and QNN backend configurations
            -- llama3_1_8b_0.bin
            -- llama3_1_8b_1.bin
            -- llama3_1_8b_2.bin
            -- llama3_1_8b_3.bin
            -- lmhead.bin
    -- qwen2_7b_instruct            # another model
        -- ...

```

### Convert Models For CPU

```shell
# Under the root directory of PowerServe
python ./tools/gguf_export.py -m <hf-model> -o models/llama3.1-8b-instruct
```


### Convert Models For NPU

If you just want to run PowerServe on CPUs, this step can be skipped. More details please refer to [QNN Model Conversion](./tools/qnn_converter/README.md)

```shell
# Under the root directory of PowerServe
cd powerserve/tools/qnn_converter

# This may take a long time...
python converter.py                                 \
    --model-folder Llama-3.1-8B-Instruct            \
    --model-name llama3_1_8b                        \
    --system-prompt-file system_prompt_llama.txt    \
    --prompt-file lab_intro_llama.md                \
    --batch-sizes 1 128                             \
    --artifact-name llama3_1_8b                     \
    --n-model-chunk 4                               \
    --output-folder ./llama3.1-8b-QNN               \
    --build-folder ./llama3.1-8b-QNN-tmp            \
    --silent \
    --clear-build-files \
    --soc 8650

```
Convert GGUF models and integrate them with QNN models

Note: this scripts can only create fp32 and q8_0 in ./llama3.1-8b-instruct-model/ggml/weights.gguf,
if you want to use q4_0, please use llama-quantize in llama.cpp like: `./build/bin/llama-quantize --pure /<path>/llama3.1-fp32.gguf Q4_0`, then replace weight file: `cp /<path>/ggml-model-Q4_0.gguf ./llama3.1-8b-instruct-model/ggml/weights.gguf`

```shell
# Under the root directory of PowerServe
python ./tools/gguf_export.py -m <hf-llama3.1-model> --qnn-path tools/qnn_converter/llama3.1-8b-QNN -o ./llama3.1-8b-instruct-model
```

## Compile PowerServe

The options of platform and ABI vary when deploying on different devices. DO CARE about the configuration.

### Build for Linux cpu
```shell
# Under the root directory of PowerServe
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Build for Android cpu
```shell
# Under the root directory of PowerServe
cmake -B build                                                      \
    -DCMAKE_BUILD_TYPE=Release                                      \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a                                         \
    -DANDROID_PLATFORM=android-35                                   \
    -DGGML_OPENMP=OFF                                               \
    -DPOWERSERVE_WITH_QNN=OFF

cmake --build build
```

### Build for Android qnn
- ❗️ Because the llama3.1-8b model is too large, qnn needs to open multiple sessions when loading. We conducted tests on 4 mobile phones. Among them, one plus 12, one plus 13 and Xiaomi 14 need to be updated to android 15 to apply for additional sessions in non-root mode, while honor Magic6 updates to android 15 to run in non-root mode will cause an error.

```shell
# Under the root directory of PowerServe
cmake -B build                                                      \
    -DCMAKE_BUILD_TYPE=Release                                      \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a                                         \
    -DANDROID_PLATFORM=android-35                                   \
    -DGGML_OPENMP=OFF                                               \
    -DPOWERSERVE_WITH_QNN=ON                                        \
    -DPOWERSERVE_ENABLE_HTPRPCPOLL=ON                               \
    -DPOWERSERVE_ENABLE_HMXPWRCFG=ON                                \
    -DPOWERSERVE_USE_DUMMY=ON

cmake --build build
```


## Prepare PowerServe Workspace

```shell
# Under the root directory of PowerServe
mkdir -p models

# Generate PowerServe Workspace
./powerserve create -m ./llama3.1-8b-instruct-model --exe-path ./build/out -o ./models/llama3.1-8b-instruct
```

## Execution

### CLI
More details please refer to [CLI App](./app/run/README.md)

For pure CPU execution
```shell
# Under the root directory of PowerServe
./models/llama3.1-8b-instruct/bin/powerserve-run --work-folder ./models/llama3.1-8b-instruct --prompt "Once upon a time, there was a little girl named Lucy" --no-qnn
```
For NPU execution
```shell
# Under the root directory of PowerServe
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 && ./models/llama3.1-8b-instruct/bin/powerserve-run --work-folder ./models/llama3.1-8b-instruct --prompt "Once upon a time, there was a little girl named Lucy"
```

### Server
More details please refer to [Server App](./app/server/README.md)
```shell
# Under the root directory of PowerServe
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 && ./models/llama3.1-8b-instruct/bin/powerserve-server --work-folder ./models --host <ip-addr> --port <port>
```

## Known Issues

### Execution

1. **When inferencing with QNN**: Failed to open lib /vendor/lib64/libcdsprpc.so: dlopen failed: library "/vendor/lib64/libcdsprpc.so" needed or dlopened by "/data/data/com.termux/files/home/workspace/qnn/llama-3.2-1b-instruct/bin/powerserve-run" is not accessible for the namespace "(default)

    > Use `export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64` before executing the program.
    >
    > Because `libcdsprpc.so` depends on `/system/lib64/libbinder.so` instead of `/vendor/lib64/libbinder.so`. If the linker searches the `/vendor/lib64` at first, it may find and links `/vendor/lib64/libbinder.so` which does not contain corresponding function definitions.

2. **Some mobile phones cannot run large models**: Some mobile phones cannot run larger models due to different security policies.

    **Some of known models and phones are listed below:**

    | Phone    | Models can't be run |
    |----------|---------------------|
    | All smartphones of HONOR | LLMs larger than 3B |

## Release Notes

[2025/4/22] Bug fixes, more debug tools, backward compatability and more QOL updates.

### Bug fixes

- Fixed a bug that caused the GGUF model loading failure due to the 32-bit system `size_t` overflow.
- Specify protobuf and pytorch versions in `./tools/qnn_converter/requirements.txt` for support of large models.
- QNN converter's `--n-threads` flag (defaults to 4) now can also control the maximum thread number of all converting phases
- QNN converter's library build logs now can be correctly generated across the build directories

### Debug tools

- Tensor dump: Added a debug tool to dump the tensor data in the model. This is useful for debugging and verifying the correctness of the model. To enable the tool, compile the executable with `-DPOWERSERVE_DUMP_TENSORS=ON`. The tensor data is dumped in text format, the format of data is configurable in the source code.

### Backward compatibility

- Added backward compatibility for the SA8295 (Hexagon V68). To run the framework on SA8295, compile the executable with `-DPOWERSERVE_ENABLE_HTPRPCPOLL=OFF` and `-DPOWERSERVE_ENABLE_HMXPWRCFG=OFF`. This is required because the SA8295 does not support these performance configurations.

## QOL updates

- Added several prompt files with instruction templates integrated with the prompt
- Added `--embd-only` commandline flag to the `convert_hf_to_gguf.py` script to extract only the embedding part of the model, to minimize the disk and memory usage in NPU-only inference scenarios
- Added `./tools/convert_hf_to_gguf/llama-quantize-x86_64-clang`, an instance of `llama-quantize` executable used for quantizing f32 gguf models into q4 but skips the model completeness check in the original llama.cpp. To quantize a embedding-only gguf model, you need to use this special executable (or skip the sanity check by yourself in llama.cpp).
- Added end-to-end parameter search from host feature, which is suitable for devices without python environment. All parameter search tools now require the framework executable built with `-DPOWERSERVE_DUMP_SPEEDINFO=ON` to enable the executable to dump speculative tree info / speed info to the file specified in environment variable `dump_file`.
- Added `--silent` flag to hide the commandline arguments for the qualcomm converter tools
- Added `--clear-build-files` flag to automatically clear the intermediate build files after build (and also slightly reduces the peak disk usage)
- `--soc` flag now receives the SoC number (for example, 8650 for Snapdragon 8 Gen 3, 8750 for Snapdragon 8 Elite, 8295 for Automotive SA8295) instead of abbreviation of the SoC to avoid confusion
- Added spinquant models' export parameters
- Specified hardware requirements to convert qnn models
- Added `POWERSERVE_USE_DUMMY` flag to let users decide whether to use dummy buffers to help allocate NPU buffers, should be enabled on SM8650/8750 and disabled on SA8295.
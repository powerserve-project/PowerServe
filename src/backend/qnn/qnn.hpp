// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "HTP/QnnHtpContext.h"
#include "HTP/QnnHtpDevice.h"
#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "core/exception.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"
#include "storage/file_loader.hpp"
#include "uv.h"

#include <memory>
#include <string>
#include <vector>

namespace powerserve::qnn {

using QNNDataType = Qnn_DataType_t;

#define POWERSERVE_ASSERT_QNN_ENV(expr, ...) POWERSERVE_ASSERT_ENV(expr, "QNN", __VA_ARGS__)

#define POWERSERVE_ASSERT_QNN_LOADER(expr, ...) POWERSERVE_ASSERT_MODULE(expr, "QNN", "Loader" __VA_ARGS__)

#define POWERSERVE_ASSERT_QNN_GRAPH(expr, ...) POWERSERVE_ASSERT_MODULE(expr, "QNN", "Graph" __VA_ARGS__)

static constexpr size_t type_size(QNNDataType type) {
    switch (type) {
    case QNN_DATATYPE_FLOAT_32:
        return 4;

    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_FLOAT_16:
        return 2;

    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
        return 1;

    default:
        POWERSERVE_ABORT("unknown QNN data type: {}", static_cast<int>(type));
    }
}

struct Library {
    static constexpr auto m_log_level = QNN_LOG_LEVEL_ERROR;

    uv_lib_t m_lib_backend = {};
    uv_lib_t m_lib_system  = {};
    uv_lib_t m_lib_rpc     = {};

    QNN_INTERFACE_VER_TYPE m_qnn_backend       = {};
    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnn_system = {};

    struct {
        typedef void *(*RpcMemAllocFn_t)(int heapid, uint32_t flags, int size);
        typedef void (*RpcMemFreeFn_t)(void *ptr);
        typedef int (*RpcMemToFdFn_t)(void *ptr);

        enum rpc_heap_ids {
            /**
             * Memory for secure use cases only.
             * - Secure heap is to be used only by clients migrating to CPZ
             */
            RPCMEM_HEAP_ID_SECURE = 9,

            /**
             * Contiguous physical memory:
             * - Very limited memory is available (< 8 MB)
             * - Recommended for subsystems without SMMU (sDSP and mDSP)
             * - Contiguous heap memory will be deprecated from archs after v73
             */
            RPCMEM_HEAP_ID_CONTIG = 22,

            /**
             * Non-contiguous system physical memory.
             * - Recommended for all use cases that do not require using a specific heap
             * - Used with subsystems with SMMU (cDSP and aDSP)
             */
            RPCMEM_HEAP_ID_SYSTEM = 25,
        };

        uint32_t RPCMEM_DEFAULT_FLAGS = 1;

        RpcMemAllocFn_t rpcmem_alloc = nullptr;
        RpcMemFreeFn_t rpcmem_free   = nullptr;
        RpcMemToFdFn_t rpcmem_to_fd  = nullptr;
    } m_rpc;

    Qnn_LogHandle_t m_logger = nullptr;

    Library() = default;
    ~Library();

    void initialize(
        const Path &lib_backend_path,
        const Path &lib_system_path,
        const Path &lib_rpc_path = "/vendor/lib64/libcdsprpc.so"
    );
    void open_qnn_backend_library(const Path &lib_backend_path);
    void open_qnn_system_library(const Path &lib_system_path);
    void open_rpc_library(const Path &lib_rpc_path);
    void print_info();
    void create_logger();
    void destroy_logger();
};

extern Library lib;

struct Backend {
    Qnn_BackendHandle_t m_handle = nullptr;
    Qnn_DeviceHandle_t m_device  = nullptr;

    Backend();
    ~Backend();

    void print_info();
};

struct HTPDevice {
    uint32_t m_device_id       = 0;
    uint32_t m_core_id         = 0;
    uint32_t m_power_config_id = 0;

    QnnDevice_Infrastructure_t m_infra             = {};
    QnnHtpDevice_Infrastructure_t *m_htp_infra     = nullptr;
    QnnHtpDevice_PerfInfrastructure_t m_perf_infra = {};

    HTPDevice(uint32_t device_id = 0, uint32_t core_id = 0);
    ~HTPDevice();

    void set_memory_grow_size(size_t size = 1ul * 1024 * 1024);
    void enter_performance_mode();
};

struct ContextGroup {
    size_t m_buffer_size               = 0;
    Qnn_ContextHandle_t m_first_handle = nullptr;

    ContextGroup(size_t buffer_size);

    auto get_config() const -> QnnHtpContext_CustomConfig_t;
    void add_context_handle(Qnn_ContextHandle_t handle);
};

struct Context {
    static constexpr auto DEFAULT_FILE_LOAD_METHOD = storage::FileLoaderMethod::MMap;

    std::string m_binary_filename;
    std::unique_ptr<storage::FileLoader> m_binary_loader;

    Qnn_ContextHandle_t m_handle                       = nullptr;
    QnnSystemContext_Handle_t m_system_context         = nullptr;
    const QnnSystemContext_BinaryInfo_t *m_binary_info = nullptr;

    Context(Backend &backend, const Path &binary_file_path, ContextGroup *group = nullptr);
    ~Context();

    void print_info();
    void free_system_context();
};

struct SharedBufferAllocator {
    void *m_data    = nullptr;
    size_t m_size   = 0;
    size_t m_offset = 0;
    int m_fd        = -1;

    SharedBufferAllocator(size_t _size);
    ~SharedBufferAllocator();

    auto unallocated_size() const -> size_t;
};

struct SharedBuffer {
    Qnn_MemHandle_t m_handle = nullptr;
    void *m_data             = nullptr;
    QNNDataType m_type       = {};
    size_t m_size            = 0;

    SharedBuffer(Context &context, SharedBufferAllocator &allocator, QNNDataType type, size_t n_elements);
    ~SharedBuffer();

    void memset(int byte);
};

struct QNNTensor {
    Qnn_Tensor_t m_tensor;

    QNNTensor(const Qnn_Tensor_t &source);
    ~QNNTensor();

    auto name() const -> std::string;
    size_t n_elements() const;
    auto type() const -> QNNDataType;
    size_t size() const;
    auto shape() const -> std::vector<size_t>;
    void setup_normal_buffer();
    void setup_shared_buffer(SharedBuffer &buffer);
    auto data() -> void *;
    int quantization_offset() const;
    float quantization_scale() const;
    auto check(const std::vector<size_t> &shape, Qnn_DataType_t datatype) -> QNNTensor *;
    void print();
#ifdef POWERSERVE_DUMP_TENSORS
    void dump(std::vector<size_t> &n_dump_elems);
#endif //POWERSERVE_DUMP_TENSORS
};

struct Graph {
    Qnn_GraphHandle_t m_handle = nullptr;

    std::string m_name;
    std::vector<QNNTensor> m_inputs;
    std::vector<QNNTensor> m_outputs;

    Graph(Context &context, const std::string &name);

    auto get_tensor(const std::string &name, bool required = true) -> QNNTensor *;
    bool has_tensor(const std::string &name);
    void set_n_hvx_threads(size_t n_threads);
    void execute();
};

struct Session {
    std::unique_ptr<Backend> m_backend;
    std::unique_ptr<HTPDevice> m_htp_device;
    std::unique_ptr<ContextGroup> m_group;
    size_t m_count = 0;
    Session(const Path &libs_folder);
    ~Session() = default;
};

struct ContextBinary {
    std::unique_ptr<Context> m_context;
    std::unique_ptr<SharedBufferAllocator> m_alloc;

    ContextBinary(Backend &backend, const Path &path);
};

} // namespace powerserve::qnn

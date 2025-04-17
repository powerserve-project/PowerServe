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

#include "qnn.hpp"

#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpMem.h"
#include "HTP/QnnHtpSystemContext.h"
#include "QnnContext.h"
#include "core/defines.hpp"
#include "core/logger.hpp"
#include "core/typedefs.hpp"
#include "qnn_type_macros.hpp"
#include "storage/file_loader.hpp"

#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <linux/mman.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <numeric>

#ifdef POWERSERVE_ANDROID_LOG
#include <android/log.h>
#endif // POWERSERVE_ANDROID_LOG

namespace powerserve::qnn {

static void log_callback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    POWERSERVE_UNUSED(level);
    POWERSERVE_UNUSED(timestamp);

    [[maybe_unused]]
    const int length = vprintf(fmt, args);

#ifdef POWERSERVE_ANDROID_LOG
    std::string message(length, '\0');
    vsnprintf(message.data(), length, fmt, args);
    __android_log_write(ANDROID_LOG_INFO, "PowerServe", message.c_str());
#endif // POWERSERVE_ANDROID_LOG
}

static auto format_qnn_version(Qnn_Version_t version) -> std::string {
    return fmt::format("{}.{}.{}", version.major, version.minor, version.patch);
}

Library::~Library() {
    destroy_logger();
    uv_dlclose(&m_lib_system);
    uv_dlclose(&m_lib_backend);
    uv_dlclose(&m_lib_rpc);
}

void Library::initialize(const Path &lib_backend_path, const Path &lib_system_path, const Path &lib_rpc_path) {
    open_qnn_backend_library(lib_backend_path);
    open_qnn_system_library(lib_system_path);
    open_rpc_library(lib_rpc_path);
    create_logger();
}

void Library::open_qnn_backend_library(const Path &lib_backend_path) {
    int ret = uv_dlopen(lib_backend_path.c_str(), &m_lib_backend);
    POWERSERVE_ASSERT_QNN_ENV(ret == 0, "failed to open lib {}: {}", lib_backend_path, m_lib_backend.errmsg);

    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
        const QnnInterface_t ***providerList, uint32_t *numProviders
    );

    QnnInterfaceGetProvidersFn_t get_interface_providers;
    const char *interface_providers_name = "QnnInterface_getProviders";
    ret = uv_dlsym(&m_lib_backend, interface_providers_name, (void **)&get_interface_providers);
    POWERSERVE_ASSERT_QNN_ENV(
        ret == 0,
        "failed to get symbol {} from lib {}: {}",
        interface_providers_name,
        lib_backend_path,
        m_lib_backend.errmsg
    );

    const QnnInterface_t **interface_providers;
    uint32_t n_providers = 0;
    ret                  = get_interface_providers(&interface_providers, &n_providers);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get interface providers");
    POWERSERVE_ASSERT_QNN_ENV(n_providers > 0, "no interface provider was found");

    bool found = false;
    for (size_t i = 0; i < n_providers; i++) {
        auto api_version = interface_providers[i]->apiVersion.coreApiVersion;
        if (QNN_API_VERSION_MAJOR == api_version.major && QNN_API_VERSION_MINOR <= api_version.minor) {
            found         = true;
            m_qnn_backend = interface_providers[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
    POWERSERVE_ASSERT_QNN_ENV(found);

    Qnn_ApiVersion_t api_version;
    ret = m_qnn_backend.backendGetApiVersion(&api_version);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get the version of QNN API");
    POWERSERVE_LOG_INFO("QNN core API version: {}", format_qnn_version(api_version.coreApiVersion));
    POWERSERVE_LOG_INFO("QNN backend API version: {}", format_qnn_version(api_version.backendApiVersion));
}

void Library::open_qnn_system_library(const Path &lib_system_path) {
    int ret = uv_dlopen(lib_system_path.c_str(), &m_lib_system);
    POWERSERVE_ASSERT_QNN_ENV(ret == 0, "failed to open lib {}: {}", lib_system_path, m_lib_system.errmsg);

    typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
        const QnnSystemInterface_t ***providerList, uint32_t *numProviders
    );

    QnnSystemInterfaceGetProvidersFn_t get_system_inferface_providers;
    const char *sysmte_interface_providers = "QnnSystemInterface_getProviders";
    ret = uv_dlsym(&m_lib_system, sysmte_interface_providers, (void **)&get_system_inferface_providers);
    POWERSERVE_ASSERT_QNN_ENV(
        ret == 0,
        "failed to get symbol {} from lib {}: {}",
        sysmte_interface_providers,
        lib_system_path,
        m_lib_system.errmsg
    );

    const QnnSystemInterface_t **system_interface_providers;
    uint32_t n_providers = 0;
    ret                  = get_system_inferface_providers(&system_interface_providers, &n_providers);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get system interface providers");
    POWERSERVE_ASSERT_QNN_ENV(n_providers > 0, "no system interface provider was found");

    bool found = false;
    for (size_t i = 0; i < n_providers; i++) {
        auto api_version = system_interface_providers[i]->systemApiVersion;
        if (QNN_SYSTEM_API_VERSION_MAJOR == api_version.major && QNN_SYSTEM_API_VERSION_MINOR <= api_version.minor) {
            found = true;
            POWERSERVE_LOG_INFO("QNN system API version: {}", format_qnn_version(api_version));
            m_qnn_system = system_interface_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    POWERSERVE_ASSERT_QNN_ENV(found, "no system API providers was found");
}

void Library::open_rpc_library(const Path &lib_rpc_path) {
    int ret = uv_dlopen(lib_rpc_path.c_str(), &m_lib_rpc);
    POWERSERVE_ASSERT_QNN_ENV(ret == 0, "failed to open lib {}: {}", lib_rpc_path, m_lib_rpc.errmsg);

    const char *rpc_mem_alloc_name = "rpcmem_alloc";
    ret                            = uv_dlsym(&m_lib_rpc, rpc_mem_alloc_name, (void **)&m_rpc.rpcmem_alloc);
    POWERSERVE_ASSERT_QNN_ENV(
        ret == 0, "failed to get symbol {} from lib {}: {}", rpc_mem_alloc_name, lib_rpc_path, m_lib_rpc.errmsg
    );

    const char *rpc_mem_free_name = "rpcmem_free";
    ret                           = uv_dlsym(&m_lib_rpc, rpc_mem_free_name, (void **)&m_rpc.rpcmem_free);
    POWERSERVE_ASSERT_QNN_ENV(
        ret == 0, "failed to get symbol {} from lib {}: {}", rpc_mem_free_name, lib_rpc_path, m_lib_rpc.errmsg
    );

    const char *rpc_mem_to_fd_name = "rpcmem_to_fd";
    ret                            = uv_dlsym(&m_lib_rpc, rpc_mem_to_fd_name, (void **)&m_rpc.rpcmem_to_fd);
    POWERSERVE_ASSERT_QNN_ENV(
        ret == 0, "failed to get symbol {} from lib {}: {}", rpc_mem_to_fd_name, lib_rpc_path, m_lib_rpc.errmsg
    );
}

void Library::print_info() {
    auto print_property = [&](const std::string &name, QnnProperty_Key_t property) {
        auto ret = m_qnn_backend.propertyHasCapability(property);

        const char *status = "Unknown";
        if (ret == QNN_PROPERTY_SUPPORTED) {
            status = "Yes";
        } else if (ret == QNN_PROPERTY_NOT_SUPPORTED) {
            status = "No";
        }

        POWERSERVE_LOG_INFO("- {}: {}", name, status);
    };

    POWERSERVE_LOG_INFO("QNN backend properties:");
    print_property("Create context from binary list", QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC);
    print_property("Dynamic batch", QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE);
    print_property("Early termination", QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION);
    print_property("Dynamic dimensions", QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS);
    print_property("Blockwise quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCK);
    print_property(
        "Blockwise quantization with expansion", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
    );
    print_property("Vector quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_VECTOR);
    print_property("Tensor sparsity", QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY);
    print_property("Updateable application tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_APP_TENSORS);
    print_property("Updateable native tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_NATIVE_TENSORS);
    print_property("Updateable static tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_STATIC_TENSORS);
}

void Library::create_logger() {
    auto ret = m_qnn_backend.logCreate(log_callback, m_log_level, &m_logger);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to create QNN logger");
}

void Library::destroy_logger() {
    if (m_logger) {
        auto ret = m_qnn_backend.logFree(m_logger);
        POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to free QNN logger");
        m_logger = nullptr;
    }
}

Library lib;

Backend::Backend() {
    auto ret = lib.m_qnn_backend.backendCreate(lib.m_logger, nullptr, &m_handle);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to create QNN backend");

    ret = lib.m_qnn_backend.deviceCreate(lib.m_logger, nullptr, &m_device);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to create QNN device");
}

Backend::~Backend() {
    auto ret = lib.m_qnn_backend.deviceFree(m_device);
    POWERSERVE_ASSERT(ret == QNN_SUCCESS, "failed to free QNN device");

    ret = lib.m_qnn_backend.backendFree(m_handle);
    POWERSERVE_ASSERT(ret == QNN_SUCCESS, "failed to free QNN backend");
}

void Backend::print_info() {
    const QnnDevice_PlatformInfo_t *platform_info_ptr;
    auto ret = lib.m_qnn_backend.deviceGetInfo(m_device, &platform_info_ptr);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get the QNN device info");
    POWERSERVE_ASSERT_QNN_ENV(
        platform_info_ptr->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1,
        "unknown platform info version: {}",
        static_cast<int>(platform_info_ptr->version)
    );

    auto &platform_info = platform_info_ptr->v1;

    POWERSERVE_LOG_INFO("Hardware device information:");
    for (size_t i = 0; i < platform_info.numHwDevices; i++) {
        auto &hw_info_struct = platform_info.hwDevices[i];
        POWERSERVE_ASSERT_QNN_ENV(hw_info_struct.version == QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1);

        auto &hw_info = hw_info_struct.v1;
        POWERSERVE_LOG_INFO(
            "[{}] id={}, type={}, num_cores={}, ext_type={}",
            i,
            hw_info.deviceId,
            hw_info.deviceType,
            hw_info.numCores,
            (int)hw_info.deviceInfoExtension->devType
        );

        for (size_t j = 0; j < hw_info.numCores; j++) {
            auto &core_info_struct = hw_info.cores[j];
            POWERSERVE_ASSERT_QNN_ENV(core_info_struct.version == QNN_DEVICE_CORE_INFO_VERSION_1);

            auto &core_info = core_info_struct.v1;
            POWERSERVE_LOG_INFO("[{}] core[{}]: id={}, type={}", i, j, core_info.coreId, core_info.coreType);
        }

        if (hw_info.deviceInfoExtension->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            auto &on_chip_info = hw_info.deviceInfoExtension->onChipDevice;
            POWERSERVE_LOG_INFO(
                "[{}] on_chip: soc={}, arch={}, dlbc={}, signed_pd={}, vtcm_size={}",
                i,
                on_chip_info.socModel,
                (int)on_chip_info.arch,
                on_chip_info.dlbcSupport,
                on_chip_info.signedPdSupport,
                on_chip_info.vtcmSize
            );
        }
    }

    ret = lib.m_qnn_backend.deviceFreePlatformInfo(lib.m_logger, platform_info_ptr);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to free QNN platform info");
}

HTPDevice::HTPDevice(uint32_t device_id, uint32_t core_id) : m_device_id(device_id), m_core_id(core_id) {
    auto ret = lib.m_qnn_backend.deviceGetInfrastructure(&m_infra);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get HTP device infrastructure");

    m_htp_infra  = (QnnHtpDevice_Infrastructure_t *)m_infra;
    m_perf_infra = m_htp_infra->perfInfra;
    ret          = m_perf_infra.createPowerConfigId(device_id, core_id, &m_power_config_id);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get power config id of HTP");
}

HTPDevice::~HTPDevice() {
    auto ret = m_perf_infra.destroyPowerConfigId(m_power_config_id);
    POWERSERVE_ASSERT(ret == QNN_SUCCESS, "failed to destroy powerserve config id");
}

void HTPDevice::set_memory_grow_size(size_t size) {
    QnnHtpPerfInfrastructure_MemoryConfig_t grow_size_config = {
        .option            = QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE,
        .memGrowSizeConfig = (uint32_t)size,
    };

    const QnnHtpPerfInfrastructure_MemoryConfig_t *memory_config[] = {
        &grow_size_config,
        nullptr,
    };
    auto ret = m_perf_infra.setMemoryConfig(m_device_id, m_core_id, memory_config);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to set HTP memory config");
}

void HTPDevice::enter_performance_mode() {
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs_v3_config = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config =
            {
                .contextId = m_power_config_id,

                .setDcvsEnable = 1,
                .dcvsEnable    = 0,

                .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,

                .setSleepLatency = 1,
                .sleepLatency    = 40,

                .setSleepDisable = 1,
                .sleepDisable    = 1,

                .setBusParams           = 1,
                .busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,

                .setCoreParams           = 1,
                .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            },
    };

#ifdef POWERSERVE_ENABLE_HMXPWRCFG
    QnnHtpPerfInfrastructure_PowerConfig_t hmx_config = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2,
        .hmxV2Config =
            {
                .hmxPickDefault         = 0,
                .hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX,
                .hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX,
                .hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX,
                .hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH,
            },
    };
#endif // POWERSERVE_ENABLE_HMXPWRCFG

#ifdef POWERSERVE_ENABLE_HTPRPCPOLL
    QnnHtpPerfInfrastructure_PowerConfig_t rpc_poll_config = {
        .option               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
        .rpcPollingTimeConfig = 9999,
    };
#endif //POWERSERVE_ENABLE_HTPRPCPOLL

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_ctrl_config = {
        .option                  = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY,
        .rpcControlLatencyConfig = 100,
    };

    const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = {
        &dcvs_v3_config,
        &rpc_ctrl_config,
#ifdef POWERSERVE_ENABLE_HMXPWRCFG
        &hmx_config,
#endif // POWERSERVE_ENABLE_HMXPWRCFG
#ifdef POWERSERVE_ENABLE_HTPRPCPOLL
        &rpc_poll_config,
#endif // POWERSERVE_ENABLE_HTPRPCPOLL
        nullptr,
    };
    auto ret = m_perf_infra.setPowerConfig(m_power_config_id, power_configs);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to set HTP power config");
}

ContextGroup::ContextGroup(size_t buffer_size) : m_buffer_size(buffer_size) {}

auto ContextGroup::get_config() const -> QnnHtpContext_CustomConfig_t {
    return QnnHtpContext_CustomConfig_t{
        .option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS,
        .groupRegistration =
            {
                .firstGroupHandle   = m_first_handle,
                .maxSpillFillBuffer = m_buffer_size,
            },
    };
}

void ContextGroup::add_context_handle(Qnn_ContextHandle_t handle) {
    if (!m_first_handle) {
        m_first_handle = handle;
    }
}

Context::Context(Backend &backend, const Path &binary_file_path, ContextGroup *group) :
    m_binary_filename(binary_file_path) {
    POWERSERVE_ASSERT_QNN_ENV(
        std::filesystem::exists(binary_file_path) && std::filesystem::is_regular_file(binary_file_path),
        "could not found regular binary file: {}",
        binary_file_path
    );

    /*
     * mmap binary
     */
    m_binary_loader    = storage::build_file_loader(binary_file_path, DEFAULT_FILE_LOAD_METHOD);
    auto binary_buffer = m_binary_loader->get_buffer();
    POWERSERVE_LOG_INFO(
        "load binary file {} (size: {}) into address space [{}, {})",
        binary_file_path,
        binary_buffer.size(),
        static_cast<void *>(binary_buffer.data()),
        static_cast<void *>(binary_buffer.end().base())
    );

    std::vector<const QnnContext_Config_t *> context_configs;

    QnnHtpContext_CustomConfig_t htp_group_config;
    QnnContext_Config_t group_config = {
        .option       = QNN_CONTEXT_CONFIG_PERSISTENT_BINARY,
        .customConfig = &htp_group_config,
    };

    if (group) {
        htp_group_config = group->get_config();
        context_configs.push_back(&group_config);
    }

    QnnHtpContext_CustomConfig_t htp_io_estimation_config = {
        .option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION,
        .ioMemEstimation = true,
    };
    QnnContext_Config_t io_estimation_config = {
        .option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_io_estimation_config,
    };
    context_configs.push_back(&io_estimation_config);

    context_configs.push_back(nullptr);

    auto ret = lib.m_qnn_backend.contextCreateFromBinary(
        backend.m_handle,
        backend.m_device,
        context_configs.data(),
        binary_buffer.data(),
        binary_buffer.size(),
        &m_handle,
        nullptr
    );
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to create QNN context from binary: {}", binary_file_path);

    if (group) {
        group->add_context_handle(m_handle);
    }

    ret = lib.m_qnn_system.systemContextCreate(&m_system_context);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to create QNN context");

    Qnn_ContextBinarySize_t binary_info_size = 0;
    ret                                      = lib.m_qnn_system.systemContextGetBinaryInfo(
        m_system_context, binary_buffer.data(), binary_buffer.size(), &m_binary_info, &binary_info_size
    );
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to get info of QNN context binary");

    /* Unload binary caches as possible */
    m_binary_loader->unload();
}

Context::~Context() {
    free_system_context();
    {
        const auto ret = lib.m_qnn_backend.contextFree(m_handle, nullptr);
        POWERSERVE_ASSERT(ret == QNN_SUCCESS, "failed to free QNN context");
    }
}

void Context::print_info() {
    POWERSERVE_ASSERT_QNN_ENV(m_binary_info);
    switch (m_binary_info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
        auto &info = m_binary_info->contextBinaryInfoV1;

        auto hw_blob_info_ptr = (QnnHtpSystemContext_HwBlobInfo_t *)info.hwInfoBlob;
        POWERSERVE_ASSERT_QNN_ENV(hw_blob_info_ptr->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1);
        auto &hw_blob_info = hw_blob_info_ptr->contextBinaryHwInfoBlobV1_t;

        POWERSERVE_LOG_INFO("Context core API version: {}", format_qnn_version(info.coreApiVersion));
        POWERSERVE_LOG_INFO("Context backend API version: {}", format_qnn_version(info.backendApiVersion));
        POWERSERVE_LOG_INFO("Context blob version: {}", format_qnn_version(info.contextBlobVersion));
        POWERSERVE_LOG_INFO("Number of graphs: {}", info.numGraphs);
        POWERSERVE_LOG_INFO("Spill-fill buffer size: {:.3f} MiB", hw_blob_info.spillFillBufferSize / 1024.0 / 1024);

    } break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
        auto &info = m_binary_info->contextBinaryInfoV2;

        auto hw_blob_info_ptr = (QnnHtpSystemContext_HwBlobInfo_t *)info.hwInfoBlob;
        POWERSERVE_ASSERT_QNN_ENV(hw_blob_info_ptr->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1);
        auto &hw_blob_info = hw_blob_info_ptr->contextBinaryHwInfoBlobV1_t;

        POWERSERVE_LOG_INFO("Context core API version: {}", format_qnn_version(info.coreApiVersion));
        POWERSERVE_LOG_INFO("Context backend API version: {}", format_qnn_version(info.backendApiVersion));
        POWERSERVE_LOG_INFO("Context blob version: {}", format_qnn_version(info.contextBlobVersion));
        POWERSERVE_LOG_INFO("Number of graphs: {}", info.numGraphs);
        POWERSERVE_LOG_INFO("Spill-fill buffer size: {:.3f} MiB", hw_blob_info.spillFillBufferSize / 1024.0 / 1024);
    } break;
#if (QNN_API_VERSION_MINOR >= 21)
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
        auto &info = m_binary_info->contextBinaryInfoV3;
        POWERSERVE_LOG_INFO("Context core API version: {}", format_qnn_version(info.coreApiVersion));
        POWERSERVE_LOG_INFO("Context backend API version: {}", format_qnn_version(info.backendApiVersion));
        POWERSERVE_LOG_INFO("Context blob version: {}", format_qnn_version(info.contextBlobVersion));
        POWERSERVE_LOG_INFO("Number of graphs: {}", info.numGraphs);
    } break;
#endif
    default:
        break;
    }
}

void Context::free_system_context() {
    if (m_system_context) {
        auto ret = lib.m_qnn_system.systemContextFree(m_system_context);
        POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to free QNN system context");
    }

    m_binary_info    = nullptr;
    m_system_context = nullptr;
}

SharedBufferAllocator::SharedBufferAllocator(size_t _size) : m_size(_size) {
    m_data = lib.m_rpc.rpcmem_alloc(lib.m_rpc.RPCMEM_HEAP_ID_SYSTEM, lib.m_rpc.RPCMEM_DEFAULT_FLAGS, m_size);
    POWERSERVE_ASSERT_QNN_ENV(m_data, "failed to allocate RPC memory");

    m_fd = lib.m_rpc.rpcmem_to_fd(m_data);
    POWERSERVE_ASSERT_QNN_ENV(m_fd != -1, "failed to convert RPC memory into file descriptor");
}

SharedBufferAllocator::~SharedBufferAllocator() {
    lib.m_rpc.rpcmem_free(m_data);
}

auto SharedBufferAllocator::unallocated_size() const -> size_t {
    return m_size - m_offset;
}

SharedBuffer::SharedBuffer(Context &context, SharedBufferAllocator &allocator, QNNDataType type, size_t n_elements) :
    m_type(type) {
    m_size = type_size(type) * n_elements;

    POWERSERVE_ASSERT_QNN_ENV(
        allocator.m_offset + m_size <= allocator.m_size,
        "no free memory in QNN shared buffer allocator (required: {}, avail: {}, total: {})",
        m_size,
        allocator.m_size - allocator.m_offset,
        allocator.m_size
    );
    m_data = (char *)allocator.m_data + allocator.m_offset;

    QnnMemHtp_Descriptor_t htp_mem_desc = {
        .type = QNN_HTP_MEM_SHARED_BUFFER,
        .size = allocator.m_size, // NOTE: It's the total size of the shared buffer allocator
        .sharedBufferConfig =
            (QnnHtpMem_SharedBufferConfig_t){
                .fd     = allocator.m_fd,
                .offset = allocator.m_offset,
            },
    };

    uint32_t shape[1]            = {(uint32_t)n_elements};
    Qnn_MemDescriptor_t mem_desc = {
        .memShape =
            {
                .numDim      = 1,
                .dimSize     = shape,
                .shapeConfig = nullptr,
            },
        .dataType   = type,
        .memType    = QNN_MEM_TYPE_CUSTOM,
        .customInfo = &htp_mem_desc,
    };

    auto ret = lib.m_qnn_backend.memRegister(context.m_handle, &mem_desc, 1, &m_handle);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to register QNN shared buffer");

    allocator.m_offset += m_size;
}

SharedBuffer::~SharedBuffer() {
    auto ret = lib.m_qnn_backend.memDeRegister(&m_handle, 1);
    POWERSERVE_ASSERT(ret == QNN_SUCCESS, "failed to unregister QNN shared buffer");
}

void SharedBuffer::memset(int byte) {
    std::memset(m_data, byte, m_size);
}

static void deep_copy_tensor(Qnn_Tensor_t &dst, const Qnn_Tensor_t &src) {
    dst = QNN_TENSOR_INIT;

    // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
    // to correctly assign values
    dst.version = src.version;

    const char *tensorName = QNN_TENSOR_GET_NAME(src);
    if (!tensorName) {
        QNN_TENSOR_SET_NAME(dst, nullptr);
    } else {
        QNN_TENSOR_SET_NAME(dst, strdup(tensorName));
    }

    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding         = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;

        if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
            qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
                QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t)
            );

            if (qParams.axisScaleOffsetEncoding.scaleOffset) {
                for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
                     idx++) {
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
                }
            }
        }
    }

    QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);

    if (QNN_TENSOR_GET_RANK(src) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));

        if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
            memcpy(
                QNN_TENSOR_GET_DIMENSIONS(dst),
                QNN_TENSOR_GET_DIMENSIONS(src),
                QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)
            );
        }

        // I think the original code is wrong...
        POWERSERVE_ASSERT_QNN_LOADER(!QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src));
    }

    QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
}

std::unordered_map<QNNTensor *, void *> buffer_map{};

QNNTensor::QNNTensor(const Qnn_Tensor_t &source) {
    deep_copy_tensor(m_tensor, source);
    POWERSERVE_ASSERT_QNN_LOADER(QNN_TENSOR_GET_MEM_TYPE(m_tensor) == QNN_TENSORMEMTYPE_UNDEFINED);
}

QNNTensor::~QNNTensor() {
    const auto tensor_mem_type = QNN_TENSOR_GET_MEM_TYPE(m_tensor);
    switch (tensor_mem_type) {
    case QNN_TENSORMEMTYPE_RAW: {
        free(QNN_TENSOR_GET_CLIENT_BUF(m_tensor).data);
    } break;

    case QNN_TENSORMEMTYPE_MEMHANDLE: // Released by SharedBufferAllocator
    case QNN_TENSORMEMTYPE_UNDEFINED:
        break;

    default:
        POWERSERVE_ABORT("unknown memory type of tensor: {}", static_cast<int>(tensor_mem_type));
    }
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_UNDEFINED);

    free(QNN_TENSOR_GET_DIMENSIONS(m_tensor));
    free((void *)QNN_TENSOR_GET_NAME(m_tensor));
}

auto QNNTensor::name() const -> std::string {
    return QNN_TENSOR_GET_NAME(m_tensor);
}

size_t QNNTensor::n_elements() const {
    size_t n_elements = 1;
    for (size_t i = 0; i < QNN_TENSOR_GET_RANK(m_tensor); i++) {
        n_elements *= QNN_TENSOR_GET_DIMENSIONS(m_tensor)[i];
    }
    return n_elements;
}

auto QNNTensor::type() const -> QNNDataType {
    return QNN_TENSOR_GET_DATA_TYPE(m_tensor);
}

size_t QNNTensor::size() const {
    return n_elements() * type_size(type());
}

auto QNNTensor::shape() const -> std::vector<size_t> {
    std::vector<size_t> shape(QNN_TENSOR_GET_RANK(m_tensor));
    for (size_t i = 0; i < shape.size(); i++) {
        shape[i] = QNN_TENSOR_GET_DIMENSIONS(m_tensor)[i];
    }
    return shape;
}

void QNNTensor::setup_normal_buffer() {
    Qnn_ClientBuffer_t buffer = QNN_CLIENT_BUFFER_INIT;
    buffer.dataSize           = size();
    buffer.data               = malloc(buffer.dataSize);
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(m_tensor, buffer);
}

void QNNTensor::setup_shared_buffer(SharedBuffer &buffer) {
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(m_tensor, buffer.m_handle);
    buffer_map.emplace(this, buffer.m_data);
}

auto QNNTensor::data() -> void * {
    const auto tensor_mem_type = QNN_TENSOR_GET_MEM_TYPE(m_tensor);
    switch (tensor_mem_type) {
    case QNN_TENSORMEMTYPE_RAW:
        return QNN_TENSOR_GET_CLIENT_BUF(m_tensor).data;

    case QNN_TENSORMEMTYPE_MEMHANDLE: // User should fill the shared buffer directly
        POWERSERVE_ABORT("user should fill the shared buffer directly when using QNN memhandle under tensor");
        break;

    default:
        POWERSERVE_ABORT("unknown memory type of tensor: {}", static_cast<int>(tensor_mem_type));
    }
}

int QNNTensor::quantization_offset() const {
    return QNN_TENSOR_GET_QUANT_PARAMS(m_tensor).scaleOffsetEncoding.offset;
}

float QNNTensor::quantization_scale() const {
    return QNN_TENSOR_GET_QUANT_PARAMS(m_tensor).scaleOffsetEncoding.scale;
}

auto QNNTensor::check(const std::vector<size_t> &shape, Qnn_DataType_t datatype) -> QNNTensor * {
    POWERSERVE_ASSERT_QNN_LOADER(this->shape() == shape);
    POWERSERVE_ASSERT_QNN_LOADER(this->type() == datatype);
    return this;
}

void QNNTensor::print() {
    if (type() == QNN_DATATYPE_FLOAT_32) {
        auto buf = (const float *)buffer_map.at(this);
        for (size_t i = 0; i < n_elements(); i++) {
            fmt::println(stderr, "{}", buf[i]);
        }
    } else if (type() == QNN_DATATYPE_FLOAT_16) {
        auto buf = (const __fp16 *)buffer_map.at(this);
        for (size_t i = 0; i < n_elements(); i++) {
            fmt::println(stderr, "{}", (float)buf[i]);
        }
    }
}

#ifdef POWERSERVE_DUMP_TENSORS
void dump_dimensions_fp32(size_t dimension_index, const float* buffer, const std::vector<size_t> &shape, const std::vector<size_t> &stride, std::vector<size_t> &curr_index, const std::vector<size_t> &n_dump_elems) {
    if(dimension_index == shape.size() - 1) {
        // Recurse ends in the last dimension, print elements
        fmt::print("Dumping elements in dimension {}:", curr_index);
        size_t base_index = 0;
        for(size_t i = 0; i < curr_index.size(); i++) {
            base_index += stride[i] * curr_index[i];
        } 
        for(size_t i = 0; i < n_dump_elems[dimension_index] && i < shape[dimension_index]; i++) {
            fmt::print(" {:.6f}", buffer[base_index + i]);
        }
        fmt::println("");
    } else {
        for(size_t i = 0; i < n_dump_elems[dimension_index] && i < shape[dimension_index]; i++) {
            std::vector<size_t> new_index(curr_index);
            new_index.emplace_back(i);
            dump_dimensions_fp32(dimension_index + 1, buffer, shape, stride, new_index, n_dump_elems);
        }
    }
}

void dump_dimensions_fp16(size_t dimension_index, const __fp16* buffer, const std::vector<size_t> &shape, const std::vector<size_t> &stride, std::vector<size_t> &curr_index, const std::vector<size_t> &n_dump_elems) {
    if(dimension_index == shape.size() - 1) {
        // Recurse ends in the last dimension, print elements
        fmt::print("Dumping elements in dimension {}:", curr_index);
        size_t base_index = 0;
        for(size_t i = 0; i < curr_index.size(); i++) {
            base_index += stride[i] * curr_index[i];
        } 
        for(size_t i = 0; i < n_dump_elems[dimension_index] && i < shape[dimension_index]; i++) {
            fmt::print(" {:.6f}", (float)buffer[base_index + i]);
        }
        fmt::println("");
    } else {
        for(size_t i = 0; i < n_dump_elems[dimension_index] && i < shape[dimension_index]; i++) {
            std::vector<size_t> new_index(curr_index);
            new_index.emplace_back(i);
            dump_dimensions_fp16(dimension_index + 1, buffer, shape, stride, new_index, n_dump_elems);
        }
    }
}

void QNNTensor::dump(std::vector<size_t> &n_dump_elems) {
    
    fmt::println("--------------------Dumping QNN Tensor--------------------");
    fmt::println("Tensor name: {}", this->name());
    fmt::println("Tensor rank: {}", this->shape().size());
    POWERSERVE_ASSERT(n_dump_elems.size() == this->shape().size());
    fmt::println("Tensor shape: {}", this->shape());

    std::vector<size_t> shape = this->shape();
    std::vector<size_t> stride(shape.size(), 1);
    for(int i = shape.size() - 1; i > -1; i--) {
        for(int j = shape.size() - 1; j > i; j--) {
            stride[i] *= shape[j];
        }
    }

    std::vector<size_t> init_index = {};        
    if (type() == QNN_DATATYPE_FLOAT_32) {
        fmt::println("Tensor dtype: FP32");
        auto buf = (const float *)buffer_map.at(this);
        dump_dimensions_fp32(0, buf, shape, stride, init_index, n_dump_elems);
    } else if (type() == QNN_DATATYPE_FLOAT_16) {
        fmt::println("Tensor dtype: FP16");
        auto buf = (const __fp16 *)buffer_map.at(this);
        dump_dimensions_fp16(0, buf, shape, stride, init_index, n_dump_elems);
    }
}
#endif //POWERSERVE_DUMP_TENSORS

Graph::Graph(Context &context, const std::string &name) : m_name(name) {
    auto processGraphInfo = [&](auto &graph_info) {
        size_t n_inputs = graph_info.numGraphInputs;
        m_inputs.reserve(n_inputs);
        for (size_t i = 0; i < n_inputs; i++) {
            m_inputs.emplace_back(graph_info.graphInputs[i]);
        }

        size_t n_outputs = graph_info.numGraphOutputs;
        m_outputs.reserve(n_outputs);
        for (size_t i = 0; i < n_outputs; i++) {
            m_outputs.emplace_back(graph_info.graphOutputs[i]);
        }

        auto ret = lib.m_qnn_backend.graphRetrieve(context.m_handle, name.c_str(), &m_handle);
        POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS);
    };
    switch (context.m_binary_info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
        auto &info = context.m_binary_info->contextBinaryInfoV1;
        for (size_t i = 0; i < info.numGraphs; i++) {
            const auto *current_graph = &info.graphs[i];
            POWERSERVE_ASSERT_QNN_ENV(current_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1);
            if (current_graph->graphInfoV1.graphName == name) {
                processGraphInfo(current_graph->graphInfoV1);
                break;
            }
        }
    } break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
        auto &info = context.m_binary_info->contextBinaryInfoV2;
        for (size_t i = 0; i < info.numGraphs; i++) {
            const auto *current_graph = &info.graphs[i];
            POWERSERVE_ASSERT_QNN_ENV(current_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2);
            if (current_graph->graphInfoV2.graphName == name) {
                processGraphInfo(current_graph->graphInfoV2);
                break;
            }
        }
    } break;
#if (QNN_API_VERSION_MINOR >= 21)
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
        auto &info = context.m_binary_info->contextBinaryInfoV3;
        for (size_t i = 0; i < info.numGraphs; i++) {
            const auto *current_graph = &info.graphs[i];
            POWERSERVE_ASSERT_QNN_ENV(current_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3);
            if (current_graph->graphInfoV3.graphName == name) {
                processGraphInfo(current_graph->graphInfoV3);
                break;
            }
        }
    } break;
#endif
    default:
        POWERSERVE_ABORT("Unknown QNN binary version: {}", static_cast<int>(context.m_binary_info->version));
        break;
    }
}

auto Graph::get_tensor(const std::string &name, bool required) -> QNNTensor * {
    for (auto &t : m_inputs) {
        if (t.name() == name) {
            return &t;
        }
    }

    for (auto &t : m_outputs) {
        if (t.name() == name) {
            return &t;
        }
    }

    if (required) {
        POWERSERVE_ABORT("Cannot find tensor with name \"{}\"", name);
    } else {
        return nullptr;
    }
}

bool Graph::has_tensor(const std::string &name) {
    return get_tensor(name, false) != nullptr;
}

void Graph::set_n_hvx_threads(size_t n_threads) {
    QnnHtpGraph_CustomConfig_t htp_hvx_thread_config = {
        .option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS,
        .numHvxThreads = n_threads,
    };

    QnnGraph_Config_t hvx_thread_config = {
        .option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_hvx_thread_config,
    };

    const QnnGraph_Config_t *graph_configs[] = {&hvx_thread_config, nullptr};
    auto ret                                 = lib.m_qnn_backend.graphSetConfig(m_handle, graph_configs);
    POWERSERVE_ASSERT_QNN_ENV(ret == QNN_SUCCESS, "failed to sett QNN graph config");
}

void Graph::execute() {
    auto ret = lib.m_qnn_backend.graphExecute(
        m_handle,
        (const Qnn_Tensor_t *)m_inputs.data(),
        m_inputs.size(),
        (Qnn_Tensor_t *)m_outputs.data(),
        m_outputs.size(),
        nullptr,
        nullptr
    );
    POWERSERVE_ASSERT_QNN_GRAPH(ret == QNN_SUCCESS, "failed to execute QNN graph");
}

Session::Session(const Path &libs_folder) {
    m_count = 0;
    uv_os_setenv("ADSP_LIBRARY_PATH", libs_folder.c_str());
    lib.initialize(libs_folder / "libQnnHtp.so", libs_folder / "libQnnSystem.so");
    lib.print_info();

    m_backend = std::make_unique<qnn::Backend>();
    m_backend->print_info();

    m_htp_device = std::make_unique<HTPDevice>();
    m_htp_device->set_memory_grow_size();
    m_htp_device->enter_performance_mode();

    m_group = std::make_unique<ContextGroup>(10 * 1024 * 1024);
}

ContextBinary::ContextBinary(Backend &backend, const Path &path) {
    m_context = std::make_unique<Context>(backend, path, nullptr);
}

} // namespace powerserve::qnn

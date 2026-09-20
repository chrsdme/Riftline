// M5.4b sec 10: live device + software identity collectors. CUDA-only TU, deliberately separate
// from calibration_collect.cpp (host-only, decision 3) so the host test target never links CUDA.
// This TU only needs to COMPILE for this milestone -- it is not exercised by any host test and
// nothing in this task runs it. Field-query pattern (cudaGetDeviceProperties / cudaUUID_t
// formatting / cudaRuntimeGetVersion) reused from
// src/serve/request_log.cpp's query_server_log_environment -- see cuda_uuid_string/
// cuda_version_string there for the precedent this mirrors. No NVML dependency added.

#include "targets/calibration_collect.h"
#include "targets/calibration_collect_cuda.h"

#include <cuda_runtime.h>

#include <iomanip>
#include <sstream>

namespace ninfer::targets::calibration_collect {
namespace {

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

} // namespace

DeviceCollectionResult collect_device_identity(int logical_slot, int physical_index) {
    DeviceCollectionResult result;
    result.device.logical_slot = logical_slot;
    result.device.physical_index = physical_index;

    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, physical_index);
    if (status != cudaSuccess) {
        result.diagnostic = {"device[" + std::to_string(logical_slot) + "]",
                              std::string("cudaGetDeviceProperties failed: ") +
                                  cudaGetErrorString(status),
                              true};
        return result; // device stays at fail-closed defaults: uuid empty, cc 0/0, vram 0.
    }
    result.device.uuid = cuda_uuid_string(properties.uuid);
    result.device.model_name = properties.name;
    result.device.compute_capability_major = properties.major;
    result.device.compute_capability_minor = properties.minor;
    result.device.vram_bytes = static_cast<std::uint64_t>(properties.totalGlobalMem);
    return result;
}

SoftwareCollectionResult collect_software_identity() {
    SoftwareCollectionResult result;

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        result.software.cuda_runtime_version = cuda_version_string(runtime_version);
    } else {
        result.diagnostics.push_back({"software", "cudaRuntimeGetVersion failed", true});
    }

    result.software.driver_version = collect_nvidia_driver_release();
    if (result.software.driver_version.empty()) {
        result.diagnostics.push_back(
            {"software", "NVIDIA driver release unavailable", true});
    }

    // Toolkit/build version: CUDART_VERSION is a compile-time constant baked into this TU by the
    // CUDA headers used to build it -- reported honestly as the toolkit this binary was built
    // against, distinct from the runtime/driver versions actually present on the machine.
    result.software.cuda_toolkit_version = cuda_version_string(CUDART_VERSION);

    return result;
}

} // namespace ninfer::targets::calibration_collect

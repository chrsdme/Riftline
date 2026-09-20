#include "options.h"

#include "targets/calibration.h"
#include "targets/calibration_collect.h"
#include "targets/calibration_collect_cuda.h"
#include "targets/calibration_io.h"
#include "targets/calibration_measurement.h"
#include "targets/calibration_summary.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using ninfer::targets::calibration::CalibrationIdentity;
using ninfer::targets::calibration::DeviceIdentity;
using ninfer::targets::calibration::EvidenceClass;
using ninfer::targets::calibration::HostInvolvement;
using ninfer::targets::calibration::RecordKind;
using ninfer::targets::calibration::SourceState;
using ninfer::targets::calibration::TransferOverlap;
using ninfer::targets::calibration::TransferPayloadClass;
using ninfer::targets::calibration::TransferRoute;
using ninfer::targets::calibration::TransferSync;
using ninfer::targets::calibration::ValidityState;
using ninfer::targets::calibration::CorrectnessStatus;
using ninfer::targets::calibration::source_state_name;
using ninfer::targets::calibration::missing_identity_requirements;
using ninfer::targets::calibration_collect::CollectionDiagnostic;
using ninfer::targets::calibration_collect::DeviceCollectionResult;
using ninfer::targets::calibration_collect::SoftwareCollectionResult;
using ninfer::targets::calibration_collect::assemble_identity_snapshot;
using ninfer::targets::calibration_collect::collect_device_identity;
using ninfer::targets::calibration_collect::collect_software_identity;

namespace {

#define CALIBRATION_CUDA_CHECK(expr)                                                            \
    do {                                                                                         \
        const cudaError_t status__ = (expr);                                                      \
        if (status__ != cudaSuccess) {                                                           \
            throw std::runtime_error(std::string("CUDA failure at ") + __FILE__ + ":" +       \
                                     std::to_string(__LINE__) + ": " +                          \
                                     cudaGetErrorName(status__) + ": " +                         \
                                     cudaGetErrorString(status__));                               \
        }                                                                                        \
    } while (false)

struct IdentityGateResult {
    CalibrationIdentity identity;
    std::vector<CollectionDiagnostic> diagnostics;
    std::vector<std::string> missing;
    bool qualified = false;
};

std::uint64_t now_unix_ms() {
    using namespace std::chrono;
    return static_cast<std::uint64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

std::string reconstruct_command_line(int argc, char** argv) {
    std::ostringstream out;
    for (int i = 0; i < argc; ++i) {
        if (i != 0) { out << ' '; }
        out << argv[i];
    }
    return out.str();
}

void print_diagnostics(const std::vector<CollectionDiagnostic>& diagnostics) {
    for (const CollectionDiagnostic& d : diagnostics) {
        std::cout << "IDENTITY_DIAGNOSTIC component=" << d.component
                   << " failed=" << (d.failed ? "true" : "false") << " detail=\"" << d.detail
                   << "\"\n";
    }
}

void print_device(const DeviceIdentity& device) {
    std::cout << "IDENTITY_DEVICE logical_slot=" << device.logical_slot
               << " physical_index=" << device.physical_index << " uuid=" << device.uuid
               << " model_name=\"" << device.model_name << "\""
               << " cc=" << device.compute_capability_major << '.'
               << device.compute_capability_minor << " vram_bytes=" << device.vram_bytes << '\n';
}

IdentityGateResult collect_identity(const ninfer::calibration_probe::Options& options,
                                    const std::string& command_line) {
    std::vector<DeviceIdentity> devices;
    std::vector<CollectionDiagnostic> device_diagnostics;

    const DeviceCollectionResult slot0 = collect_device_identity(0, options.slot0_cuda_index);
    devices.push_back(slot0.device);
    if (slot0.diagnostic.has_value()) { device_diagnostics.push_back(*slot0.diagnostic); }

    const DeviceCollectionResult slot1 = collect_device_identity(1, options.slot1_cuda_index);
    devices.push_back(slot1.device);
    if (slot1.diagnostic.has_value()) { device_diagnostics.push_back(*slot1.diagnostic); }

    const SoftwareCollectionResult software_result = collect_software_identity();

    const auto result = assemble_identity_snapshot(
        options.source_root, options.artifact, options.target_id, options.weights_id,
        options.container_version, devices, device_diagnostics, software_result.software,
        software_result.diagnostics, /*branch_name=*/{}, /*hostname=*/{},
        /*os_description=*/{}, command_line);

    IdentityGateResult out;
    out.identity = result.identity;
    out.diagnostics = result.diagnostics;
    out.missing = missing_identity_requirements(out.identity);
    out.qualified = result.all_collectors_succeeded() && out.missing.empty() &&
                    out.identity.source.state == SourceState::Clean;
    return out;
}

void print_identity_report(const IdentityGateResult& gate) {
    const CalibrationIdentity& identity = gate.identity;
    std::cout << "IDENTITY_SOURCE state=" << source_state_name(identity.source.state)
              << " commit=" << identity.source.commit << '\n';
    std::cout << "IDENTITY_BUILD build_id=" << identity.build.build_id
              << " cuda_arch_list=" << identity.build.cuda_arch_list
              << " build_type=" << identity.build.build_type << " compile_flags=[";
    for (std::size_t i = 0; i < identity.build.compile_flags.size(); ++i) {
        if (i != 0) { std::cout << ','; }
        std::cout << identity.build.compile_flags[i];
    }
    std::cout << "]\n";

    std::cout << "IDENTITY_ARTIFACT path=" << identity.artifact.path
              << " name=" << identity.artifact.name
              << " size_bytes=" << identity.artifact.size_bytes
              << " sha256=" << identity.artifact.sha256
              << " target_id=" << identity.artifact.target_id
              << " weights_id=" << identity.artifact.weights_id
              << " container_version=" << identity.artifact.container_version << '\n';

    for (const DeviceIdentity& device : identity.devices) { print_device(device); }

    std::cout << "IDENTITY_SOFTWARE driver_version=" << identity.software.driver_version
              << " cuda_runtime_version=" << identity.software.cuda_runtime_version
              << " cuda_toolkit_version=" << identity.software.cuda_toolkit_version << '\n';

    print_diagnostics(gate.diagnostics);
    for (const std::string& reason : gate.missing) { std::cout << "IDENTITY_MISSING " << reason << '\n'; }
    std::cout << "IDENTITY_QUALIFIED=" << (gate.qualified ? "YES" : "NO") << '\n';
}

const DeviceIdentity& device_for_slot(const CalibrationIdentity& identity, int slot) {
    for (const DeviceIdentity& device : identity.devices) {
        if (device.logical_slot == slot) { return device; }
    }
    throw std::runtime_error("identity is missing requested logical slot " + std::to_string(slot));
}

struct TransferCase {
    TransferPayloadClass payload_class;
    std::string label;
    std::uint64_t bytes = 0;
    int source_slot = 0;
    int destination_slot = 1;
};

std::string payload_name(TransferPayloadClass payload_class) {
    return std::string(ninfer::targets::calibration::transfer_payload_class_name(payload_class));
}

std::vector<unsigned char> expected_bytes(std::uint64_t bytes, unsigned char pattern) {
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("payload too large for host correctness check");
    }
    return std::vector<unsigned char>(static_cast<std::size_t>(bytes), pattern);
}

void assert_destination_pattern(int cuda_index, void* pointer, std::uint64_t bytes,
                                unsigned char pattern) {
    std::vector<unsigned char> host = expected_bytes(bytes, 0);
    CALIBRATION_CUDA_CHECK(cudaSetDevice(cuda_index));
    CALIBRATION_CUDA_CHECK(cudaMemcpy(host.data(), pointer, static_cast<std::size_t>(bytes),
                                      cudaMemcpyDeviceToHost));
    for (unsigned char byte : host) {
        if (byte != pattern) { throw std::runtime_error("transfer correctness check failed"); }
    }
}

double time_one_copy_us(void* dst, const void* src, std::uint64_t bytes, cudaStream_t stream) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    CALIBRATION_CUDA_CHECK(cudaEventCreate(&begin));
    CALIBRATION_CUDA_CHECK(cudaEventCreate(&end));
    CALIBRATION_CUDA_CHECK(cudaEventRecord(begin, stream));
    CALIBRATION_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(bytes),
                                           cudaMemcpyDeviceToDevice, stream));
    CALIBRATION_CUDA_CHECK(cudaEventRecord(end, stream));
    CALIBRATION_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CALIBRATION_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    CALIBRATION_CUDA_CHECK(cudaEventDestroy(begin));
    CALIBRATION_CUDA_CHECK(cudaEventDestroy(end));
    return static_cast<double>(ms) * 1000.0;
}

struct TransferRunStats {
    int cases = 0;
    int qualified = 0;
    int unqualified = 0;
    std::string raw_path;
    std::string summary_path;
    std::uint64_t raw_records = 0;
    std::uint64_t summary_records = 0;
};

struct RouteProbeResult {
    int can01 = 0;
    int can10 = 0;
    bool peer_capable = false;
};

RouteProbeResult probe_transfer_route(const ninfer::calibration_probe::Options& options) {
    RouteProbeResult route;
    CALIBRATION_CUDA_CHECK(cudaDeviceCanAccessPeer(&route.can01, options.slot0_cuda_index,
                                                   options.slot1_cuda_index));
    CALIBRATION_CUDA_CHECK(cudaDeviceCanAccessPeer(&route.can10, options.slot1_cuda_index,
                                                   options.slot0_cuda_index));
    route.peer_capable = route.can01 != 0 && route.can10 != 0;
    std::cout << "TRANSFER_ROUTE api=cudaMemcpyAsync kind=cudaMemcpyDeviceToDevice uva=YES"
              << " cudaDeviceCanAccessPeer_0_to_1=" << route.can01
              << " cudaDeviceCanAccessPeer_1_to_0=" << route.can10
              << " peer_capability=" << (route.peer_capable ? "AVAILABLE" : "UNAVAILABLE")
              << " host_involvement="
              << (route.peer_capable ? "unresolved" : "driver_managed_staging")
              << " route=" << (route.peer_capable ? "unresolved" : "runtime_managed_memcpy")
              << " sync=async overlap=no_overlap\n";
    return route;
}

TransferRunStats run_transfer_mode(const ninfer::calibration_probe::Options& options,
                                   const std::string& command_line) {
    const IdentityGateResult gate = collect_identity(options, command_line);
    print_identity_report(gate);
    if (!gate.qualified) { throw std::runtime_error("identity gate failed; refusing transfer timing"); }

    const int warmup = options.warmup >= 0
                           ? options.warmup
                           : (options.mode == ninfer::calibration_probe::ProbeMode::TransferSmoke ? 5 : 20);
    const int samples = options.samples >= 0
                            ? options.samples
                            : (options.mode == ninfer::calibration_probe::ProbeMode::TransferSmoke ? 10 : 100);

    const RouteProbeResult route = probe_transfer_route(options);
    if (route.peer_capable) {
        throw std::runtime_error(
            "route provenance incomplete: direct peer route needs separate proof before timing");
    }

    const auto hidden = ninfer::targets::calibration_measurement::current_layer_boundary_hidden_geometry();
    const auto logits = ninfer::targets::calibration_measurement::current_logits_result_geometry();
    const std::uint64_t hidden_bytes = hidden.total_bytes(1);
    const std::uint64_t logits_bytes = logits.total_bytes(1);
    std::vector<TransferCase> cases = {
        {TransferPayloadClass::LayerBoundaryHidden, "layer_0_to_1", hidden_bytes, 0, 1},
        {TransferPayloadClass::LayerBoundaryHidden, "layer_1_to_0", hidden_bytes, 1, 0},
        {TransferPayloadClass::LogitsResult, "logits_0_to_1", logits_bytes, 0, 1},
        {TransferPayloadClass::LogitsResult, "logits_1_to_0", logits_bytes, 1, 0},
    };

    namespace cal = ninfer::targets::calibration;
    namespace cio = ninfer::targets::calibration_io;
    namespace meas = ninfer::targets::calibration_measurement;
    const std::string raw_dir =
        options.raw_dir.empty() ? std::string(cal::kDefaultCalibrationRawDir) : options.raw_dir;
    const std::string summary_dir = options.summary_dir.empty()
                                        ? std::string(cal::kDefaultCalibrationSummariesDir)
                                        : options.summary_dir;
    cio::RawWriter raw(raw_dir, options.run_id);
    std::vector<cal::CalibrationRecord> records;
    records.reserve(static_cast<std::size_t>(cases.size() * samples));

    TransferRunStats stats;
    stats.cases = static_cast<int>(cases.size());

    for (const TransferCase& c : cases) {
        const DeviceIdentity& src_device = device_for_slot(gate.identity, c.source_slot);
        const DeviceIdentity& dst_device = device_for_slot(gate.identity, c.destination_slot);
        const int src_cuda = src_device.physical_index;
        const int dst_cuda = dst_device.physical_index;
        const unsigned char pattern =
            static_cast<unsigned char>(0x30 + c.source_slot * 3 + c.destination_slot);

        void* src = nullptr;
        void* dst = nullptr;
        cudaStream_t stream = nullptr;
        CALIBRATION_CUDA_CHECK(cudaSetDevice(src_cuda));
        CALIBRATION_CUDA_CHECK(cudaMalloc(&src, static_cast<std::size_t>(c.bytes)));
        CALIBRATION_CUDA_CHECK(cudaMemset(src, pattern, static_cast<std::size_t>(c.bytes)));
        CALIBRATION_CUDA_CHECK(cudaSetDevice(dst_cuda));
        CALIBRATION_CUDA_CHECK(cudaMalloc(&dst, static_cast<std::size_t>(c.bytes)));
        CALIBRATION_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CALIBRATION_CUDA_CHECK(cudaMemsetAsync(dst, 0, static_cast<std::size_t>(c.bytes), stream));

        for (int i = 0; i < warmup; ++i) {
            CALIBRATION_CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(c.bytes),
                                                   cudaMemcpyDeviceToDevice, stream));
        }
        CALIBRATION_CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int i = 0; i < samples; ++i) {
            CALIBRATION_CUDA_CHECK(cudaMemsetAsync(dst, 0, static_cast<std::size_t>(c.bytes), stream));
            const double us = time_one_copy_us(dst, src, c.bytes, stream);
            assert_destination_pattern(dst_cuda, dst, c.bytes, pattern);

            cal::TransferIdentity transfer =
                meas::transfer_identity_skeleton({c.source_slot, c.destination_slot, c.payload_class});
            transfer.source_device_uuid = src_device.uuid;
            transfer.destination_device_uuid = dst_device.uuid;
            transfer.payload_bytes = c.bytes;
            transfer.transfer_api = "cudaMemcpyAsync(cudaMemcpyDeviceToDevice,UVA)";
            transfer.sync = TransferSync::Async;
            transfer.route = TransferRoute::RuntimeManagedMemcpy;
            transfer.host_involvement = HostInvolvement::DriverManagedStaging;
            transfer.overlap = TransferOverlap::NoOverlap;

            meas::ObservationInput input;
            input.identity = gate.identity;
            input.kind = RecordKind::Transfer;
            // Current M5.4c-2 transfer-smoke/transfer-run cases are all part of the batch-1 decode
            // path; operation.phase is a common-core qualification requirement for every record
            // kind (calibration.cpp missing_requirements), not just Operation-kind records.
            input.operation.phase = ninfer::targets::calibration::ExecutionPhase::Decode;
            input.transfer = transfer;
            input.run_id = options.run_id;
            input.observation_id = c.label;
            input.sample_index = static_cast<std::uint64_t>(i);
            input.value = us;
            input.unit = "us";
            input.timestamp_unix_ms = now_unix_ms();
            input.evidence = EvidenceClass::Measured;
            input.requested_validity = ValidityState::Valid;
            input.correctness = CorrectnessStatus::Pass;
            meas::EmittedRecord emitted = meas::emit_record(input);
            if (emitted.is_qualified_evidence) {
                ++stats.qualified;
            } else {
                ++stats.unqualified;
                std::cout << "TRANSFER_UNQUALIFIED case=" << c.label << " sample=" << i;
                for (const std::string& reason : emitted.missing_requirements) {
                    std::cout << " reason=\"" << reason << "\"";
                }
                std::cout << '\n';
            }
            raw.append(emitted.record);
            records.push_back(std::move(emitted.record));
        }

        CALIBRATION_CUDA_CHECK(cudaStreamDestroy(stream));
        CALIBRATION_CUDA_CHECK(cudaSetDevice(src_cuda));
        CALIBRATION_CUDA_CHECK(cudaFree(src));
        CALIBRATION_CUDA_CHECK(cudaSetDevice(dst_cuda));
        CALIBRATION_CUDA_CHECK(cudaFree(dst));
    }

    raw.close();
    stats.raw_path = raw.file_path();
    stats.raw_records = raw.records_written();
    const auto summaries = ninfer::targets::calibration_summary::derive_summaries(
        records, gate.identity, static_cast<std::uint64_t>(samples));
    cio::SummaryWriter summary(summary_dir, options.run_id);
    for (const auto& s : summaries) { summary.append(s); }
    summary.close();
    stats.summary_path = summary.file_path();
    stats.summary_records = summary.summaries_written();

    std::cout << "TRANSFER_RESULT run_id=" << options.run_id << " warmup=" << warmup
              << " samples=" << samples << " cases=" << stats.cases
              << " qualified_samples=" << stats.qualified
              << " unqualified_samples=" << stats.unqualified
              << " raw_path=" << stats.raw_path << " summary_path=" << stats.summary_path
              << " summaries=" << stats.summary_records << '\n';
    return stats;
}

// Runs identity mode: collects device/software identity via the CUDA collectors, assembles the
// identity snapshot via the shared M5.4b assembler, prints a report, and evaluates the identity
// gate. Never writes a file, never runs a timing benchmark.
int run_identity_mode(const ninfer::calibration_probe::Options& options,
                       const std::string& command_line) {
    const IdentityGateResult gate = collect_identity(options, command_line);
    print_identity_report(gate);
    return gate.qualified ? 0 : 1;
}

int run_transfer_route_mode(const ninfer::calibration_probe::Options& options,
                            const std::string& command_line) {
    const IdentityGateResult gate = collect_identity(options, command_line);
    print_identity_report(gate);
    if (!gate.qualified) { throw std::runtime_error("identity gate failed; refusing route check"); }
    const RouteProbeResult route = probe_transfer_route(options);
    return route.peer_capable ? 2 : 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::calibration_probe::Options options;
    try {
        options = ninfer::calibration_probe::parse_options(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << ninfer::calibration_probe::usage_text(argv[0]);
        return 1;
    }

    try {
        const std::string command_line = reconstruct_command_line(argc, argv);
        switch (options.mode) {
        case ninfer::calibration_probe::ProbeMode::Identity:
            return run_identity_mode(options, command_line);
        case ninfer::calibration_probe::ProbeMode::TransferRoute:
            return run_transfer_route_mode(options, command_line);
        case ninfer::calibration_probe::ProbeMode::TransferSmoke:
        case ninfer::calibration_probe::ProbeMode::TransferRun:
            (void)run_transfer_mode(options, command_line);
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 1; // unreachable: switch above is exhaustive over ProbeMode.
}

#pragma once

// M5.4b sec 10: CUDA-only device/software identity collection. Kept out of calibration_collect.h
// (host-only, decision 3) so the CPU-only calibration test target never pulls in a CUDA include.
// Callers feed the results into calibration_collect::assemble_identity_snapshot(), which itself
// stays CUDA-free.

#include "targets/calibration.h"
#include "targets/calibration_collect.h"

#include <optional>
#include <vector>

namespace ninfer::targets::calibration_collect {

struct DeviceCollectionResult {
    ninfer::targets::calibration::DeviceIdentity device; // fail-closed defaults if diagnostic set.
    std::optional<CollectionDiagnostic> diagnostic;
};

// Queries CUDA runtime device properties for `physical_index` and tags the result with the
// requested `logical_slot`. Fail-closed: on any CUDA query failure, `device` keeps its
// default-unknown fields (empty uuid, cc 0/0, vram 0) and `diagnostic` reports the failure -- never
// a fabricated UUID or assumed VRAM.
[[nodiscard]] DeviceCollectionResult collect_device_identity(int logical_slot, int physical_index);

struct SoftwareCollectionResult {
    ninfer::targets::calibration::SoftwareIdentity software;
    std::vector<CollectionDiagnostic> diagnostics;
};

// Queries live CUDA driver/runtime version plus this binary's compiled-against toolkit version.
// Fail-closed per field: a failed query leaves that field empty and adds a diagnostic, never a
// default value substituted for an unqueried one.
[[nodiscard]] SoftwareCollectionResult collect_software_identity();

} // namespace ninfer::targets::calibration_collect

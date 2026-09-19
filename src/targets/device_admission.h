#pragma once

// Per-device admission accounting shared by the live load path (registry.cpp phase A / phase B)
// and the M5 analytical placement planner. Extracted verbatim from registry.cpp (M5.0) so both
// consumers use ONE memory vocabulary: the byte categories below keep exactly the meaning the
// `[admission]` log lines have carried since M2.

#include "ninfer/types.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::targets {

// Live cudaMemGetInfo reads for one device ordinal (scoped device switch).
[[nodiscard]] std::size_t current_free_device_bytes(int device);
[[nodiscard]] std::size_t current_total_device_bytes(int device);

// Empirically calibrated allocator/driver reserve (8 MiB default, NINFER_ALLOCATOR_RESERVE_BYTES
// override). The ONLY reserve policy; see the calibration note in device_admission.cpp.
inline constexpr std::uint64_t kAllocatorDriverReserveBytes = 8ULL * 1024ULL * 1024ULL;
[[nodiscard]] std::uint64_t effective_allocator_reserve_bytes();

enum class DeviceAdmissionDecision { Admit, Reject };

struct DeviceAdmissionPlan {
    int device                          = 0;
    int slot                            = 0;
    std::uint64_t planned_weight_bytes  = 0;
    std::uint64_t exact_persistent_bytes = 0;
    std::uint64_t persistent_kv_payload_bytes = 0; // report-only, already inside persistent_bytes
    std::uint64_t workspace_bytes       = 0;
    std::uint64_t graph_bytes           = 0;
    std::uint64_t request_transient_bytes = 0;
    std::uint64_t allocator_driver_reserve = kAllocatorDriverReserveBytes;
    std::uint64_t total_required_bytes  = 0;
    // Phase A: this device's own live cudaMemGetInfo free (pre-materialize). Phase B: the
    // caller-supplied post-weight measured free (`free_before_runtime[rank]`), not re-measured here.
    std::uint64_t effective_capacity    = 0;
    std::int64_t predicted_headroom     = 0; // effective_capacity - total_required_bytes, may be negative
    DeviceAdmissionDecision decision    = DeviceAdmissionDecision::Admit;
    std::string reason_code;    // e.g. "WEIGHT_CAPACITY", "TOTAL_DEVICE_CAPACITY"
    std::string reason_detail;
    // KV capacity resolution diagnostics (M2.1). Empty/zero at phase A, where the KV point hasn't
    // been resolved yet; populated by the phase-B call site from `capacity_resolution`.
    std::string kv_mode;                  // "auto" or "explicit"; empty at phase A
    std::uint32_t resolved_tokens       = 0;
    std::uint32_t resolved_page_groups  = 0;
    // M4/M5.0: which slot owns the output endpoint group for the placement this plan describes.
    // Filled by the CALLER from its PlacementSpec (manual path: Package::manual_placement; planner:
    // the explicit candidate) -- the printer no longer reads the environment, so a candidate report
    // cannot be mislabelled with the manual selection.
    int endpoint_owner_slot             = 0;
};


// Sentinel: caller wants `current_free_device_bytes(device)` measured fresh (phase A, pre-
// materialize). Phase B and the M5 planner pass an explicit measured capacity instead.
inline constexpr std::uint64_t kMeasureCapacityLive = static_cast<std::uint64_t>(-1);

// Builds the per-device admission plan for device SLOT `slot` (0 or 1). See device_admission.cpp
// for the phase A / phase B / planner call-site contract and the ANTI-DOUBLE-COUNT rule.
// `reserve_override`: kUseEffectiveReserve (default) charges effective_allocator_reserve_bytes();
// the M5 planner passes the reserve it captured once for a whole report so every candidate ledger
// is charged the identical value.
inline constexpr std::uint64_t kUseEffectiveReserve = static_cast<std::uint64_t>(-1);
[[nodiscard]] DeviceAdmissionPlan
build_device_admission_plan(int device, int slot, std::uint64_t planned_weight_bytes,
                            const targets::qwen3_6::DeviceLayoutPreview& preview,
                            std::uint64_t capacity_override = kMeasureCapacityLive,
                            std::uint64_t reserve_override  = kUseEffectiveReserve);

// Greppable single-line `[admission]` summary. `phase`: "A", "B" (live) -- the M5 planner renders
// its own report and does not use this.
void print_device_admission_summary(const DeviceAdmissionPlan& plan, std::string_view split_context,
                                    std::string_view phase);

} // namespace ninfer::targets

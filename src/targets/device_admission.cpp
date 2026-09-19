#include "targets/device_admission.h"

#include "core/device.h"

#include <cerrno>
#include <cstdlib>
#include <iostream>

namespace ninfer::targets {
namespace {

// Save/restore the current device around a per-device query.
class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

} // namespace

std::size_t current_free_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

std::size_t current_total_device_bytes(int device) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes;
}

// cudaMemGetInfo().free overstates the largest satisfiable cudaMalloc: driver/context bookkeeping
// and allocator fragmentation both eat into it before a real allocation lands. 128 MiB was the
// original guess to separate the failing 58/6 GPU0 case from known-good cases, but ground-truth
// M2 recalibration (2026-09-17) found it over-rejected a genuinely safe config: 58/6 GPU0's real
// nominal surplus (measured, no reserve) is only 4,623,104 B, while 12/52 GPU1's is 10,175,999 B
// -- and running 12/52 with the reserve forced to 0 completes on real hardware (GPU1 materializes
// down to 6.88 MiB free, exit 0, tokens generated). So the true safe/unsafe boundary sits inside
// [4.6 MB, 10.2 MB], a band only 2.2x wide.
//
// A reserve keyed to the largest single pending arena (persistent vs. workspace) was considered
// as a more "mechanistic" alternative, since 58/6's actual failure was one large single cudaMalloc
// succeeding as a summed-total check but failing in practice. It was rejected: at 58/6 GPU0 the
// largest single arena is persistent_bytes (536,957,184), not the workspace_bytes (180,953,088)
// allocation that actually failed -- so it keys on an arena that never failed, and the resulting
// window (0.86%-2.09% of that arena) is *wider* (2.42x) than the plain scalar band, not narrower.
// It bought no precision for the extra complexity, so it was not implemented.
//
// 8 MiB (8,388,608 B) is chosen as a round value inside the empirical band, near its midpoint
// (~7.4 MB): it still rejects 58/6 (8,388,608 > 4,623,104 shortfall-free surplus) and still admits
// 12/52 (8,388,608 <= 10,175,999). This is an EMPIRICALLY CALIBRATED value fit to two measured
// data points, not a derived constant -- retune here if a future device/driver/model combination
// needs a different margin; this is the only place the value is defined.
//
// Tunable for calibration: NINFER_ALLOCATOR_RESERVE_BYTES overrides this at process start when
// set to a valid non-negative integer (parsed with strtoull so "0" is distinguishable from unset
// or garbage); unset or unparsable falls back to this constant. The effective value actually used
// is always the one printed in the [admission] line's allocator_driver_reserve field.
// (kAllocatorDriverReserveBytes is defined in device_admission.h; the note above is its owner.)

std::uint64_t effective_allocator_reserve_bytes() {
    const char* env = std::getenv("NINFER_ALLOCATOR_RESERVE_BYTES");
    if (env == nullptr || *env == '\0') {
        return kAllocatorDriverReserveBytes;
    }
    char* end            = nullptr;
    errno                = 0;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || *end != '\0' || errno == ERANGE) {
        std::cerr << "[admission] NINFER_ALLOCATOR_RESERVE_BYTES='" << env
                 << "' is not a valid non-negative integer; using default "
                 << kAllocatorDriverReserveBytes << "\n";
        return kAllocatorDriverReserveBytes;
    }
    return static_cast<std::uint64_t>(parsed);
}



// Builds the per-device admission plan for device SLOT `slot` (0 or 1, matching
// device_slot/layer_owner[]). Two callers, two phases (see registry.cpp construct_registered):
//   Phase A (pre-materialize): `preview` is the MINIMUM-page-group layout -- the smallest the
//     finalized sequence plan can be -- `planned_weight_bytes` is the planned weight estimate, and
//     `capacity_override` is left at the default (a fresh live cudaMemGetInfo read). Conservative
//     feasibility gate; can only under-reject, never over-reject.
//   Phase B (post-materialize, M2.1): `preview` is built at the RESOLVED `--kv-capacity auto` page
//     count, `planned_weight_bytes` is 0 (weights are already resident, so `capacity_override` --
//     the same `free_before_runtime[slot]` the resolver itself was sized against -- already has
//     them subtracted; adding the planned estimate again would double-count them), and
//     `capacity_override` is passed explicitly rather than re-measuring live. Exact check: what it
//     approves is what finalize() then allocates.
//

// ANTI-DOUBLE-COUNT: `planned_weight_bytes` (load_plan.materialization().device_capacity_bytes[slot])
// ALREADY INCLUDES this device's endpoint tensors. `exact_persistent_bytes` (from that slot's own
// layout) ALREADY INCLUDES KV and GDN state bytes. Both `persistent_kv_payload_bytes` below and any
// future endpoint decomposition are REPORT-ONLY fields; they are NEVER added again on top of
// `planned_weight_bytes` or `exact_persistent_bytes` in the summation below.
DeviceAdmissionPlan build_device_admission_plan(int device, int slot, std::uint64_t planned_weight_bytes,
                                                const targets::qwen3_6::DeviceLayoutPreview& preview,
                                                std::uint64_t capacity_override,
                                                std::uint64_t reserve_override) {
    DeviceAdmissionPlan plan;
    plan.device                     = device;
    plan.slot                       = slot;
    plan.planned_weight_bytes       = planned_weight_bytes;
    plan.exact_persistent_bytes     = preview.persistent_bytes;
    plan.persistent_kv_payload_bytes = preview.persistent_kv_payload_bytes;
    plan.workspace_bytes            = preview.workspace_capacity_bytes;
    plan.graph_bytes                = preview.graph_allowance_bytes;
    plan.request_transient_bytes    = preview.request_transient_capacity_bytes;
    plan.allocator_driver_reserve   = reserve_override == kUseEffectiveReserve
                                          ? effective_allocator_reserve_bytes()
                                          : reserve_override;
    // Sum: weights (endpoints included) + persistent (KV/GDN included) + workspace + request
    // transient + graph allowance + fixed allocator/driver reserve. No addend here duplicates a
    // component already folded into `planned_weight_bytes` or `exact_persistent_bytes` -- see the
    // ANTI-DOUBLE-COUNT comment above.
    // Phase A note: at the CALL SITE that passes the MINIMUM-page-group preview, this total is
    // conservative-permissive under `--kv-capacity auto` (resolved plan can be larger -- auto-fit
    // resolves after phase A runs), so it may under-reject there but never over-reject. That gap is
    // closed exactly, not projected, by the phase-B call site below (M2.1): same function, a
    // preview built at the RESOLVED page count via the identical build_sequence_candidate path
    // finalize() uses. Do not add a second, curve-delta-based byte-accounting path here.
    plan.total_required_bytes = plan.planned_weight_bytes + plan.exact_persistent_bytes +
                                plan.workspace_bytes + plan.request_transient_bytes +
                                plan.graph_bytes + plan.allocator_driver_reserve;
    plan.effective_capacity = capacity_override == kMeasureCapacityLive
                                  ? current_free_device_bytes(device)
                                  : capacity_override;
    plan.predicted_headroom = static_cast<std::int64_t>(plan.effective_capacity) -
                              static_cast<std::int64_t>(plan.total_required_bytes);
    if (plan.predicted_headroom >= 0) {
        plan.decision      = DeviceAdmissionDecision::Admit;
        plan.reason_code    = "OK";
        plan.reason_detail  = "predicted headroom " + std::to_string(plan.predicted_headroom) + " bytes";
        return plan;
    }
    plan.decision = DeviceAdmissionDecision::Reject;
    // Attribute the shortfall to the most specific limiting category available, in the same
    // priority order admission is conventionally reasoned about: weights first (largest, measured
    // outright), then persistent (exact per-owner layout), then workspace, then the fixed reserve,
    // falling back to a generic total-capacity label if none of those alone explain it.
    if (plan.planned_weight_bytes > plan.effective_capacity) {
        plan.reason_code = "WEIGHT_CAPACITY";
    } else if (plan.planned_weight_bytes + plan.exact_persistent_bytes > plan.effective_capacity) {
        plan.reason_code = "PERSISTENT_CAPACITY";
    } else if (plan.planned_weight_bytes + plan.exact_persistent_bytes + plan.workspace_bytes >
              plan.effective_capacity) {
        plan.reason_code = "WORKSPACE_CAPACITY";
    } else if (plan.planned_weight_bytes + plan.exact_persistent_bytes + plan.workspace_bytes +
                  plan.request_transient_bytes + plan.graph_bytes <=
              plan.effective_capacity) {
        plan.reason_code = "ALLOCATOR_RESERVE";
    } else {
        plan.reason_code = "TOTAL_DEVICE_CAPACITY";
    }
    plan.reason_detail = "shortfall " + std::to_string(-plan.predicted_headroom) + " bytes";
    return plan;
}

// Greppable single-line-per-category summary block for one device. Printed for every device under
// layer split (report-only at TP2/single-device, per M2 item 5); never per-tensor spam.
// `phase`: "A" (pre-materialize, minimum-page-group preview) or "B" (post-materialize, resolved-
// page-group preview, M2.1) -- distinguishes the two admission passes in the log stream.
void print_device_admission_summary(const DeviceAdmissionPlan& plan, std::string_view split_context,
                                    std::string_view phase) {
    // `phase=` is appended at the END of the line (not inserted after the "[admission]" tag) so
    // any existing consumer that greps/parses "[admission] device=..." from before M2.1 keeps
    // matching field-for-field; only new consumers need to read the trailing field.
    std::cerr << "[admission] device=" << plan.device << " slot=" << plan.slot
             << " split=" << split_context
             << " weight_bytes=" << plan.planned_weight_bytes
             << " persistent_bytes=" << plan.exact_persistent_bytes
             << " persistent_kv_payload_bytes(report_only)=" << plan.persistent_kv_payload_bytes
             << " workspace_bytes=" << plan.workspace_bytes
             << " request_transient_bytes=" << plan.request_transient_bytes
             << " graph_bytes=" << plan.graph_bytes
             << " allocator_driver_reserve=" << plan.allocator_driver_reserve
             << " total_required_bytes=" << plan.total_required_bytes
             << " effective_capacity=" << plan.effective_capacity
             << " predicted_headroom=" << plan.predicted_headroom
             << " decision=" << (plan.decision == DeviceAdmissionDecision::Admit ? "ADMIT" : "REJECT")
             << " reason=" << plan.reason_code << " (" << plan.reason_detail << ")";
    if (!plan.kv_mode.empty()) {
        std::cerr << " kv_mode=" << plan.kv_mode << " resolved_tokens=" << plan.resolved_tokens
                 << " resolved_page_groups=" << plan.resolved_page_groups;
    }
    // M4: which device the output endpoint group (final_norm + output_head) was placed on, so two
    // runs that differ ONLY in placement are distinguishable in the log rather than by filename.
    // M5.0: plumbed from the caller's PlacementSpec (plan.endpoint_owner_slot) instead of parsed
    // from the environment here, so the live path prints the EFFECTIVE manual selection and a
    // planner candidate prints its own explicit slot.
    std::cerr << " endpoint_owner_slot=" << plan.endpoint_owner_slot << " phase=" << phase << "\n";
}


} // namespace ninfer::targets

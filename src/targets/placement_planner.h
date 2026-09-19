#pragma once

// M5.1 / M5.2: analytical (pre-materialization) placement feasibility for a contiguous layer split
// across two device slots, and exhaustive enumeration of every structurally legal
// boundary x endpoint-owner candidate.
//
// HARD FEASIBILITY ONLY. This module answers "can this placement legally run within the captured
// capacity snapshot, and if not, exactly which device and byte category rejects it?". It contains
// no performance model, no score, no ranking and no selection; every candidate is reported.
//
// Authorities reused (never re-derived here):
//   weights      -> the real materialization planner (Package::plan_load with an explicit
//                   PlacementSpec), supplied per device by the caller;
//   persistent/KV/GDN/workspace/request-transient/graph -> SequencePlanner::layout_preview(slot[,groups]),
//                   supplied as a callable so the engine is testable with synthetic layouts;
//   KV capacity  -> runtime::resolve_kv_capacity_symmetric (four-arg exact Automatic form, M2.2b);
//   reserve      -> effective_allocator_reserve_bytes() (device_admission.h), supplied by caller;
//   ledger       -> build_device_admission_plan (device_admission.h), the same function and the same
//                   attribution order phase A / phase B use live.
//
// Capacity basis: `capacity_snapshot_bytes[slot]` is the device's free memory captured BEFORE any
// candidate weights exist. Every figure is analytical against that snapshot; actual load still
// passes the existing phase A / phase B admission, which remain the fail-closed authority.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "targets/device_admission.h"
#include "targets/placement.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::targets {

inline constexpr int kPlacementSlots = 2;

struct PlacementDeviceInputs {
    int device                            = 0; // CUDA ordinal, report only
    std::uint64_t planned_weight_bytes    = 0; // from the real load plan for THIS candidate
    std::uint64_t capacity_snapshot_bytes = 0; // pre-materialization free bytes, captured once
};

// `preview(slot, groups)`: nullopt previews the curve's minimum page-group point; a value previews
// that exact page count (SequencePlanner::layout_preview semantics). Must be pure.
using LayoutPreviewFn = std::function<qwen3_6::DeviceLayoutPreview(int, std::optional<std::uint32_t>)>;

struct PlacementFeasibilityInputs {
    PlacementSpec spec;
    std::array<PlacementDeviceInputs, kPlacementSlots> devices{};
    std::uint64_t allocator_reserve_bytes = 0;
    KvCapacityPolicy kv_policy;
    runtime::SequenceCapacityCurve curve;
    LayoutPreviewFn preview;
};

enum class PlacementStatus { Feasible, Rejected };

// Structured reason codes. The byte-category codes are exactly build_device_admission_plan's
// (WEIGHT_CAPACITY, PERSISTENT_CAPACITY, WORKSPACE_CAPACITY, ALLOCATOR_RESERVE,
// TOTAL_DEVICE_CAPACITY). KV_CAPACITY is a rejection raised by the KV resolver itself.
// LAYOUT_PREVIEW_UNAVAILABLE: the planner could not produce a layout for this candidate (the live
// path would skip the check; the planner fails closed instead). PLANNER_ERROR: candidate
// construction threw (reported, never propagated -- see enumerate_placements). On the Qwen3.8-27B
// target this is exactly the set of boundaries the live runtime itself refuses to load (an owner
// with zero full-attention layers -> "Paged KV cache geometry is invalid", or zero GDN layers ->
// "LinearAttentionStatePool layers must be nonzero"); the detail carries that same message.
inline constexpr const char* kReasonKvCapacity        = "KV_CAPACITY";
inline constexpr const char* kReasonPreviewUnavailable = "LAYOUT_PREVIEW_UNAVAILABLE";
inline constexpr const char* kReasonPlannerError       = "PLANNER_ERROR";

struct PlacementFeasibility {
    PlacementSpec spec;
    PlacementStatus status = PlacementStatus::Rejected;
    // Per-slot ledgers. At the RESOLVED page-group point when the resolver succeeded; at the
    // minimum point when the candidate was rejected before/at resolution; `ledgers_valid` is false
    // only when no layout preview could be built at all.
    std::array<DeviceAdmissionPlan, kPlacementSlots> ledgers{};
    bool ledgers_valid = false;
    // KV outcome (zero/empty when the resolver rejected or was never reached).
    std::string kv_mode; // "auto" or "explicit"
    std::uint32_t resolved_tokens      = 0;
    std::uint32_t resolved_page_groups = 0;
    // Rejection attribution (limiting_slot == -1 when Feasible).
    int limiting_slot = -1;
    std::string reason_code;
    std::string reason_detail;
    std::uint64_t required_bytes  = 0; // of the limiting slot, in the failing check's own basis
    std::uint64_t available_bytes = 0;
    std::int64_t shortfall_bytes  = 0; // required - available (> 0 when rejected)
};

// Evaluates ONE candidate. Never throws for a rejection; only programming errors (a null preview
// callable) propagate.
[[nodiscard]] PlacementFeasibility evaluate_placement(const PlacementFeasibilityInputs& inputs);

// ---- M5.2: enumeration -------------------------------------------------------------------------

struct PlacementCandidateResult {
    PlacementFeasibility feasibility;
    int owner0_layers                = 0;
    int owner1_layers                = 0;
    int owner0_full_attention_layers = 0;
    int owner0_gdn_layers            = 0;
    int owner1_full_attention_layers = 0;
    int owner1_gdn_layers            = 0;
};

struct PlacementLayerCounts {
    int owner0_full_attention = 0;
    int owner0_gdn            = 0;
    int owner1_full_attention = 0;
    int owner1_gdn            = 0;
};

// Target-bound callbacks the enumerator drives. Each may throw for a candidate; the enumerator
// catches per candidate and records PLANNER_ERROR rather than propagating.
struct PlacementEnumerationSource {
    int layer_count = 0;
    // The SAME legality rule manual `--layer-split` uses (Target::resolve_options); returns false
    // (or throws) for an illegal boundary.
    std::function<bool(int boundary)> boundary_is_legal;
    // Builds the full feasibility inputs for one explicit candidate (weights from the real load
    // plan, previews/curve from the candidate's own sequence planner, snapshot + reserve shared).
    std::function<PlacementFeasibilityInputs(PlacementSpec)> make_inputs;
    std::function<PlacementLayerCounts(int boundary)> layer_counts; // optional
};

struct PlacementReport {
    // Request / capacity envelope (the runtime's own option vocabulary).
    std::uint32_t max_context     = 0;
    KvCapacityPolicy kv_policy;
    std::uint32_t max_concurrency = 1;
    std::string kv_dtype;
    int layer_count = 0;
    // Capacity snapshot this report is tied to. Free VRAM is dynamic: a result is valid for THIS
    // snapshot, not a timeless hardware fact.
    std::array<int, kPlacementSlots> devices{};
    std::array<std::uint64_t, kPlacementSlots> capacity_snapshot_bytes{};
    std::uint64_t allocator_reserve_bytes = 0;
    // Deterministic order: boundary ascending, endpoint owner 0 then 1.
    std::vector<PlacementCandidateResult> candidates;
    std::size_t feasible_count = 0;
    std::size_t rejected_count = 0;
};

// Enumerates boundary in [1, layer_count-1] (filtered by `boundary_is_legal`) x endpoint owner
// {0, 1}, evaluates each through evaluate_placement, and returns the complete set. Never throws
// for a candidate failure. `report` must have its envelope/snapshot fields filled by the caller.
void enumerate_placements(const PlacementEnumerationSource& source, PlacementReport& report);

// Deterministic human-readable rendering. Prints PASS/REJECT per candidate; never SELECT, BEST,
// OPTIMAL or PREFERRED -- there is no ranker.
[[nodiscard]] std::string render_placement_report(const PlacementReport& report);

} // namespace ninfer::targets

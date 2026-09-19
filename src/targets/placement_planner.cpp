#include "targets/placement_planner.h"

#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ninfer::targets {
namespace {

// B_i: the slot's own minimum-point reservation -- the identical composition registry.cpp feeds
// the four-arg resolver and layouts_impl.h sums into `device_reservation_bytes`.
std::size_t reservation_base(const qwen3_6::DeviceLayoutPreview& preview) {
    return preview.persistent_bytes + preview.workspace_capacity_bytes +
           preview.request_transient_capacity_bytes + preview.graph_allowance_bytes;
}

// Attributes a rejection to the ledger with the most negative predicted headroom.
void attribute_from_ledgers(PlacementFeasibility& out) {
    int worst                  = -1;
    std::int64_t worst_headroom = 0;
    for (int slot = 0; slot < kPlacementSlots; ++slot) {
        const DeviceAdmissionPlan& ledger = out.ledgers[static_cast<std::size_t>(slot)];
        if (ledger.decision != DeviceAdmissionDecision::Reject) { continue; }
        if (worst < 0 || ledger.predicted_headroom < worst_headroom) {
            worst          = slot;
            worst_headroom = ledger.predicted_headroom;
        }
    }
    if (worst < 0) { return; }
    const DeviceAdmissionPlan& ledger = out.ledgers[static_cast<std::size_t>(worst)];
    out.status          = PlacementStatus::Rejected;
    out.limiting_slot   = worst;
    out.reason_code     = ledger.reason_code;
    out.reason_detail   = ledger.reason_detail;
    out.required_bytes  = ledger.total_required_bytes;
    out.available_bytes = ledger.effective_capacity;
    out.shortfall_bytes = -ledger.predicted_headroom;
}

bool build_ledgers(const PlacementFeasibilityInputs& inputs, std::optional<std::uint32_t> groups,
                   PlacementFeasibility& out) {
    for (int slot = 0; slot < kPlacementSlots; ++slot) {
        const auto index = static_cast<std::size_t>(slot);
        const qwen3_6::DeviceLayoutPreview preview = inputs.preview(slot, groups);
        if (!preview.available) {
            out.ledgers_valid   = false;
            out.status          = PlacementStatus::Rejected;
            out.limiting_slot   = slot;
            out.reason_code     = kReasonPreviewUnavailable;
            out.reason_detail   = groups.has_value()
                                      ? "no layout preview at " + std::to_string(*groups) +
                                            " page groups"
                                      : "no minimum-point layout preview";
            return false;
        }
        const PlacementDeviceInputs& device = inputs.devices[index];
        // Same function, same summation and attribution as live phase A/B; the reserve is the
        // one captured for this report so every candidate is charged the identical value.
        DeviceAdmissionPlan ledger = build_device_admission_plan(
            device.device, slot, device.planned_weight_bytes, preview,
            /*capacity_override=*/device.capacity_snapshot_bytes,
            /*reserve_override=*/inputs.allocator_reserve_bytes);
        ledger.endpoint_owner_slot = inputs.spec.endpoint_owner_slot;
        out.ledgers[index]         = ledger;
    }
    out.ledgers_valid = true;
    return true;
}

std::string bytes_str(std::uint64_t value) { return std::to_string(value); }

} // namespace

PlacementFeasibility evaluate_placement(const PlacementFeasibilityInputs& inputs) {
    if (!inputs.preview) { throw std::invalid_argument("evaluate_placement: preview callable is empty"); }
    PlacementFeasibility out;
    out.spec    = inputs.spec;
    out.kv_mode = inputs.kv_policy.mode == KvCapacityMode::Automatic ? "auto" : "explicit";
    // Feasible until a stage rejects; every stage below either returns a Rejected result or
    // leaves this untouched.
    out.status = PlacementStatus::Feasible;

    // Stage 1 -- minimum-point ledger (phase-A equivalent, reserve included). Rejects weights that
    // do not fit, then persistent / workspace / reserve / total, with the live attribution order.
    if (!build_ledgers(inputs, std::nullopt, out)) { return out; }
    attribute_from_ledgers(out);
    if (out.status == PlacementStatus::Rejected) { return out; }

    // Stage 2 -- KV resolution on the reserve-protected budget F_i = snapshot_i - W_i - R with the
    // slot's own base B_i. Automatic: exact min_i(F_i - B_i) (four-arg, M2.2b). Explicit: the same
    // call delegates to the unchanged conservative form. The reserve is subtracted here and added
    // back exactly once in the resolved ledger below, so KV can never be sized into it.
    std::array<std::size_t, kPlacementSlots> budgets{};
    std::array<std::size_t, kPlacementSlots> bases{};
    for (int slot = 0; slot < kPlacementSlots; ++slot) {
        const auto index = static_cast<std::size_t>(slot);
        const PlacementDeviceInputs& device = inputs.devices[index];
        // Stage 1 admitted, so snapshot >= W + B + R holds and this cannot underflow.
        budgets[index] = static_cast<std::size_t>(device.capacity_snapshot_bytes -
                                                  device.planned_weight_bytes -
                                                  inputs.allocator_reserve_bytes);
        bases[index]   = reservation_base(inputs.preview(slot, std::nullopt));
    }
    runtime::KvCapacityResolution resolution;
    try {
        resolution = runtime::resolve_kv_capacity_symmetric(inputs.kv_policy, inputs.curve, budgets,
                                                            bases);
    } catch (const std::exception& error) {
        // Structured attribution from our own arithmetic, not from the message: the tightest slot
        // is the argmin residual (Automatic) or the argmin budget (Explicit's bottleneck form).
        int limiting              = 0;
        std::size_t limiting_value = std::numeric_limits<std::size_t>::max();
        for (int slot = 0; slot < kPlacementSlots; ++slot) {
            const auto index         = static_cast<std::size_t>(slot);
            const std::size_t budget = budgets[index];
            const std::size_t value  = inputs.kv_policy.mode == KvCapacityMode::Automatic
                                           ? (budget >= bases[index] ? budget - bases[index] : 0)
                                           : budget;
            if (value < limiting_value) {
                limiting_value = value;
                limiting       = slot;
            }
        }
        const auto index    = static_cast<std::size_t>(limiting);
        out.status          = PlacementStatus::Rejected;
        out.limiting_slot   = limiting;
        out.reason_code     = kReasonKvCapacity;
        out.reason_detail   = error.what();
        out.available_bytes = budgets[index];
        // Automatic: the slot's fixed requirement B_i + headroom. Explicit: the curve's minimum
        // reservation (a lower bound; the resolver's message carries the exact explicit-point
        // figure, and re-deriving the explicit page count here would duplicate its arithmetic).
        const std::uint64_t required =
            inputs.kv_policy.mode == KvCapacityMode::Automatic
                ? bases[index] + inputs.kv_policy.automatic_headroom_bytes
                : inputs.curve.minimum_device_reservation_bytes;
        out.required_bytes  = required;
        out.shortfall_bytes = static_cast<std::int64_t>(required) -
                              static_cast<std::int64_t>(out.available_bytes);
        return out;
    }
    out.resolved_tokens      = resolution.resolved_tokens;
    out.resolved_page_groups = resolution.main_page_groups;

    // Stage 3 -- mandatory independent ledger at the RESOLVED point from the true per-slot layout
    // (not the affine curve). This is authoritative: the shared curve's stride is a min->min+1
    // delta of a max-over-slots reservation and is not guaranteed affine for every boundary.
    if (!build_ledgers(inputs, resolution.main_page_groups, out)) { return out; }
    for (auto& ledger : out.ledgers) {
        ledger.kv_mode              = out.kv_mode;
        ledger.resolved_tokens      = out.resolved_tokens;
        ledger.resolved_page_groups = out.resolved_page_groups;
    }
    attribute_from_ledgers(out);
    return out;
}

void enumerate_placements(const PlacementEnumerationSource& source, PlacementReport& report) {
    report.candidates.clear();
    report.feasible_count = 0;
    report.rejected_count = 0;
    report.layer_count    = source.layer_count;
    for (int boundary = 1; boundary < source.layer_count; ++boundary) {
        bool legal = false;
        try {
            legal = source.boundary_is_legal && source.boundary_is_legal(boundary);
        } catch (const std::exception&) { legal = false; }
        if (!legal) { continue; }
        PlacementLayerCounts counts;
        if (source.layer_counts) {
            try { counts = source.layer_counts(boundary); } catch (const std::exception&) {}
        }
        for (int endpoint = 0; endpoint < kPlacementSlots; ++endpoint) {
            const PlacementSpec spec{.boundary = boundary, .endpoint_owner_slot = endpoint};
            PlacementCandidateResult result;
            result.owner0_layers                = boundary;
            result.owner1_layers                = source.layer_count - boundary;
            result.owner0_full_attention_layers = counts.owner0_full_attention;
            result.owner0_gdn_layers            = counts.owner0_gdn;
            result.owner1_full_attention_layers = counts.owner1_full_attention;
            result.owner1_gdn_layers            = counts.owner1_gdn;
            try {
                result.feasibility = evaluate_placement(source.make_inputs(spec));
            } catch (const std::exception& error) {
                result.feasibility               = PlacementFeasibility{};
                result.feasibility.spec          = spec;
                result.feasibility.status        = PlacementStatus::Rejected;
                result.feasibility.reason_code   = kReasonPlannerError;
                result.feasibility.reason_detail = error.what();
            }
            if (result.feasibility.status == PlacementStatus::Feasible) {
                ++report.feasible_count;
            } else {
                ++report.rejected_count;
            }
            report.candidates.push_back(std::move(result));
        }
    }
}

std::string render_placement_report(const PlacementReport& report) {
    std::ostringstream out;
    out << "PLACEMENT ANALYSIS (hard feasibility only; no ranking, no selection)\n";
    out << "context=" << report.max_context << " kv_mode="
        << (report.kv_policy.mode == KvCapacityMode::Automatic ? "auto" : "explicit");
    if (report.kv_policy.mode == KvCapacityMode::Automatic) {
        out << " auto_headroom=" << report.kv_policy.automatic_headroom_bytes;
    } else {
        out << " kv_tokens=" << report.kv_policy.explicit_tokens;
    }
    out << " kv_dtype=" << report.kv_dtype << " max_concurrency=" << report.max_concurrency
        << " layers=" << report.layer_count << "\n";
    out << "capacity_basis=pre-materialization cudaMemGetInfo free, captured once for this report\n";
    for (int slot = 0; slot < kPlacementSlots; ++slot) {
        const auto index = static_cast<std::size_t>(slot);
        out << "capacity_snapshot_bytes[slot" << slot << " gpu" << report.devices[index]
            << "]=" << bytes_str(report.capacity_snapshot_bytes[index]) << "\n";
    }
    out << "allocator_reserve=" << bytes_str(report.allocator_reserve_bytes) << "\n";
    out << "candidates=" << report.candidates.size() << " feasible=" << report.feasible_count
        << " rejected=" << report.rejected_count << "\n";
    for (const PlacementCandidateResult& candidate : report.candidates) {
        const PlacementFeasibility& f = candidate.feasibility;
        out << (f.status == PlacementStatus::Feasible ? "PASS " : "REJECT ")
            << candidate.owner0_layers << "/" << candidate.owner1_layers << " E"
            << f.spec.endpoint_owner_slot;
        out << "  full_attn=" << candidate.owner0_full_attention_layers << "/"
            << candidate.owner1_full_attention_layers << " gdn=" << candidate.owner0_gdn_layers
            << "/" << candidate.owner1_gdn_layers << "\n";
        if (!f.kv_mode.empty() && f.resolved_page_groups != 0) {
            out << "  kv=" << f.kv_mode << " resolved_tokens=" << f.resolved_tokens
                << " resolved_page_groups=" << f.resolved_page_groups << "\n";
        }
        if (f.status == PlacementStatus::Rejected) {
            out << "  limiting_slot=" << f.limiting_slot << " reason=" << f.reason_code << "\n";
            out << "  required=" << bytes_str(f.required_bytes)
                << " available=" << bytes_str(f.available_bytes)
                << " shortfall=" << f.shortfall_bytes << "\n";
            if (!f.reason_detail.empty()) { out << "  detail=" << f.reason_detail << "\n"; }
        }
        if (f.ledgers_valid) {
            for (int slot = 0; slot < kPlacementSlots; ++slot) {
                const DeviceAdmissionPlan& l = f.ledgers[static_cast<std::size_t>(slot)];
                out << "  slot" << slot << " gpu" << l.device << " weights=" << l.planned_weight_bytes
                    << " persistent=" << l.exact_persistent_bytes
                    << " kv_payload(report_only)=" << l.persistent_kv_payload_bytes
                    << " workspace=" << l.workspace_bytes
                    << " request_transient=" << l.request_transient_bytes
                    << " graph=" << l.graph_bytes << " reserve=" << l.allocator_driver_reserve
                    << " total_required=" << l.total_required_bytes
                    << " capacity=" << l.effective_capacity
                    << " headroom=" << l.predicted_headroom
                    << " decision=" << (l.decision == DeviceAdmissionDecision::Admit ? "ADMIT" : "REJECT")
                    << " reason=" << l.reason_code << "\n";
            }
        }
    }
    return out.str();
}

} // namespace ninfer::targets

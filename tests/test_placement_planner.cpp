// M5.1/M5.2 host-only test: the exact placement feasibility engine on SYNTHETIC layouts (no
// artifact, no device). Every case fixes a snapshot/weights/base configuration and asserts the
// structured verdict: status, limiting slot, reason code, resolved page groups, headroom.
#include "targets/placement_planner.h"

#include "runtime/engine/kv_capacity.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using ninfer::KvCapacityPolicy;
using ninfer::targets::DeviceAdmissionDecision;
using ninfer::targets::evaluate_placement;
using ninfer::targets::kPlacementSlots;
using ninfer::targets::PlacementFeasibility;
using ninfer::targets::PlacementFeasibilityInputs;
using ninfer::targets::PlacementSpec;
using ninfer::targets::PlacementStatus;
using ninfer::targets::qwen3_6::DeviceLayoutPreview;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Synthetic per-slot layout: persistent = base + (groups - min) * stride[slot]; workspace and
// graph fixed; request transient 0 (layer split forces --no-cuda-graph, so graph is 0 live too,
// but keep a nonzero workspace so the composition is exercised).
struct SyntheticLayout {
    std::uint32_t minimum_groups = 128;
    std::uint32_t maximum_groups = 128;
    std::size_t persistent_base[kPlacementSlots] = {0, 0};
    std::size_t stride[kPlacementSlots]          = {0, 0};
    std::size_t workspace                        = 1000;
    bool available                               = true;

    [[nodiscard]] DeviceLayoutPreview operator()(int slot, std::optional<std::uint32_t> groups) const {
        if (!available) { return DeviceLayoutPreview{}; }
        const std::uint32_t at = groups.value_or(minimum_groups);
        return DeviceLayoutPreview{
            .available                        = true,
            .persistent_bytes                 = persistent_base[slot] + (at - minimum_groups) * stride[slot],
            .persistent_kv_payload_bytes      = persistent_base[slot] / 2,
            .workspace_capacity_bytes         = workspace,
            .request_transient_capacity_bytes = 0,
            .graph_allowance_bytes            = 0,
        };
    }

    // Curve exactly as make_sequence_planner_impl derives it: base = max over slots of the
    // minimum-point reservation, stride = the max-over-slots delta at min -> min+1.
    [[nodiscard]] ninfer::runtime::SequenceCapacityCurve curve() const {
        const std::size_t base0 = persistent_base[0] + workspace;
        const std::size_t base1 = persistent_base[1] + workspace;
        const std::size_t min_reservation = std::max(base0, base1);
        std::size_t curve_stride = 0;
        if (minimum_groups < maximum_groups) {
            curve_stride = std::max(base0 + stride[0], base1 + stride[1]) - min_reservation;
        }
        return ninfer::runtime::SequenceCapacityCurve{
            .main_page_tokens                     = 64,
            .minimum_main_page_groups             = minimum_groups,
            .maximum_main_page_groups             = maximum_groups,
            .minimum_device_reservation_bytes     = min_reservation,
            .bytes_per_additional_main_page_group = curve_stride,
        };
    }
};

PlacementFeasibilityInputs make_inputs(const SyntheticLayout& layout, std::uint64_t w0, std::uint64_t w1,
                                       std::uint64_t snap0, std::uint64_t snap1, std::uint64_t reserve,
                                       KvCapacityPolicy policy) {
    PlacementFeasibilityInputs inputs;
    inputs.spec                   = PlacementSpec{.boundary = 57, .endpoint_owner_slot = 0};
    inputs.devices[0]             = {.device = 0, .planned_weight_bytes = w0, .capacity_snapshot_bytes = snap0};
    inputs.devices[1]             = {.device = 1, .planned_weight_bytes = w1, .capacity_snapshot_bytes = snap1};
    inputs.allocator_reserve_bytes = reserve;
    inputs.kv_policy              = policy;
    inputs.curve                  = layout.curve();
    inputs.preview                = layout;
    return inputs;
}

} // namespace

int main() {
    const KvCapacityPolicy explicit_min = KvCapacityPolicy::explicit_capacity(128 * 64);
    const KvCapacityPolicy auto_no_headroom = KvCapacityPolicy::automatic(0);

    // A. WEIGHT_CAPACITY: slot 0 cannot hold its planned weights.
    {
        SyntheticLayout layout;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        const auto r = evaluate_placement(make_inputs(layout, 20000, 1000, 15000, 100000, 100, explicit_min));
        check(r.status == PlacementStatus::Rejected, "A: status");
        check(r.limiting_slot == 0, "A: limiting slot 0");
        check(r.reason_code == "WEIGHT_CAPACITY", "A: reason " + r.reason_code);
        check(r.shortfall_bytes == 20000 + 5000 + 1000 + 100 - 15000, "A: shortfall is the full ledger gap");
        check(r.ledgers_valid && r.ledgers[1].decision == DeviceAdmissionDecision::Admit, "A: slot 1 admits");
    }

    // B. PERSISTENT_CAPACITY: weights fit, resolved runtime state does not.
    {
        SyntheticLayout layout;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        // 10000 + 5000 > 14000 but 10000 <= 14000.
        const auto r = evaluate_placement(make_inputs(layout, 10000, 1000, 14000, 100000, 100, explicit_min));
        check(r.status == PlacementStatus::Rejected && r.limiting_slot == 0, "B: rejected on slot 0");
        check(r.reason_code == "PERSISTENT_CAPACITY", "B: reason " + r.reason_code);
        check(r.shortfall_bytes == 10000 + 5000 + 1000 + 100 - 14000, "B: shortfall");
    }

    // C. WORKSPACE / RESERVE BOUNDARY: Automatic growth stops exactly at F = snapshot - W - R.
    //    With stride 10 and 5 whole strides of slack beyond B + R, plus 9 bytes: exactly 5 extra
    //    groups. If the reserve were NOT protected the same slack would admit one more group
    //    (reserve 100 = 10 strides), and if workspace were not in the base the count would be
    //    100 higher still.
    {
        SyntheticLayout layout;
        layout.maximum_groups     = 1024;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        layout.stride[0]          = 10;
        layout.stride[1]          = 10;
        const std::uint64_t w0 = 10000, reserve = 100;
        const std::uint64_t snap0 = w0 + (5000 + 1000) + reserve + 5 * 10 + 9;
        const auto r = evaluate_placement(make_inputs(layout, w0, 1000, snap0, 1000000, reserve, auto_no_headroom));
        check(r.status == PlacementStatus::Feasible, "C: feasible " + r.reason_code + " " + r.reason_detail);
        check(r.resolved_page_groups == 128 + 5, "C: resolved groups " + std::to_string(r.resolved_page_groups));
        check(r.ledgers[0].predicted_headroom == 9, "C: slot 0 headroom is the sub-stride remainder");
        check(r.ledgers[0].allocator_driver_reserve == reserve, "C: reserve charged once in the ledger");
        check(r.ledgers[0].total_required_bytes ==
                  w0 + (5000 + 5 * 10) + 1000 + reserve,
              "C: ledger total = W + persistent(resolved) + workspace + reserve");
        // Same snapshot, reserve 0: the growth reclaims exactly those 10 strides.
        const auto r0 = evaluate_placement(make_inputs(layout, w0, 1000, snap0, 1000000, 0, auto_no_headroom));
        check(r0.resolved_page_groups == 128 + 15, "C: without reserve growth would have been 15");
    }

    // D. DEVICE ASSOCIATION: slot 1 is the tight one, slot 0 comfortable.
    {
        SyntheticLayout layout;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        const auto r = evaluate_placement(make_inputs(layout, 10000, 8000, 1000000, 8000 + 3000 + 1000 + 100 - 1, 100, explicit_min));
        check(r.status == PlacementStatus::Rejected && r.limiting_slot == 1, "D: limiting slot 1");
        check(r.reason_code == "ALLOCATOR_RESERVE", "D: reason " + r.reason_code);
        check(r.shortfall_bytes == 1, "D: shortfall 1");
        check(r.ledgers[0].decision == DeviceAdmissionDecision::Admit, "D: slot 0 admits");
        check(r.ledgers[1].device == 1, "D: ledger carries device ordinal");
    }

    // E. M2.2b EXACT AUTOMATIC REUSE: F=[1400,2000] (after W and R), B=[200,1000] (incl workspace).
    //    Conservative min(F)-max(B) = 400 -> 4 extra groups at stride 100; exact min_i(F_i-B_i) =
    //    min(1200, 1000) = 1000 -> 10 extra groups. The planner must follow the four-arg result.
    {
        SyntheticLayout layout;
        layout.maximum_groups     = 1024;
        layout.workspace          = 100;
        layout.persistent_base[0] = 100;  // B0 = 200
        layout.persistent_base[1] = 900;  // B1 = 1000
        layout.stride[0]          = 100;
        layout.stride[1]          = 100;
        const std::uint64_t reserve = 50;
        const std::uint64_t w0 = 5000, w1 = 7000;
        const auto inputs = make_inputs(layout, w0, w1, w0 + reserve + 1400, w1 + reserve + 2000, reserve, auto_no_headroom);
        const auto r      = evaluate_placement(inputs);
        check(r.status == PlacementStatus::Feasible, "E: feasible " + r.reason_code);
        const std::size_t F[] = {1400, 2000};
        const std::size_t B[] = {200, 1000};
        const auto exact = ninfer::runtime::resolve_kv_capacity_symmetric(auto_no_headroom, inputs.curve, F, B);
        const auto conservative = ninfer::runtime::resolve_kv_capacity_symmetric(auto_no_headroom, inputs.curve, F);
        check(r.resolved_page_groups == exact.main_page_groups, "E: matches four-arg resolver");
        check(exact.main_page_groups == 128 + 10 && conservative.main_page_groups == 128 + 4,
              "E: exact 10 extra groups vs conservative 4");
        check(r.ledgers[0].predicted_headroom == 200 && r.ledgers[1].predicted_headroom == 0,
              "E: per-slot headroom after exact growth");
        // Explicit at the minimum point through the same engine: unchanged conservative path.
        const auto e = evaluate_placement(make_inputs(layout, w0, w1, w0 + reserve + 1400, w1 + reserve + 2000, reserve, explicit_min));
        check(e.status == PlacementStatus::Feasible && e.resolved_page_groups == 128, "E: explicit minimum point");
        check(e.kv_mode == "explicit" && r.kv_mode == "auto", "E: kv_mode strings");
    }

    // E2. Automatic per-device headroom rejection names the device via our own arithmetic.
    {
        SyntheticLayout layout;
        layout.maximum_groups     = 1024;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        layout.stride[0] = layout.stride[1] = 10;
        const KvCapacityPolicy headroom = KvCapacityPolicy::automatic(500);
        // Slot 1: F1 = 3000 + 1000 + 499 -> short by 1 of B1 + headroom. Slot 0 comfortable.
        const auto r = evaluate_placement(make_inputs(layout, 10000, 8000, 1000000, 8000 + 100 + 3000 + 1000 + 499, 100, headroom));
        check(r.status == PlacementStatus::Rejected && r.reason_code == "KV_CAPACITY", "E2: KV_CAPACITY " + r.reason_code);
        check(r.limiting_slot == 1, "E2: limiting slot 1");
        check(r.required_bytes == 4000 + 500 && r.available_bytes == 4499 && r.shortfall_bytes == 1, "E2: figures");
        check(!r.reason_detail.empty(), "E2: resolver message preserved as detail");
    }

    // F. Ledger authority (Opus R3): slot 1's true stride exceeds the shared curve's stride at the
    //    resolved point, so the resolver admits but the resolved-point ledger must reject, naming
    //    slot 1 -- the planner fails closed, never over-admits.
    {
        SyntheticLayout layout;
        layout.maximum_groups     = 1024;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        layout.stride[0]          = 10;
        layout.stride[1]          = 10;
        // Curve computed with equal strides, then slot 1's true layout grows faster.
        auto inputs = make_inputs(layout, 10000, 8000, 10000 + 100 + 6000 + 200, 8000 + 100 + 4000 + 200, 100, auto_no_headroom);
        SyntheticLayout steep = layout;
        steep.stride[1] = 30;
        inputs.preview  = steep;
        const auto r = evaluate_placement(inputs);
        check(r.status == PlacementStatus::Rejected, "F: ledger rejects what the curve admitted");
        check(r.limiting_slot == 1, "F: slot 1 limits");
        check(r.reason_code == "PERSISTENT_CAPACITY" || r.reason_code == "TOTAL_DEVICE_CAPACITY" ||
                  r.reason_code == "ALLOCATOR_RESERVE" || r.reason_code == "WORKSPACE_CAPACITY",
              "F: byte-category attribution " + r.reason_code);
        check(r.resolved_page_groups == 128 + 20, "F: resolver had grown 20 groups");
        check(r.ledgers[1].exact_persistent_bytes == 3000 + 20 * 30, "F: ledger used the true layout");
    }

    // G. Preview unavailable fails closed.
    {
        SyntheticLayout layout;
        layout.available = false;
        const auto r = evaluate_placement(make_inputs(layout, 1, 1, 100, 100, 1, explicit_min));
        check(r.status == PlacementStatus::Rejected && r.reason_code == "LAYOUT_PREVIEW_UNAVAILABLE" && !r.ledgers_valid,
              "G: unavailable preview");
    }

    // H. Enumerator: per-candidate exception isolation, ordering, counts, rendering vocabulary.
    {
        using ninfer::targets::PlacementEnumerationSource;
        using ninfer::targets::PlacementReport;
        SyntheticLayout layout;
        layout.persistent_base[0] = 5000;
        layout.persistent_base[1] = 3000;
        PlacementEnumerationSource source;
        source.layer_count       = 8;
        source.boundary_is_legal = [](int b) { return b >= 1 && b <= 7 && b != 3; };
        source.make_inputs       = [&](PlacementSpec spec) {
            if (spec.boundary == 5 && spec.endpoint_owner_slot == 1) { throw std::runtime_error("synthetic planner fault"); }
            auto in = make_inputs(layout, 10000, 1000, spec.boundary == 6 ? 1 : 100000, 100000, 100, explicit_min);
            in.spec = spec;
            return in;
        };
        source.layer_counts = [](int b) { return ninfer::targets::PlacementLayerCounts{.owner0_full_attention = b}; };
        PlacementReport report;
        report.max_context = 8192;
        report.kv_policy   = explicit_min;
        report.kv_dtype    = "int8";
        ninfer::targets::enumerate_placements(source, report);
        check(report.candidates.size() == 12, "H: 6 legal boundaries x 2 endpoints");
        check(report.candidates[0].feasibility.spec.boundary == 1 && report.candidates[0].feasibility.spec.endpoint_owner_slot == 0 &&
                  report.candidates[1].feasibility.spec.endpoint_owner_slot == 1 && report.candidates[2].feasibility.spec.boundary == 2,
              "H: deterministic order");
        std::size_t faults = 0, weight_rejects = 0;
        for (const auto& c : report.candidates) {
            if (c.feasibility.reason_code == "PLANNER_ERROR") { ++faults; check(c.feasibility.spec.boundary == 5 && c.feasibility.spec.endpoint_owner_slot == 1, "H: fault attributed to 5/E1"); }
            if (c.feasibility.reason_code == "WEIGHT_CAPACITY") { ++weight_rejects; }
        }
        check(faults == 1 && weight_rejects == 2, "H: one fault, two weight rejects (6/E0, 6/E1)");
        check(report.feasible_count == 9 && report.rejected_count == 3, "H: counts");
        const std::string text = ninfer::targets::render_placement_report(report);
        check(text.find("PLACEMENT ANALYSIS") == 0, "H: header");
        check(text.find("REJECT 6/2 E0") != std::string::npos && text.find("PASS 1/7 E1") != std::string::npos, "H: candidate lines");
        check(text.find("PLANNER_ERROR") != std::string::npos && text.find("synthetic planner fault") != std::string::npos, "H: fault reported");
        for (const char* banned : {"SELECT", "BEST", "OPTIMAL", "PREFERRED"}) {
            check(text.find(banned) == std::string::npos, std::string("H: banned word ") + banned);
        }
        check(text.find("capacity_snapshot_bytes[slot0") != std::string::npos, "H: snapshot basis printed");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

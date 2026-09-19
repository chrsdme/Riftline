#pragma once

// M5.0: pure structural placement of a layer-split model across two device slots. This is the
// ONE representation both the real loader (materialization planning + runtime view construction)
// and the analytical placement planner consume, so ownership cannot drift between them.
//
// A PlacementSpec is a CHOICE, not a requirement: request/runtime properties (context, KV policy,
// concurrency, split mode, ...) stay in EngineOptions. Candidate evaluation constructs specs
// explicitly and never reads or writes the process environment; only the manual runtime path
// resolves NINFER_ENDPOINT_DEVICE, exactly once, into a spec (see manual_endpoint_owner_slot).

namespace ninfer::targets {

struct PlacementSpec {
    // Contiguous transformer-layer boundary: slot 0 owns layers [0, boundary), slot 1 owns
    // [boundary, layer_count). Meaningful only under SplitMode::Layer.
    int boundary = 0;
    // Owner of the output endpoint group (final norm + output head): 0 or 1. The token embedding
    // always stays on slot 0 (see qwen3_6_27b bindings.cpp for why).
    int endpoint_owner_slot = 0;

    friend bool operator==(const PlacementSpec&, const PlacementSpec&) = default;
};

// The manual runtime's endpoint selection: NINFER_ENDPOINT_DEVICE read ONCE per process (cached),
// exactly "1" selects slot 1, anything else (unset, "0", garbage) is slot 0 -- the pre-M4 default.
// Shared by the qwen3_6_27b loader (M4a) and the admission printer (registry) so the two parsers
// cannot disagree. Planner candidates never call this.
[[nodiscard]] int manual_endpoint_owner_slot();

} // namespace ninfer::targets

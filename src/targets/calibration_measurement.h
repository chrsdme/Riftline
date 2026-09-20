#pragma once

// M5.4b sec 17-20: record emitter adapter, payload geometry derivation, transfer harness
// plumbing, compute bench adapter foundation. Host-only, CUDA-free (sec 20/19 tests are
// compile/shape-expansion only -- nothing here executes a kernel or a transfer).

#include "targets/calibration.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::calibration_measurement {

// ---------------------------------------------------------------------------------------------
// sec 17: record emitter adapter
// ---------------------------------------------------------------------------------------------

// One timing observation plus everything the M5.4a schema needs to describe it. Deliberately a
// flat aggregate, not a builder -- every field a caller must decide is visible at the call site.
struct ObservationInput {
    ninfer::targets::calibration::CalibrationIdentity identity;
    ninfer::targets::calibration::RecordKind kind = ninfer::targets::calibration::RecordKind::Operation;
    ninfer::targets::calibration::OperationIdentity operation;
    ninfer::targets::calibration::TransferIdentity transfer;
    ninfer::targets::calibration::GdnIdentity gdn;
    std::string run_id;
    std::string observation_id;
    std::uint64_t sample_index = 0;
    double value = 0.0;
    std::string unit;
    std::optional<std::uint64_t> timestamp_unix_ms;
    ninfer::targets::calibration::EvidenceClass evidence =
        ninfer::targets::calibration::EvidenceClass::Unknown;
    // Caller-requested validity. NOT taken at face value -- see emit_record(): this adapter
    // downgrades to Unqualified whenever missing_requirements() is non-empty, so a caller cannot
    // fabricate Valid merely by setting this field (sec 17/22).
    ninfer::targets::calibration::ValidityState requested_validity =
        ninfer::targets::calibration::ValidityState::Unqualified;
    ninfer::targets::calibration::CorrectnessStatus correctness =
        ninfer::targets::calibration::CorrectnessStatus::NotRun;
};

struct EmittedRecord {
    ninfer::targets::calibration::CalibrationRecord record;
    std::vector<std::string> missing_requirements; // empty iff qualification requirements are met.
    bool is_qualified_evidence = false;             // is_qualified_calibration_evidence(record).
};

// Assembles a CalibrationRecord from `input` and runs it through the existing M5.4a qualification
// logic (missing_requirements / is_qualified_calibration_evidence) -- never rewritten here. The
// record's ValidityState is `input.requested_validity` ONLY IF missing_requirements() on the
// assembled record is empty; otherwise it is forced to Unqualified regardless of what the caller
// asked for, so a caller can never fabricate Valid on an incomplete record through this adapter.
[[nodiscard]] EmittedRecord emit_record(const ObservationInput& input);

// ---------------------------------------------------------------------------------------------
// sec 18: current target shape / payload geometry derivation
// ---------------------------------------------------------------------------------------------

// bytes-per-column for a BF16 buffer with `columns` rows of width `columns_width` elements, i.e.
// columns_width * 2 bytes (BF16 = 2 bytes/element). All 64-bit math -- sec 18 explicitly requires
// no integer overflow when scaling; std::uint64_t leaves ample headroom versus the current
// hidden=5120 / vocab=248320 scale. Throws std::overflow_error (rather than wrapping) if
// `columns_width * 2` would overflow std::uint64_t -- sec 11: unchecked multiplication is a
// fail-open defect, not an acceptable "practically never happens" corner.
[[nodiscard]] std::uint64_t bf16_bytes_per_column(std::uint64_t columns_width);

// One payload class's derived geometry: how many BF16 elements wide one column is, and the
// resulting bytes/column. `total_bytes(num_columns)` scales bytes exactly (bytes_per_column *
// num_columns, 64-bit) -- sec 18: "changing columns scales payload bytes exactly." Throws
// std::overflow_error if that product would overflow std::uint64_t (sec 11).
struct PayloadGeometry {
    ninfer::targets::calibration::TransferPayloadClass payload_class =
        ninfer::targets::calibration::TransferPayloadClass::Unknown;
    std::uint64_t columns_width = 0; // elements per column (e.g. hidden_size, or vocab_size).
    std::uint64_t bytes_per_column = 0;

    [[nodiscard]] std::uint64_t total_bytes(std::uint64_t num_columns) const {
        if (num_columns != 0 &&
            bytes_per_column > std::numeric_limits<std::uint64_t>::max() / num_columns) {
            throw std::overflow_error("PayloadGeometry::total_bytes overflowed std::uint64_t");
        }
        return bytes_per_column * num_columns; // checked above; caller controls num_columns.
    }
};

// Derives geometry for the four current transfer payload classes (sec 18) from the Qwen3.8-27B
// TextConfig runtime constants (src/targets/qwen3_6_27b/impl/config.h) -- hidden/output_rows are
// the implementation authority, never re-hard-coded literals. LayerBoundaryHidden, EndpointHidden,
// and NormalizedHidden all carry one hidden-width BF16 column (they differ in WHERE in the
// pipeline the transfer occurs, not in per-column byte count); LogitsResult carries one
// vocab-width BF16 column.
[[nodiscard]] PayloadGeometry current_layer_boundary_hidden_geometry();
[[nodiscard]] PayloadGeometry current_endpoint_hidden_geometry();
[[nodiscard]] PayloadGeometry current_normalized_hidden_geometry();
[[nodiscard]] PayloadGeometry current_logits_result_geometry();

// ---------------------------------------------------------------------------------------------
// sec 19: transfer harness foundation (PLUMBING ONLY -- no live transfer executed here)
// ---------------------------------------------------------------------------------------------

// What the caller intends to run (device pair + payload class); distinct from TransferIdentity,
// which describes what ACTUALLY happened. A caller builds one of these, runs the actual transfer
// through its own CUDA code, observes the real API/sync/route/overlap/host-involvement, and only
// THEN constructs a TransferIdentity to hand to emit_record -- this module never guesses those
// fields on the caller's behalf.
struct TransferIntent {
    int source_logical_slot = -1;
    int destination_logical_slot = -1;
    ninfer::targets::calibration::TransferPayloadClass payload_class =
        ninfer::targets::calibration::TransferPayloadClass::Unknown;
};

// Builds a TransferIdentity skeleton for `intent` with every route/sync/overlap/host-involvement
// field left Unknown and both device UUIDs left empty -- the caller MUST fill in source_device_uuid
// / destination_device_uuid (from the identity snapshot's devices, keyed by
// source_logical_slot/destination_logical_slot) and every route field (from what the actual
// transfer call reported) before this can qualify. `direction` is the one field this function DOES
// set (to DeviceToDevice, whenever both logical slots are specified and differ -- a structural fact
// about the intent, not an assumption about how the transfer executed) -- everything about HOW the
// transfer happened (P2P, overlap, sync/async, route) stays Unknown; sec 19's "do NOT assume" rules
// are enforced by simply never populating those fields here.
[[nodiscard]] ninfer::targets::calibration::TransferIdentity transfer_identity_skeleton(
    const TransferIntent& intent);

// ---------------------------------------------------------------------------------------------
// sec 20: compute bench adapter foundation (structure only -- does not execute any bench)
// ---------------------------------------------------------------------------------------------

enum class BenchFamily : std::uint8_t {
    GdnDecode,
    GdnChunkedPrefill,
    FullAttentionDecode,
    FullAttentionPrefill,
    Projection,
    Mlp,
    FinalNorm,
    LmHead,
};

// One shape/identity expansion request for a future M5.4c bench-adapter call. `shape` is the
// runtime shape to measure at; `execution_mode`/`warm_state` are explicit first-class measurement
// keys per Round-2 deltas (sec 21) -- callers must set both, never leave them Unknown, when they
// intend to actually emit a measurement (enforced by emit_record's missing_requirements check, not
// by this struct).
struct BenchAdapterRequest {
    BenchFamily family = BenchFamily::Projection;
    ninfer::targets::calibration::ShapeDims shape;
    int device_logical_slot = -1;
    ninfer::targets::calibration::ExecutionMode execution_mode =
        ninfer::targets::calibration::ExecutionMode::Unknown;
    ninfer::targets::calibration::WarmState warm_state =
        ninfer::targets::calibration::WarmState::Unknown;
    // Projection/Mlp/FinalNorm/LmHead run during both prefill and decode passes with no dedicated
    // ExecutionPhase of their own (GdnDecode/GdnChunkedPrefill/FullAttentionDecode/
    // FullAttentionPrefill instead derive their phase from `family` -- this field is ignored for
    // those four). The caller must say which pass invoked the op; this adapter does not guess.
    // Left Unknown here means missing_requirements() will correctly reject the resulting record
    // as unqualified, which is the honest outcome for a caller that didn't say.
    ninfer::targets::calibration::ExecutionPhase phase =
        ninfer::targets::calibration::ExecutionPhase::Unknown;
};

// Expands `request` into the OperationIdentity (or GdnIdentity, for the two GDN families) that the
// eventual bench call must tag its measurement with -- family/phase/route population only, no
// execution. Kept separate per family: GDN families populate a GdnIdentity with the matching
// GdnStage; every other family populates an OperationIdentity with the matching OperationFamily/
// ExecutionPhase.
struct BenchOperationExpansion {
    ninfer::targets::calibration::OperationIdentity operation;
    bool is_gdn = false;
    ninfer::targets::calibration::GdnIdentity gdn; // meaningful only when is_gdn == true.
};

[[nodiscard]] BenchOperationExpansion expand_bench_request(const BenchAdapterRequest& request);

} // namespace ninfer::targets::calibration_measurement

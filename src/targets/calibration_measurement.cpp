#include "targets/calibration_measurement.h"

#include "targets/qwen3_6_27b/impl/config.h"

namespace ninfer::targets::calibration_measurement {

EmittedRecord emit_record(const ObservationInput& input) {
    EmittedRecord result;
    result.record.kind = input.kind;
    result.record.identity = input.identity;
    result.record.operation = input.operation;
    result.record.transfer = input.transfer;
    result.record.gdn = input.gdn;
    result.record.sample.run_id = input.run_id;
    result.record.sample.observation_id = input.observation_id;
    result.record.sample.sample_index = input.sample_index;
    result.record.sample.value = input.value;
    result.record.sample.unit = input.unit;
    result.record.sample.timestamp_unix_ms = input.timestamp_unix_ms;
    result.record.sample.qualification = input.correctness;
    result.record.qualification.evidence = input.evidence;
    result.record.qualification.correctness = input.correctness;

    result.missing_requirements = ninfer::targets::calibration::missing_requirements(result.record);
    // Never fabricate Valid: only honor the caller's requested_validity when the assembled record
    // has no missing requirements. Otherwise force Unqualified, regardless of what was requested.
    result.record.qualification.validity = result.missing_requirements.empty()
                                                ? input.requested_validity
                                                : ninfer::targets::calibration::ValidityState::Unqualified;

    result.is_qualified_evidence =
        ninfer::targets::calibration::is_qualified_calibration_evidence(result.record);
    return result;
}

std::uint64_t bf16_bytes_per_column(std::uint64_t columns_width) {
    if (columns_width > std::numeric_limits<std::uint64_t>::max() / 2ULL) {
        throw std::overflow_error("bf16_bytes_per_column overflowed std::uint64_t");
    }
    return columns_width * 2ULL; // BF16 = 2 bytes/element, checked above.
}

namespace {

PayloadGeometry hidden_geometry(ninfer::targets::calibration::TransferPayloadClass payload_class) {
    PayloadGeometry geometry;
    geometry.payload_class = payload_class;
    geometry.columns_width =
        static_cast<std::uint64_t>(ninfer::targets::qwen3_6_27b::detail::TextConfig::hidden);
    geometry.bytes_per_column = bf16_bytes_per_column(geometry.columns_width);
    return geometry;
}

} // namespace

PayloadGeometry current_layer_boundary_hidden_geometry() {
    return hidden_geometry(ninfer::targets::calibration::TransferPayloadClass::LayerBoundaryHidden);
}
PayloadGeometry current_endpoint_hidden_geometry() {
    return hidden_geometry(ninfer::targets::calibration::TransferPayloadClass::EndpointHidden);
}
PayloadGeometry current_normalized_hidden_geometry() {
    return hidden_geometry(ninfer::targets::calibration::TransferPayloadClass::NormalizedHidden);
}

PayloadGeometry current_logits_result_geometry() {
    PayloadGeometry geometry;
    geometry.payload_class = ninfer::targets::calibration::TransferPayloadClass::LogitsResult;
    geometry.columns_width =
        static_cast<std::uint64_t>(ninfer::targets::qwen3_6_27b::detail::TextConfig::output_rows);
    geometry.bytes_per_column = bf16_bytes_per_column(geometry.columns_width);
    return geometry;
}

ninfer::targets::calibration::TransferIdentity transfer_identity_skeleton(
    const TransferIntent& intent) {
    ninfer::targets::calibration::TransferIdentity transfer;
    transfer.payload_class = intent.payload_class;
    if (intent.source_logical_slot >= 0 && intent.destination_logical_slot >= 0 &&
        intent.source_logical_slot != intent.destination_logical_slot) {
        transfer.direction = ninfer::targets::calibration::TransferDirection::DeviceToDevice;
    }
    // route/sync/host_involvement/overlap/device UUIDs deliberately left at their default
    // Unknown/empty -- populated only by the caller after observing the actual transfer.
    return transfer;
}

BenchOperationExpansion expand_bench_request(const BenchAdapterRequest& request) {
    using ninfer::targets::calibration::ExecutionPhase;
    using ninfer::targets::calibration::GdnStage;
    using ninfer::targets::calibration::OperationFamily;

    BenchOperationExpansion expansion;

    switch (request.family) {
    case BenchFamily::GdnDecode:
        expansion.is_gdn = true;
        expansion.gdn.stage = GdnStage::RecurrentDecode;
        expansion.gdn.shape = request.shape;
        expansion.operation.family = OperationFamily::Gdn;
        expansion.operation.phase = ExecutionPhase::Decode;
        break;
    case BenchFamily::GdnChunkedPrefill:
        expansion.is_gdn = true;
        expansion.gdn.stage = GdnStage::ChunkedPrefill;
        expansion.gdn.shape = request.shape;
        expansion.operation.family = OperationFamily::Gdn;
        expansion.operation.phase = ExecutionPhase::Prefill;
        break;
    case BenchFamily::FullAttentionDecode:
        expansion.operation.family = OperationFamily::Attention;
        expansion.operation.phase = ExecutionPhase::Decode;
        break;
    case BenchFamily::FullAttentionPrefill:
        expansion.operation.family = OperationFamily::Attention;
        expansion.operation.phase = ExecutionPhase::Prefill;
        break;
    case BenchFamily::Projection:
        expansion.operation.family = OperationFamily::Projection;
        expansion.operation.phase = request.phase; // caller says which pass invoked it.
        break;
    case BenchFamily::Mlp:
        expansion.operation.family = OperationFamily::Mlp;
        expansion.operation.phase = request.phase;
        break;
    case BenchFamily::FinalNorm:
        expansion.operation.family = OperationFamily::FinalNorm;
        expansion.operation.phase = request.phase;
        break;
    case BenchFamily::LmHead:
        expansion.operation.family = OperationFamily::LmHead;
        expansion.operation.phase = request.phase;
        break;
    }

    expansion.operation.shape = request.shape;
    expansion.operation.device_logical_slot = request.device_logical_slot;
    expansion.operation.execution_mode = request.execution_mode;
    expansion.operation.warm_state = request.warm_state;
    return expansion;
}

} // namespace ninfer::targets::calibration_measurement

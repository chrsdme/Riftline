#include "targets/calibration.h"

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace ninfer::targets::calibration {
namespace {

// ---- name tables (mirrors the hand-rolled switch-to-string style in src/serve/request_log.cpp
//      and bench/targets/qwen3_6_27b/ninfer_bench_support.cpp) -------------------------------

std::string json_escape(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 2);
    for (char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

// Full round-trip precision for doubles: default ostream precision (6 sig figs) is lossy.
std::string number(double value) {
    std::ostringstream out;
    out.precision(17);
    out << value;
    return out.str();
}

} // namespace

namespace {

// Generic table-driven name<->enum lookup. `table` must be a reference to a static array of
// {enum_value, name} pairs; unknown values return "unknown" / value-initialized default.
template <typename Enum, std::size_t N>
std::string_view enum_name(const std::pair<Enum, std::string_view> (&table)[N], Enum value) {
    for (const auto& [enum_value, name] : table) {
        if (enum_value == value) { return name; }
    }
    return "unknown";
}
template <typename Enum, std::size_t N>
Enum enum_from_name(const std::pair<Enum, std::string_view> (&table)[N], std::string_view name) {
    for (const auto& [enum_value, table_name] : table) {
        if (table_name == name) { return enum_value; }
    }
    return Enum{};
}
// Checked lookup for parsing: an unknown token must never silently become Enum{} (which, for
// RecordKind, is Operation -- letting an unrecognized record_kind masquerade as a valid one).
// Tables that list "unknown" as a name accept that token explicitly; tables that don't (RecordKind,
// CorrectnessStatus) reject it, which is the desired "explicit unknown is allowed only where
// declared" behavior.
template <typename Enum, std::size_t N>
std::optional<Enum> enum_from_name_checked(const std::pair<Enum, std::string_view> (&table)[N],
                                            std::string_view name) {
    for (const auto& [enum_value, table_name] : table) {
        if (table_name == name) { return enum_value; }
    }
    return std::nullopt;
}

constexpr std::pair<OperationFamily, std::string_view> kOperationFamilyTable[] = {
    {OperationFamily::Unknown, "unknown"},       {OperationFamily::Gdn, "gdn"},
    {OperationFamily::Attention, "attention"},   {OperationFamily::Projection, "projection"},
    {OperationFamily::Mlp, "mlp"},               {OperationFamily::FinalNorm, "final_norm"},
    {OperationFamily::LmHead, "lm_head"},        {OperationFamily::Transfer, "transfer"},
    {OperationFamily::Endpoint, "endpoint"},
};
constexpr std::pair<ExecutionPhase, std::string_view> kExecutionPhaseTable[] = {
    {ExecutionPhase::Unknown, "unknown"},     {ExecutionPhase::Prefill, "prefill"},
    {ExecutionPhase::Decode, "decode"},       {ExecutionPhase::StateReadUpdate, "state_read_update"},
    {ExecutionPhase::Endpoint, "endpoint"},   {ExecutionPhase::Transfer, "transfer"},
};
constexpr std::pair<WarmState, std::string_view> kWarmStateTable[] = {
    {WarmState::Unknown, "unknown"}, {WarmState::Cold, "cold"}, {WarmState::Warm, "warm"},
    {WarmState::Cached, "cached"},
};
constexpr std::pair<SourceState, std::string_view> kSourceStateTable[] = {
    {SourceState::Unknown, "unknown"}, {SourceState::Clean, "clean"}, {SourceState::Dirty, "dirty"},
};
constexpr std::pair<ExecutionMode, std::string_view> kExecutionModeTable[] = {
    {ExecutionMode::Unknown, "unknown"}, {ExecutionMode::Eager, "eager"},
    {ExecutionMode::Graph, "graph"},
};
constexpr std::pair<TransferSync, std::string_view> kTransferSyncTable[] = {
    {TransferSync::Unknown, "unknown"}, {TransferSync::Sync, "sync"}, {TransferSync::Async, "async"},
};
constexpr std::pair<TransferRoute, std::string_view> kTransferRouteTable[] = {
    {TransferRoute::Unknown, "unknown"}, {TransferRoute::CopyEngine, "copy_engine"},
    {TransferRoute::Kernel, "kernel"},
    {TransferRoute::RuntimeManagedMemcpy, "runtime_managed_memcpy"},
};
constexpr std::pair<TransferRoute, std::string_view> kTransferRouteV1Table[] = {
    {TransferRoute::Unknown, "unknown"}, {TransferRoute::CopyEngine, "copy_engine"},
    {TransferRoute::Kernel, "kernel"},
};
constexpr std::pair<TransferOverlap, std::string_view> kTransferOverlapTable[] = {
    {TransferOverlap::Unknown, "unknown"}, {TransferOverlap::NoOverlap, "no_overlap"},
    {TransferOverlap::Overlap, "overlap"},
};
constexpr std::pair<TransferDirection, std::string_view> kTransferDirectionTable[] = {
    {TransferDirection::Unknown, "unknown"},
    {TransferDirection::DeviceToDevice, "device_to_device"},
    {TransferDirection::HostToDevice, "host_to_device"},
    {TransferDirection::DeviceToHost, "device_to_host"},
};
constexpr std::pair<TransferPayloadClass, std::string_view> kTransferPayloadClassTable[] = {
    {TransferPayloadClass::Unknown, "unknown"},
    {TransferPayloadClass::LayerBoundaryHidden, "layer_boundary_hidden"},
    {TransferPayloadClass::EndpointHidden, "endpoint_hidden"},
    {TransferPayloadClass::NormalizedHidden, "normalized_hidden"},
    {TransferPayloadClass::LogitsResult, "logits_result"},
    {TransferPayloadClass::Other, "other"},
};
constexpr std::pair<HostInvolvement, std::string_view> kHostInvolvementTable[] = {
    {HostInvolvement::Unknown, "unknown"},
    {HostInvolvement::None, "none"},
    {HostInvolvement::PinnedStaging, "pinned_staging"},
    {HostInvolvement::PageableStaging, "pageable_staging"},
    {HostInvolvement::DriverManagedStaging, "driver_managed_staging"},
};
constexpr std::pair<HostInvolvement, std::string_view> kHostInvolvementV1Table[] = {
    {HostInvolvement::Unknown, "unknown"},
    {HostInvolvement::None, "none"},
    {HostInvolvement::PinnedStaging, "pinned_staging"},
    {HostInvolvement::PageableStaging, "pageable_staging"},
};
constexpr std::pair<GdnStage, std::string_view> kGdnStageTable[] = {
    {GdnStage::Unknown, "unknown"},
    {GdnStage::ChunkedPrefill, "chunked_prefill"},
    {GdnStage::RecurrentDecode, "recurrent_decode"},
    {GdnStage::StateRead, "state_read"},
    {GdnStage::StateUpdate, "state_update"},
    {GdnStage::Projection, "projection"},
    {GdnStage::ShortConv, "short_conv"},
};
constexpr std::pair<EvidenceClass, std::string_view> kEvidenceClassTable[] = {
    {EvidenceClass::Unknown, "unknown"},
    {EvidenceClass::Measured, "measured"},
    {EvidenceClass::MeasuredAnchor, "measured_anchor"},
    {EvidenceClass::Modeled, "modeled"},
    {EvidenceClass::Projected, "projected"},
};
constexpr std::pair<ValidityState, std::string_view> kValidityStateTable[] = {
    {ValidityState::Unqualified, "unqualified"}, {ValidityState::Valid, "valid"},
    {ValidityState::Partial, "partial"},         {ValidityState::Stale, "stale"},
    {ValidityState::Invalid, "invalid"},         {ValidityState::Unreliable, "unreliable"},
};
constexpr std::pair<CorrectnessStatus, std::string_view> kCorrectnessStatusTable[] = {
    {CorrectnessStatus::NotRun, "not_run"}, {CorrectnessStatus::Pass, "pass"},
    {CorrectnessStatus::Fail, "fail"},      {CorrectnessStatus::NotApplicable, "not_applicable"},
};
constexpr std::pair<RecordKind, std::string_view> kRecordKindTable[] = {
    {RecordKind::Operation, "operation"}, {RecordKind::Transfer, "transfer"},
    {RecordKind::Gdn, "gdn"},
};

} // namespace

std::string_view operation_family_name(OperationFamily family) {
    return enum_name(kOperationFamilyTable, family);
}
OperationFamily operation_family_from_name(std::string_view name) {
    return enum_from_name(kOperationFamilyTable, name);
}
std::string_view execution_phase_name(ExecutionPhase phase) {
    return enum_name(kExecutionPhaseTable, phase);
}
ExecutionPhase execution_phase_from_name(std::string_view name) {
    return enum_from_name(kExecutionPhaseTable, name);
}
std::string_view warm_state_name(WarmState state) { return enum_name(kWarmStateTable, state); }
WarmState warm_state_from_name(std::string_view name) {
    return enum_from_name(kWarmStateTable, name);
}
std::string_view source_state_name(SourceState state) { return enum_name(kSourceStateTable, state); }
SourceState source_state_from_name(std::string_view name) {
    return enum_from_name(kSourceStateTable, name);
}
std::string_view execution_mode_name(ExecutionMode mode) {
    return enum_name(kExecutionModeTable, mode);
}
ExecutionMode execution_mode_from_name(std::string_view name) {
    return enum_from_name(kExecutionModeTable, name);
}
std::string_view transfer_sync_name(TransferSync sync) { return enum_name(kTransferSyncTable, sync); }
TransferSync transfer_sync_from_name(std::string_view name) {
    return enum_from_name(kTransferSyncTable, name);
}
std::string_view transfer_route_name(TransferRoute route) {
    return enum_name(kTransferRouteTable, route);
}
TransferRoute transfer_route_from_name(std::string_view name) {
    return enum_from_name(kTransferRouteTable, name);
}
std::string_view transfer_overlap_name(TransferOverlap overlap) {
    return enum_name(kTransferOverlapTable, overlap);
}
TransferOverlap transfer_overlap_from_name(std::string_view name) {
    return enum_from_name(kTransferOverlapTable, name);
}
std::string_view transfer_direction_name(TransferDirection direction) {
    return enum_name(kTransferDirectionTable, direction);
}
TransferDirection transfer_direction_from_name(std::string_view name) {
    return enum_from_name(kTransferDirectionTable, name);
}
std::string_view transfer_payload_class_name(TransferPayloadClass payload_class) {
    return enum_name(kTransferPayloadClassTable, payload_class);
}
TransferPayloadClass transfer_payload_class_from_name(std::string_view name) {
    return enum_from_name(kTransferPayloadClassTable, name);
}
std::string_view host_involvement_name(HostInvolvement involvement) {
    return enum_name(kHostInvolvementTable, involvement);
}
HostInvolvement host_involvement_from_name(std::string_view name) {
    return enum_from_name(kHostInvolvementTable, name);
}
std::string_view gdn_stage_name(GdnStage stage) { return enum_name(kGdnStageTable, stage); }
GdnStage gdn_stage_from_name(std::string_view name) {
    return enum_from_name(kGdnStageTable, name);
}
std::string_view evidence_class_name(EvidenceClass evidence_class) {
    return enum_name(kEvidenceClassTable, evidence_class);
}
EvidenceClass evidence_class_from_name(std::string_view name) {
    return enum_from_name(kEvidenceClassTable, name);
}
std::string_view validity_state_name(ValidityState state) {
    return enum_name(kValidityStateTable, state);
}
ValidityState validity_state_from_name(std::string_view name) {
    return enum_from_name(kValidityStateTable, name);
}
std::string_view correctness_status_name(CorrectnessStatus status) {
    return enum_name(kCorrectnessStatusTable, status);
}
CorrectnessStatus correctness_status_from_name(std::string_view name) {
    return enum_from_name(kCorrectnessStatusTable, name);
}
std::string_view record_kind_name(RecordKind kind) { return enum_name(kRecordKindTable, kind); }
RecordKind record_kind_from_name(std::string_view name) {
    return enum_from_name(kRecordKindTable, name);
}

// -------------------------------------------------------------------------------------------
// Emission: one flat, single-level JSON object per line. Field order below IS the emission-order
// contract; parsing does not depend on order, but round-trip determinism (test a) does.
// -------------------------------------------------------------------------------------------

namespace {

void put_str(std::ostringstream& out, bool& first, std::string_view key, std::string_view value) {
    if (!first) { out << ','; }
    first = false;
    out << '"' << key << "\":\"" << json_escape(value) << '"';
}
void put_num(std::ostringstream& out, bool& first, std::string_view key, double value) {
    // JSON has no NaN/Inf literal; emitting a bare `nan`/`inf` token would be invalid JSON that
    // silently fails to round-trip. Refuse at the source instead -- ParseError is already in scope
    // and this keeps serialize()/parse_line() symmetric fail-closed on non-finite values.
    if (!std::isfinite(value)) {
        throw ParseError("refusing to serialize non-finite value for key \"" + std::string(key) + '"');
    }
    if (!first) { out << ','; }
    first = false;
    out << '"' << key << "\":" << number(value);
}
void put_int(std::ostringstream& out, bool& first, std::string_view key, long long value) {
    if (!first) { out << ','; }
    first = false;
    out << '"' << key << "\":" << value;
}
void put_bool(std::ostringstream& out, bool& first, std::string_view key, bool value) {
    if (!first) { out << ','; }
    first = false;
    out << '"' << key << "\":" << (value ? "true" : "false");
}

} // namespace

namespace {

[[nodiscard]] bool is_supported_schema_version(long long version) {
    return version == kSchemaVersionV1 || version == kSchemaVersionV2;
}

[[nodiscard]] bool is_v2_only(TransferRoute route) {
    return route == TransferRoute::RuntimeManagedMemcpy;
}

[[nodiscard]] bool is_v2_only(HostInvolvement involvement) {
    return involvement == HostInvolvement::DriverManagedStaging;
}

[[nodiscard]] std::optional<std::string> schema_compatibility_error(const CalibrationRecord& record) {
    if (!is_supported_schema_version(static_cast<long long>(record.identity.schema_version))) {
        return "unsupported schema_version " + std::to_string(record.identity.schema_version);
    }
    if (record.identity.schema_version == kSchemaVersionV1) {
        if (is_v2_only(record.transfer.route)) {
            return "schema_version=1 cannot use v2-only transfer_route";
        }
        if (is_v2_only(record.transfer.host_involvement)) {
            return "schema_version=1 cannot use v2-only transfer_host_involvement";
        }
    }
    return std::nullopt;
}

void validate_record_schema(const CalibrationRecord& record) {
    if (const std::optional<std::string> error = schema_compatibility_error(record)) {
        throw ParseError(*error);
    }
}

} // namespace

std::string serialize(const CalibrationRecord& record) {
    validate_record_schema(record);

    std::ostringstream out;
    out << '{';
    bool first = true;

    put_int(out, first, "schema_version", record.identity.schema_version);
    put_str(out, first, "artifact_type", kArtifactType);
    put_str(out, first, "record_kind", record_kind_name(record.kind));

    // identity.source
    put_str(out, first, "source_commit", record.identity.source.commit);
    put_str(out, first, "source_state", source_state_name(record.identity.source.state));
    // identity.build
    put_str(out, first, "build_cuda_arch_list", record.identity.build.cuda_arch_list);
    put_str(out, first, "build_type", record.identity.build.build_type);
    put_str(out, first, "build_id", record.identity.build.build_id);
    put_int(out, first, "build_compile_flags_count",
            static_cast<long long>(record.identity.build.compile_flags.size()));
    for (std::size_t i = 0; i < record.identity.build.compile_flags.size(); ++i) {
        put_str(out, first, "build_compile_flags." + std::to_string(i),
                record.identity.build.compile_flags[i]);
    }
    // identity.artifact
    put_str(out, first, "artifact_path", record.identity.artifact.path);
    put_str(out, first, "artifact_name", record.identity.artifact.name);
    put_str(out, first, "artifact_sha256", record.identity.artifact.sha256);
    put_str(out, first, "artifact_container_version", record.identity.artifact.container_version);
    put_str(out, first, "artifact_target_id", record.identity.artifact.target_id);
    put_str(out, first, "artifact_weights_id", record.identity.artifact.weights_id);
    put_int(out, first, "artifact_size_bytes",
            static_cast<long long>(record.identity.artifact.size_bytes));
    // identity.devices (indexed)
    put_int(out, first, "device_count", static_cast<long long>(record.identity.devices.size()));
    for (std::size_t i = 0; i < record.identity.devices.size(); ++i) {
        const DeviceIdentity& device = record.identity.devices[i];
        const std::string prefix     = "device." + std::to_string(i) + '.';
        put_int(out, first, prefix + "logical_slot", device.logical_slot);
        put_int(out, first, prefix + "physical_index", device.physical_index);
        put_str(out, first, prefix + "uuid", device.uuid);
        put_str(out, first, prefix + "model_name", device.model_name);
        put_int(out, first, prefix + "compute_capability_major", device.compute_capability_major);
        put_int(out, first, prefix + "compute_capability_minor", device.compute_capability_minor);
        put_int(out, first, prefix + "vram_bytes", static_cast<long long>(device.vram_bytes));
    }
    // identity.software
    put_str(out, first, "driver_version", record.identity.software.driver_version);
    put_str(out, first, "cuda_runtime_version", record.identity.software.cuda_runtime_version);
    put_str(out, first, "cuda_toolkit_version", record.identity.software.cuda_toolkit_version);
    put_int(out, first, "library_versions_count",
            static_cast<long long>(record.identity.software.library_versions.size()));
    for (std::size_t i = 0; i < record.identity.software.library_versions.size(); ++i) {
        put_str(out, first, "library_versions." + std::to_string(i),
                record.identity.software.library_versions[i]);
    }
    // identity optional audit metadata
    put_str(out, first, "branch_name", record.identity.branch_name);
    put_str(out, first, "hostname", record.identity.hostname);
    put_str(out, first, "os_description", record.identity.os_description);
    put_str(out, first, "command_line", record.identity.command_line);

    // operation
    put_str(out, first, "operation_family", operation_family_name(record.operation.family));
    put_str(out, first, "operation_phase", execution_phase_name(record.operation.phase));
    put_int(out, first, "operation_device_logical_slot", record.operation.device_logical_slot);
    put_int(out, first, "operation_shape_b", static_cast<long long>(record.operation.shape.b));
    put_int(out, first, "operation_shape_t", static_cast<long long>(record.operation.shape.t));
    put_int(out, first, "operation_shape_h", static_cast<long long>(record.operation.shape.h));
    put_int(out, first, "operation_shape_k", static_cast<long long>(record.operation.shape.k));
    put_int(out, first, "operation_shape_v", static_cast<long long>(record.operation.shape.v));
    put_int(out, first, "operation_shape_chunk_size",
            static_cast<long long>(record.operation.shape.chunk_size));
    put_int(out, first, "operation_shape_context_len",
            static_cast<long long>(record.operation.shape.context_len));
    put_int(out, first, "operation_shape_rows", static_cast<long long>(record.operation.shape.rows));
    put_int(out, first, "operation_shape_cols", static_cast<long long>(record.operation.shape.cols));
    put_str(out, first, "operation_dtype", record.operation.dtype);
    put_str(out, first, "operation_quant_route", record.operation.quant_route);
    put_str(out, first, "operation_execution_mode",
            execution_mode_name(record.operation.execution_mode));
    put_str(out, first, "operation_warm_state", warm_state_name(record.operation.warm_state));
    put_int(out, first, "operation_endpoint_owner_slot", record.operation.endpoint_owner_slot);

    // transfer
    put_str(out, first, "transfer_direction", transfer_direction_name(record.transfer.direction));
    put_str(out, first, "transfer_source_device_uuid", record.transfer.source_device_uuid);
    put_str(out, first, "transfer_destination_device_uuid",
            record.transfer.destination_device_uuid);
    put_str(out, first, "transfer_payload_class",
            transfer_payload_class_name(record.transfer.payload_class));
    put_str(out, first, "transfer_payload_class_label", record.transfer.payload_class_label);
    put_int(out, first, "transfer_payload_bytes",
            static_cast<long long>(record.transfer.payload_bytes));
    put_str(out, first, "transfer_api", record.transfer.transfer_api);
    put_str(out, first, "transfer_sync", transfer_sync_name(record.transfer.sync));
    put_str(out, first, "transfer_route", transfer_route_name(record.transfer.route));
    put_str(out, first, "transfer_host_involvement",
            host_involvement_name(record.transfer.host_involvement));
    put_str(out, first, "transfer_overlap", transfer_overlap_name(record.transfer.overlap));

    // gdn
    put_str(out, first, "gdn_stage", gdn_stage_name(record.gdn.stage));
    put_int(out, first, "gdn_shape_b", static_cast<long long>(record.gdn.shape.b));
    put_int(out, first, "gdn_shape_t", static_cast<long long>(record.gdn.shape.t));
    put_int(out, first, "gdn_shape_h", static_cast<long long>(record.gdn.shape.h));
    put_int(out, first, "gdn_shape_k", static_cast<long long>(record.gdn.shape.k));
    put_int(out, first, "gdn_shape_v", static_cast<long long>(record.gdn.shape.v));
    put_int(out, first, "gdn_shape_chunk_size", static_cast<long long>(record.gdn.shape.chunk_size));
    put_str(out, first, "gdn_state_dtype", record.gdn.state_dtype);
    put_str(out, first, "gdn_activation_dtype", record.gdn.activation_dtype);
    put_bool(out, first, "gdn_requires_synchronization", record.gdn.requires_synchronization);
    put_bool(out, first, "gdn_initial_state", record.gdn.initial_state);
    put_bool(out, first, "gdn_final_state", record.gdn.final_state);
    put_int(out, first, "gdn_workspace_bytes", static_cast<long long>(record.gdn.workspace_bytes));

    // raw sample
    put_str(out, first, "sample_run_id", record.sample.run_id);
    put_str(out, first, "sample_observation_id", record.sample.observation_id);
    put_int(out, first, "sample_index", static_cast<long long>(record.sample.sample_index));
    put_num(out, first, "sample_value", record.sample.value);
    put_str(out, first, "sample_unit", record.sample.unit);
    put_bool(out, first, "sample_timestamp_known", record.sample.timestamp_unix_ms.has_value());
    put_int(out, first, "sample_timestamp_unix_ms",
            static_cast<long long>(record.sample.timestamp_unix_ms.value_or(0)));
    put_str(out, first, "sample_qualification",
            correctness_status_name(record.sample.qualification));

    // qualification state
    put_str(out, first, "evidence_class", evidence_class_name(record.qualification.evidence));
    put_str(out, first, "validity_state", validity_state_name(record.qualification.validity));
    put_str(out, first, "correctness_status",
            correctness_status_name(record.qualification.correctness));

    out << '}';
    return out.str();
}

// -------------------------------------------------------------------------------------------
// Parsing: single-pass flat-object scanner. No recursion, no nested-object/array support -- the
// schema deliberately never needs either (see calibration.h header comment). Values are strings,
// numbers, or booleans only.
// -------------------------------------------------------------------------------------------

namespace {

[[nodiscard]] inline bool is_flat_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r';
}

struct FlatObject {
    std::unordered_map<std::string, std::string> raw; // token text, unquoted for strings

    [[nodiscard]] bool has(const std::string& key) const { return raw.find(key) != raw.end(); }
    [[nodiscard]] std::string str(const std::string& key) const {
        auto it = raw.find(key);
        return it == raw.end() ? std::string{} : it->second;
    }
    // Required variant: throws when the key is absent. An empty *value* (e.g. ""source_commit":""`)
    // is legal and distinct from a missing key -- this only guards presence.
    [[nodiscard]] std::string required_str(const std::string& key) const {
        auto it = raw.find(key);
        if (it == raw.end()) { throw ParseError("record is missing required key \"" + key + '"'); }
        return it->second;
    }
    // Signed integer, required present and fully-consumed. Legitimate for fields that may be -1
    // (logical_slot, physical_index, endpoint_owner_slot, ...).
    [[nodiscard]] long long required_integer(const std::string& key) const {
        auto it = raw.find(key);
        if (it == raw.end()) { throw ParseError("record is missing required key \"" + key + '"'); }
        const std::string& token = it->second;
        long long value = 0;
        const auto* begin = token.data();
        const auto* end   = token.data() + token.size();
        const auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end) {
            throw ParseError("malformed integer for key \"" + key + "\": \"" + token + '"');
        }
        return value;
    }
    [[nodiscard]] long long integer(const std::string& key) const { return required_integer(key); }
    // Unsigned/count field: required present, fully-consumed, and rejects negative BEFORE the
    // caller casts to an unsigned type.
    [[nodiscard]] long long required_count(const std::string& key) const {
        const long long value = required_integer(key);
        if (value < 0) {
            throw ParseError("negative count for key \"" + key + "\": " + std::to_string(value));
        }
        return value;
    }
    [[nodiscard]] double real(const std::string& key) const {
        auto it = raw.find(key);
        if (it == raw.end()) { throw ParseError("record is missing required key \"" + key + '"'); }
        const std::string& token = it->second;
        if (token.empty()) {
            throw ParseError("malformed real for key \"" + key + "\": empty");
        }
        const char* begin = token.c_str();
        char* end_ptr      = nullptr;
        errno              = 0;
        const double value = std::strtod(begin, &end_ptr);
        if (end_ptr != begin + token.size()) {
            throw ParseError("malformed real for key \"" + key + "\": \"" + token + '"');
        }
        // strtod consumes "nan"/"inf" wholesale (end_ptr check above doesn't catch them) and sets
        // errno=ERANGE on magnitude overflow (e.g. 1e999) while still returning a value and
        // consuming the whole token. JSON has no NaN/Inf literal, so both are fail-closed here by
        // design rather than silently admitted.
        if (errno == ERANGE) {
            throw ParseError("real value out of range for key \"" + key + "\": \"" + token + '"');
        }
        if (!std::isfinite(value)) {
            throw ParseError("non-finite real for key \"" + key + "\": \"" + token + '"');
        }
        return value;
    }
    [[nodiscard]] bool boolean(const std::string& key) const {
        auto it = raw.find(key);
        if (it == raw.end()) { throw ParseError("record is missing required key \"" + key + '"'); }
        if (it->second == "true") { return true; }
        if (it->second == "false") { return false; }
        throw ParseError("malformed boolean for key \"" + key + "\": \"" + it->second + '"');
    }
};

// Parses one `"key":value` pair starting at `pos` (just past a '{' or ','), returns the position
// just past the parsed value. Throws ParseError on any malformed token -- callers only ever see a
// well-formed flat object or an explicit error, never a partially-populated silent guess.
std::size_t parse_pair(std::string_view line, std::size_t pos, FlatObject& out) {
    if (pos >= line.size() || line[pos] != '"') {
        throw ParseError("expected key string at offset " + std::to_string(pos));
    }
    ++pos;
    const std::size_t key_start = pos;
    while (pos < line.size() && line[pos] != '"') { ++pos; }
    if (pos >= line.size()) { throw ParseError("unterminated key string"); }
    const std::string key = std::string(line.substr(key_start, pos - key_start));
    ++pos; // past closing quote
    if (pos >= line.size() || line[pos] != ':') {
        throw ParseError("expected ':' after key \"" + key + '"');
    }
    ++pos;

    if (pos >= line.size()) { throw ParseError("truncated value for key \"" + key + '"'); }

    if (line[pos] == '"') {
        ++pos;
        std::string value;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\' && pos + 1 < line.size()) {
                const char escaped = line[pos + 1];
                switch (escaped) {
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                case '"': value += '"'; break;
                case '\\': value += '\\'; break;
                default: value += escaped; break;
                }
                pos += 2;
            } else {
                value += line[pos];
                ++pos;
            }
        }
        if (pos >= line.size()) { throw ParseError("unterminated string value for key \"" + key + '"'); }
        ++pos; // past closing quote
        if (!out.raw.emplace(key, value).second) {
            throw ParseError("duplicate key \"" + key + '"');
        }
        return pos;
    }

    // number or bool literal: read until ',' or '}'.
    const std::size_t value_start = pos;
    while (pos < line.size() && line[pos] != ',' && line[pos] != '}') { ++pos; }
    if (pos >= line.size()) { throw ParseError("truncated literal value for key \"" + key + '"'); }
    if (!out.raw.emplace(key, std::string(line.substr(value_start, pos - value_start))).second) {
        throw ParseError("duplicate key \"" + key + '"');
    }
    return pos;
}

// Returns the offset just past the closing '}' via `end_offset`, so callers (parse_line) can
// require that only whitespace follows.
FlatObject parse_flat_object(std::string_view line, std::size_t& end_offset) {
    FlatObject out;
    std::size_t pos = 0;
    while (pos < line.size() && is_flat_space(line[pos])) { ++pos; }
    if (pos >= line.size() || line[pos] != '{') { throw ParseError("record is not a JSON object"); }
    ++pos;
    while (pos < line.size() && is_flat_space(line[pos])) { ++pos; }
    if (pos < line.size() && line[pos] == '}') { // empty object
        end_offset = pos + 1;
        return out;
    }
    for (;;) {
        pos = parse_pair(line, pos, out);
        while (pos < line.size() && is_flat_space(line[pos])) { ++pos; }
        if (pos >= line.size()) { throw ParseError("record missing closing '}'"); }
        if (line[pos] == ',') {
            ++pos;
            while (pos < line.size() && is_flat_space(line[pos])) { ++pos; }
            continue;
        }
        if (line[pos] == '}') { break; }
        throw ParseError("expected ',' or '}' in record");
    }
    end_offset = pos + 1;
    return out;
}

} // namespace

namespace {

// Checked enum read from a required string key: throws on missing key (via required_str) and on
// an unrecognized token (via enum_from_name_checked).
template <typename Enum, std::size_t N>
Enum required_enum(const FlatObject& object, const std::pair<Enum, std::string_view> (&table)[N],
                    const std::string& key) {
    const std::string token = object.required_str(key);
    const std::optional<Enum> value = enum_from_name_checked(table, token);
    if (!value.has_value()) {
        throw ParseError("unknown enum token for key \"" + key + "\": \"" + token + '"');
    }
    return *value;
}

} // namespace

CalibrationRecord parse_line(std::string_view line) {
    std::size_t end_offset  = 0;
    const FlatObject object = parse_flat_object(line, end_offset);
    while (end_offset < line.size() && is_flat_space(line[end_offset])) { ++end_offset; }
    if (end_offset != line.size()) {
        throw ParseError("trailing content after closing '}'");
    }

    // Schema gate FIRST, fail loudly: absent key or wrong version is never default-and-continue.
    if (!object.has("schema_version")) {
        throw ParseError("record is missing required key \"schema_version\"");
    }
    const long long schema_version = object.required_integer("schema_version");
    if (!is_supported_schema_version(schema_version)) {
        throw ParseError("unsupported schema_version " + std::to_string(schema_version) +
                         " (expected " + std::to_string(kSchemaVersionV1) + " or " +
                         std::to_string(kSchemaVersionV2) + ")");
    }

    if (object.required_str("artifact_type") != std::string(kArtifactType)) {
        throw ParseError("unexpected artifact_type");
    }

    CalibrationRecord record;
    record.identity.schema_version = static_cast<int>(schema_version);
    record.kind = required_enum(object, kRecordKindTable, "record_kind");

    record.identity.source.commit = object.required_str("source_commit");
    record.identity.source.state  = required_enum(object, kSourceStateTable, "source_state");

    record.identity.build.cuda_arch_list = object.required_str("build_cuda_arch_list");
    record.identity.build.build_type     = object.required_str("build_type");
    record.identity.build.build_id       = object.required_str("build_id");
    const long long compile_flags_count = object.required_count("build_compile_flags_count");
    record.identity.build.compile_flags.reserve(static_cast<std::size_t>(compile_flags_count));
    for (long long i = 0; i < compile_flags_count; ++i) {
        record.identity.build.compile_flags.push_back(
            object.required_str("build_compile_flags." + std::to_string(i)));
    }

    record.identity.artifact.path              = object.required_str("artifact_path");
    record.identity.artifact.name               = object.required_str("artifact_name");
    record.identity.artifact.sha256             = object.required_str("artifact_sha256");
    record.identity.artifact.container_version   = object.required_str("artifact_container_version");
    record.identity.artifact.target_id           = object.required_str("artifact_target_id");
    record.identity.artifact.weights_id          = object.required_str("artifact_weights_id");
    record.identity.artifact.size_bytes =
        static_cast<std::uint64_t>(object.required_count("artifact_size_bytes"));

    const long long device_count = object.required_count("device_count");
    record.identity.devices.reserve(static_cast<std::size_t>(device_count));
    for (long long i = 0; i < device_count; ++i) {
        const std::string prefix = "device." + std::to_string(i) + '.';
        DeviceIdentity device;
        device.logical_slot   = static_cast<int>(object.required_integer(prefix + "logical_slot"));
        device.physical_index = static_cast<int>(object.required_integer(prefix + "physical_index"));
        device.uuid           = object.required_str(prefix + "uuid");
        device.model_name     = object.required_str(prefix + "model_name");
        device.compute_capability_major =
            static_cast<int>(object.required_integer(prefix + "compute_capability_major"));
        device.compute_capability_minor =
            static_cast<int>(object.required_integer(prefix + "compute_capability_minor"));
        device.vram_bytes =
            static_cast<std::uint64_t>(object.required_count(prefix + "vram_bytes"));
        record.identity.devices.push_back(std::move(device));
    }

    record.identity.software.driver_version       = object.required_str("driver_version");
    record.identity.software.cuda_runtime_version  = object.required_str("cuda_runtime_version");
    record.identity.software.cuda_toolkit_version  = object.required_str("cuda_toolkit_version");
    const long long library_versions_count = object.required_count("library_versions_count");
    record.identity.software.library_versions.reserve(
        static_cast<std::size_t>(library_versions_count));
    for (long long i = 0; i < library_versions_count; ++i) {
        record.identity.software.library_versions.push_back(
            object.required_str("library_versions." + std::to_string(i)));
    }

    record.identity.branch_name    = object.required_str("branch_name");
    record.identity.hostname       = object.required_str("hostname");
    record.identity.os_description = object.required_str("os_description");
    record.identity.command_line   = object.required_str("command_line");

    record.operation.family = required_enum(object, kOperationFamilyTable, "operation_family");
    record.operation.phase  = required_enum(object, kExecutionPhaseTable, "operation_phase");
    record.operation.device_logical_slot =
        static_cast<int>(object.required_integer("operation_device_logical_slot"));
    record.operation.shape.b =
        static_cast<std::uint64_t>(object.required_count("operation_shape_b"));
    record.operation.shape.t =
        static_cast<std::uint64_t>(object.required_count("operation_shape_t"));
    record.operation.shape.h =
        static_cast<std::uint64_t>(object.required_count("operation_shape_h"));
    record.operation.shape.k =
        static_cast<std::uint64_t>(object.required_count("operation_shape_k"));
    record.operation.shape.v =
        static_cast<std::uint64_t>(object.required_count("operation_shape_v"));
    record.operation.shape.chunk_size =
        static_cast<std::uint64_t>(object.required_count("operation_shape_chunk_size"));
    record.operation.shape.context_len =
        static_cast<std::uint64_t>(object.required_count("operation_shape_context_len"));
    record.operation.shape.rows =
        static_cast<std::uint64_t>(object.required_count("operation_shape_rows"));
    record.operation.shape.cols =
        static_cast<std::uint64_t>(object.required_count("operation_shape_cols"));
    record.operation.dtype       = object.required_str("operation_dtype");
    record.operation.quant_route = object.required_str("operation_quant_route");
    record.operation.execution_mode =
        required_enum(object, kExecutionModeTable, "operation_execution_mode");
    record.operation.warm_state  = required_enum(object, kWarmStateTable, "operation_warm_state");
    record.operation.endpoint_owner_slot =
        static_cast<int>(object.required_integer("operation_endpoint_owner_slot"));

    record.transfer.direction =
        required_enum(object, kTransferDirectionTable, "transfer_direction");
    record.transfer.source_device_uuid      = object.required_str("transfer_source_device_uuid");
    record.transfer.destination_device_uuid = object.required_str("transfer_destination_device_uuid");
    record.transfer.payload_class =
        required_enum(object, kTransferPayloadClassTable, "transfer_payload_class");
    record.transfer.payload_class_label = object.required_str("transfer_payload_class_label");
    record.transfer.payload_bytes =
        static_cast<std::uint64_t>(object.required_count("transfer_payload_bytes"));
    record.transfer.transfer_api = object.required_str("transfer_api");
    record.transfer.sync  = required_enum(object, kTransferSyncTable, "transfer_sync");
    if (schema_version == kSchemaVersionV1) {
        record.transfer.route = required_enum(object, kTransferRouteV1Table, "transfer_route");
    } else {
        record.transfer.route = required_enum(object, kTransferRouteTable, "transfer_route");
    }
    record.transfer.host_involvement =
        schema_version == kSchemaVersionV1
            ? required_enum(object, kHostInvolvementV1Table, "transfer_host_involvement")
            : required_enum(object, kHostInvolvementTable, "transfer_host_involvement");
    record.transfer.overlap = required_enum(object, kTransferOverlapTable, "transfer_overlap");

    record.gdn.stage = required_enum(object, kGdnStageTable, "gdn_stage");
    record.gdn.shape.b = static_cast<std::uint64_t>(object.required_count("gdn_shape_b"));
    record.gdn.shape.t = static_cast<std::uint64_t>(object.required_count("gdn_shape_t"));
    record.gdn.shape.h = static_cast<std::uint64_t>(object.required_count("gdn_shape_h"));
    record.gdn.shape.k = static_cast<std::uint64_t>(object.required_count("gdn_shape_k"));
    record.gdn.shape.v = static_cast<std::uint64_t>(object.required_count("gdn_shape_v"));
    record.gdn.shape.chunk_size =
        static_cast<std::uint64_t>(object.required_count("gdn_shape_chunk_size"));
    record.gdn.state_dtype      = object.required_str("gdn_state_dtype");
    record.gdn.activation_dtype = object.required_str("gdn_activation_dtype");
    record.gdn.requires_synchronization = object.boolean("gdn_requires_synchronization");
    record.gdn.initial_state            = object.boolean("gdn_initial_state");
    record.gdn.final_state              = object.boolean("gdn_final_state");
    record.gdn.workspace_bytes =
        static_cast<std::uint64_t>(object.required_count("gdn_workspace_bytes"));

    record.sample.run_id         = object.required_str("sample_run_id");
    record.sample.observation_id = object.required_str("sample_observation_id");
    record.sample.sample_index   = static_cast<std::uint64_t>(object.required_count("sample_index"));
    record.sample.value          = object.real("sample_value");
    record.sample.unit           = object.required_str("sample_unit");
    if (object.boolean("sample_timestamp_known")) {
        record.sample.timestamp_unix_ms =
            static_cast<std::uint64_t>(object.required_count("sample_timestamp_unix_ms"));
    } else {
        (void)object.required_integer("sample_timestamp_unix_ms"); // key still required present.
    }
    record.sample.qualification =
        required_enum(object, kCorrectnessStatusTable, "sample_qualification");

    record.qualification.evidence    = required_enum(object, kEvidenceClassTable, "evidence_class");
    record.qualification.validity    = required_enum(object, kValidityStateTable, "validity_state");
    record.qualification.correctness =
        required_enum(object, kCorrectnessStatusTable, "correctness_status");

    return record;
}

CorpusParseResult parse_corpus(std::string_view jsonl_text) {
    CorpusParseResult result;
    std::size_t line_index = 0;
    std::size_t pos        = 0;
    while (pos <= jsonl_text.size()) {
        const std::size_t newline   = jsonl_text.find('\n', pos);
        const std::string_view line = newline == std::string_view::npos
                                          ? jsonl_text.substr(pos)
                                          : jsonl_text.substr(pos, newline - pos);
        const bool blank = line.find_first_not_of(" \t\r") == std::string_view::npos;
        if (!blank) {
            try {
                result.records.push_back(parse_line(line));
            } catch (const ParseError& error) {
                result.line_errors.emplace_back(line_index, error.what());
            }
        }
        if (newline == std::string_view::npos) { break; }
        pos = newline + 1;
        ++line_index;
    }
    return result;
}

// -------------------------------------------------------------------------------------------
// Authoritative calibration-evidence gate (Task 1/2). One function: a common core that applies
// to every record kind, then a `switch (record.kind)` tail for the fields only that kind uses.
// Says nothing about blocks the kind doesn't use (an Operation record may legitimately carry a
// fully populated gdn/transfer block alongside it -- that is not itself a defect).
// -------------------------------------------------------------------------------------------

namespace {

// Any device whose `logical_slot` matches is considered "in the participating inventory" -- used
// both for an operation's device_logical_slot and for a transfer endpoint UUID membership check.
[[nodiscard]] bool device_slot_present(const std::vector<DeviceIdentity>& devices, int slot) {
    for (const DeviceIdentity& device : devices) {
        if (device.logical_slot == slot) { return true; }
    }
    return false;
}
[[nodiscard]] bool device_uuid_present(const std::vector<DeviceIdentity>& devices,
                                        const std::string& uuid) {
    for (const DeviceIdentity& device : devices) {
        if (device.uuid == uuid) { return true; }
    }
    return false;
}

// Minimum-shape rule for Operation records, family/phase-aware (spec WORK ITEM B). Only the dims
// a family/phase combination actually depends on are required non-zero; ShapeDims members left at
// 0 for an inapplicable dim are legitimate (header comment: "not applicable", not "measured as
// zero"). context_len is deliberately never required here -- it is REQUIRED KEY "when relevant"
// per prep doc sec 7, and no family/phase combination in this gate treats it as load-bearing.
[[nodiscard]] bool operation_shape_satisfied(const OperationIdentity& operation) {
    switch (operation.family) {
    case OperationFamily::Gdn:
    case OperationFamily::Attention:
        if (operation.phase == ExecutionPhase::Prefill || operation.phase == ExecutionPhase::Decode) {
            return operation.shape.b != 0 && operation.shape.t != 0;
        }
        return true;
    case OperationFamily::Projection:
    case OperationFamily::Mlp:
        return operation.shape.rows != 0 && operation.shape.cols != 0;
    case OperationFamily::LmHead:
    case OperationFamily::FinalNorm:
    case OperationFamily::Endpoint:
        return operation.shape.cols != 0;
    default:
        return true;
    }
}

// GDN minimum-shape rule by stage (spec WORK ITEM B: "adjust only if the prep doc contradicts").
// Prep doc sec 8's GDN decode/prefill rows key on device/route/graph/width -- b/t covers "width";
// StateRead/StateUpdate are keyed on batch/head per sec 7's shape classification, not tokens.
[[nodiscard]] bool gdn_shape_satisfied(const GdnIdentity& gdn) {
    switch (gdn.stage) {
    case GdnStage::ChunkedPrefill:
    case GdnStage::RecurrentDecode:
    case GdnStage::Projection:
    case GdnStage::ShortConv:
        return gdn.shape.b != 0 && gdn.shape.t != 0;
    case GdnStage::StateRead:
    case GdnStage::StateUpdate:
        return gdn.shape.b != 0 && gdn.shape.h != 0;
    default:
        return true;
    }
}

} // namespace

std::vector<std::string> missing_identity_requirements(const CalibrationIdentity& identity) {
    std::vector<std::string> reasons;

    if (!identity.source.commit_known()) { reasons.push_back("source_commit unresolved"); }
    // Unknown and Dirty both disqualify, with distinct reasons: Unknown means "never checked",
    // Dirty means "checked and the tree has uncommitted changes" -- neither is a clean-known build.
    if (identity.source.state == SourceState::Unknown) {
        reasons.push_back("source_state unresolved");
    } else if (identity.source.state == SourceState::Dirty) {
        reasons.push_back("source is dirty");
    }
    if (!identity.artifact.sha256_known()) { reasons.push_back("artifact_sha256 unresolved"); }
    if (identity.artifact.target_id.empty()) { reasons.push_back("artifact_target_id unresolved"); }
    if (identity.artifact.weights_id.empty()) { reasons.push_back("artifact_weights_id unresolved"); }
    if (identity.build.build_id.empty()) { reasons.push_back("build_id unresolved"); }
    if (identity.devices.empty()) { reasons.push_back("no devices recorded"); }
    for (const DeviceIdentity& device : identity.devices) {
        if (!device.uuid_known()) { reasons.push_back("a device uuid is unresolved"); }
        if (device.logical_slot < 0) { reasons.push_back("a device logical_slot is unresolved"); }
        if (device.physical_index < 0) { reasons.push_back("a device physical_index is unresolved"); }
        if (device.model_name.empty()) { reasons.push_back("a device model_name is unresolved"); }
        // Major only: minor == 0 is a legitimate compute capability (e.g. cc 12.0), not "unset".
        if (device.compute_capability_major <= 0) {
            reasons.push_back("a device compute_capability_major is unresolved");
        }
        if (device.vram_bytes == 0) { reasons.push_back("a device vram_bytes is unresolved"); }
    }
    if (identity.build.cuda_arch_list.empty()) { reasons.push_back("build_cuda_arch_list unresolved"); }
    if (identity.build.build_type.empty()) { reasons.push_back("build_type unresolved"); }
    if (identity.software.driver_version.empty()) { reasons.push_back("driver_version unresolved"); }
    if (identity.software.cuda_runtime_version.empty()) {
        reasons.push_back("cuda_runtime_version unresolved");
    }
    if (identity.software.cuda_toolkit_version.empty()) {
        reasons.push_back("cuda_toolkit_version unresolved");
    }
    return reasons;
}

std::vector<std::string> missing_requirements(const CalibrationRecord& record) {
    std::vector<std::string> reasons = missing_identity_requirements(record.identity);
    if (const std::optional<std::string> error = schema_compatibility_error(record)) {
        reasons.push_back(*error);
    }

    // ---- common core (sample/measurement): applies to every record kind ----
    if (record.sample.unit.empty()) { reasons.push_back("sample_unit unresolved"); }
    if (record.sample.run_id.empty()) { reasons.push_back("sample_run_id unresolved"); }
    if (record.sample.observation_id.empty()) { reasons.push_back("sample_observation_id unresolved"); }
    if (record.qualification.correctness == CorrectnessStatus::NotRun) {
        reasons.push_back("correctness has not been run");
    }
    // Sample/record correctness coupling: since status_allows_prediction already forces record-
    // level correctness == Pass for eligibility, this single equality is what rejects a sample
    // individually marked Fail or NotRun even when the record-level status says Pass, AND is the
    // conservative definition of NotApplicable adopted here -- a sample explicitly marked "not
    // applicable" never silently rides through as equivalent to a passing measured sample.
    if (record.sample.qualification != record.qualification.correctness) {
        reasons.push_back("sample_qualification does not match record-level correctness");
    }
    if (!record.sample.timestamp_unix_ms.has_value()) {
        reasons.push_back("sample timestamp unresolved");
    }
    // An in-memory record never passes through the parser (which already rejects non-finite reals
    // at parse time), so the gate needs its own finite check to cover records built directly in
    // C++.
    if (!std::isfinite(record.sample.value)) { reasons.push_back("sample value is not finite"); }
    if (record.operation.phase == ExecutionPhase::Unknown) { reasons.push_back("operation_phase unresolved"); }

    // ---- kind-specific tail ----
    switch (record.kind) {
    case RecordKind::Operation:
        if (record.operation.family == OperationFamily::Unknown) {
            reasons.push_back("operation_family unresolved");
        }
        if (record.operation.device_logical_slot < 0) {
            reasons.push_back("operation_device_logical_slot unresolved");
        } else if (!device_slot_present(record.identity.devices, record.operation.device_logical_slot)) {
            reasons.push_back("operation_device_logical_slot does not match a participating device");
        }
        if (record.operation.dtype.empty()) { reasons.push_back("operation_dtype unresolved"); }
        // "none" is the valid explicit value for an unquantized route; only emptiness is rejected.
        if (record.operation.quant_route.empty()) {
            reasons.push_back("operation_quant_route unresolved");
        }
        if (record.operation.warm_state == WarmState::Unknown) {
            reasons.push_back("operation_warm_state unresolved");
        }
        if (record.operation.execution_mode == ExecutionMode::Unknown) {
            reasons.push_back("operation_execution_mode unresolved");
        }
        if (!operation_shape_satisfied(record.operation)) {
            reasons.push_back("operation shape is missing dims required for its family/phase");
        }
        break;
    case RecordKind::Gdn:
        if (record.gdn.stage == GdnStage::Unknown) { reasons.push_back("gdn_stage unresolved"); }
        if (record.gdn.activation_dtype.empty()) { reasons.push_back("gdn_activation_dtype unresolved"); }
        if (record.operation.device_logical_slot < 0) {
            reasons.push_back("operation_device_logical_slot unresolved");
        } else if (!device_slot_present(record.identity.devices, record.operation.device_logical_slot)) {
            reasons.push_back("operation_device_logical_slot does not match a participating device");
        }
        if (record.operation.execution_mode == ExecutionMode::Unknown) {
            reasons.push_back("operation_execution_mode unresolved");
        }
        // warm/cold distinction applies to every GDN stage: the prep doc (sec 7/8) gives no basis
        // to exempt any particular stage from the warm/cold-state requirement it imposes on
        // operations generally, so it is required uniformly here rather than guessed per stage.
        if (record.operation.warm_state == WarmState::Unknown) {
            reasons.push_back("operation_warm_state unresolved");
        }
        if (!gdn_shape_satisfied(record.gdn)) {
            reasons.push_back("gdn shape is missing dims required for its stage");
        }
        // state_dtype required only for stages that actually access/update recurrent state.
        // ChunkedPrefill is deliberately excluded: prep doc sec 8's "GDN prefill" row lists its
        // independent variables as device/prefill width/route/graph-eager with no state term (the
        // recurrent state only enters at decode/state-read/state-update), so a prefill chunk pass
        // is treated like a fresh compute pass rather than a state-accessing one. Projection and
        // ShortConv are likewise excluded (spec explicit).
        switch (record.gdn.stage) {
        case GdnStage::RecurrentDecode:
        case GdnStage::StateRead:
        case GdnStage::StateUpdate:
            if (record.gdn.state_dtype.empty()) { reasons.push_back("gdn_state_dtype unresolved"); }
            break;
        default:
            break;
        }
        break;
    case RecordKind::Transfer:
        if (record.transfer.direction == TransferDirection::Unknown) {
            reasons.push_back("transfer_direction unresolved");
        }
        if (record.transfer.payload_class == TransferPayloadClass::Unknown) {
            reasons.push_back("transfer_payload_class unresolved");
        }
        if (record.transfer.payload_class == TransferPayloadClass::Other &&
            record.transfer.payload_class_label.empty()) {
            reasons.push_back("transfer_payload_class_label required when payload_class is other");
        }
        if (record.transfer.payload_bytes == 0) { reasons.push_back("transfer_payload_bytes unresolved"); }
        if (record.transfer.transfer_api.empty()) { reasons.push_back("transfer_api unresolved"); }
        if (record.transfer.host_involvement == HostInvolvement::Unknown) {
            reasons.push_back("transfer_host_involvement unresolved");
        }
        if (record.transfer.sync == TransferSync::Unknown) {
            reasons.push_back("transfer_sync unresolved");
        }
        if (record.transfer.route == TransferRoute::Unknown) {
            reasons.push_back("transfer_route unresolved");
        }
        if (record.transfer.overlap == TransferOverlap::Unknown) {
            reasons.push_back("transfer_overlap unresolved");
        }
        switch (record.transfer.direction) {
        case TransferDirection::HostToDevice:
            if (record.transfer.destination_device_uuid.empty()) {
                reasons.push_back("transfer_destination_device_uuid unresolved");
            } else if (!device_uuid_present(record.identity.devices,
                                             record.transfer.destination_device_uuid)) {
                reasons.push_back("transfer_destination_device_uuid does not match a participating device");
            }
            break;
        case TransferDirection::DeviceToHost:
            if (record.transfer.source_device_uuid.empty()) {
                reasons.push_back("transfer_source_device_uuid unresolved");
            } else if (!device_uuid_present(record.identity.devices,
                                             record.transfer.source_device_uuid)) {
                reasons.push_back("transfer_source_device_uuid does not match a participating device");
            }
            break;
        case TransferDirection::DeviceToDevice:
        case TransferDirection::Unknown:
        default:
            if (record.transfer.source_device_uuid.empty()) {
                reasons.push_back("transfer_source_device_uuid unresolved");
            } else if (!device_uuid_present(record.identity.devices,
                                             record.transfer.source_device_uuid)) {
                reasons.push_back("transfer_source_device_uuid does not match a participating device");
            }
            if (record.transfer.destination_device_uuid.empty()) {
                reasons.push_back("transfer_destination_device_uuid unresolved");
            } else if (!device_uuid_present(record.identity.devices,
                                             record.transfer.destination_device_uuid)) {
                reasons.push_back("transfer_destination_device_uuid does not match a participating device");
            }
            break;
        }
        break;
    }

    return reasons;
}

bool is_qualified_calibration_evidence(const CalibrationRecord& record) {
    if (!missing_requirements(record).empty()) { return false; }
    if (!record.qualification.status_allows_prediction()) { return false; }
    return record.qualification.evidence == EvidenceClass::Measured ||
           record.qualification.evidence == EvidenceClass::MeasuredAnchor;
}

} // namespace ninfer::targets::calibration

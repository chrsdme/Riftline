#pragma once

// M5.4a: the calibration data CONTRACT only. Types, deterministic JSONL serialization, schema
// versioning, and qualification/validity semantics for one observation. No collector, no writer
// that drives a real run, no summary/percentile engine, no benchmark or probe -- all deferred to
// M5.4b (see doc/research/NINFER_M5_4_CALIBRATION_PREPARATION_2026-09-19.md sec 12/13).
//
// Every identity field that M5.4b will eventually populate live (CUDA device query, git shell-out,
// artifact SHA256) has an explicit UNKNOWN representation here instead of a fabricated value: empty
// string / empty vector / kUnknownXxx sentinel, never "" silently treated as valid.
//
// Field-shape model: src/serve/request_log.h's ServerLogEnvironment (device/UUID/compute-cap/driver
// fields). Deliberately NOT included here -- this header takes no dependency on serve/ or any CUDA
// header, so the standalone calibration test target and any future CPU-only consumer stay
// CUDA-free. Emission style follows bench/targets/qwen3_6_27b/ninfer_bench_support.cpp's hand-rolled
// ostringstream JSON (zero new dependencies; nlohmann is available elsewhere in the repo but not
// used by the bench/target layer this module belongs next to).
//
// Serialization shape: each JSONL line is a FLAT single-level JSON object -- every value is a
// string, number, or bool; multi-device/shape/transfer fields use indexed dotted keys
// (`device.0.uuid`, `device.1.uuid`, `shape.b`, ...) instead of nested objects or arrays. This
// keeps the parser a single-pass key:value scanner with no recursion, which is what keeps a
// hand-rolled parser trustworthy (see calibration.cpp).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::calibration {

// Bumped whenever a field's MEANING changes or a required field is added/removed. Parsing an
// unsupported schema_version, or a record whose schema_version key is absent, must fail loudly --
// never default-and-continue. V1 is immutable; V2 adds driver/runtime-managed transfer provenance
// tokens without reinterpreting V1 tokens.
inline constexpr int kSchemaVersionV1 = 1;
inline constexpr int kSchemaVersionV2 = 2;
inline constexpr int kCurrentSchemaVersion = kSchemaVersionV2;
inline constexpr int kSchemaVersion = kCurrentSchemaVersion;
inline constexpr std::string_view kArtifactType = "ninfer_calibration_record";

// ---------------------------------------------------------------------------------------------
// Storage policy (M5.4a doc sec 12 non-goal: do not create/populate data directories here; this
// is a path constant other tools point at, nothing under src/hybrid-test ever gets written).
// ---------------------------------------------------------------------------------------------

// Generated calibration data must never land under the source tree or become a committed
// artifact (a corpus of raw per-op latency samples is large, host/run-specific, and has no
// business in git history). Outside PROJECT_SOURCE_DIR entirely, matching the lab's existing
// `doc/`, `logs/`, `runs/` top-level convention of keeping generated material as repo-root
// siblings rather than nested inside a specific source checkout.
inline constexpr std::string_view kDefaultCalibrationRawDir       = "/media/storage/ninfer-lab/calibration/raw";
inline constexpr std::string_view kDefaultCalibrationSummariesDir = "/media/storage/ninfer-lab/calibration/summaries";

// ---------------------------------------------------------------------------------------------
// Identity / provenance
// ---------------------------------------------------------------------------------------------

// Distinguishes "we asked and it's genuinely unavailable" from "field left at a zero default and
// silently ignored" for every provenance/identity string field below.
inline constexpr std::string_view kUnknownString = "";
[[nodiscard]] inline bool is_unknown(std::string_view value) noexcept { return value.empty(); }

// Tri-state: default MUST be Unknown so a default-constructed record never silently claims a
// verified-clean source tree. Dirty and Unknown both disqualify qualified evidence, for distinct
// reasons (see missing_requirements).
enum class SourceState : std::uint8_t { Unknown = 0, Clean, Dirty };
[[nodiscard]] std::string_view source_state_name(SourceState state);
[[nodiscard]] SourceState source_state_from_name(std::string_view name);

struct SourceIdentity {
    std::string commit;                 // empty => unresolved. Does NOT force the ValidityState
                                        // enum (a caller can still set Valid); what IS enforced is
                                        // that the record can never be qualified calibration
                                        // evidence -- see is_qualified_calibration_evidence.
    SourceState state = SourceState::Unknown;
    bool commit_known() const noexcept { return !is_unknown(commit); }
};

struct BuildIdentity {
    std::string cuda_arch_list;         // e.g. "sm_86;sm_120a"; empty => unknown.
    std::string build_type;             // e.g. "Release"; empty => unknown.
    std::vector<std::string> compile_flags; // kernel-affecting flags only, caller's judgement.
    // Deterministic build fingerprint that M5.4b will populate live (collection is NOT implemented
    // here). Resolves the "empty compile_flags = unknown vs. legitimately no extra flags"
    // ambiguity: compile_flags may legitimately be empty for a build that adds nothing beyond
    // defaults, but build_id being empty always means "identity not established." Qualified
    // evidence requires this non-empty; compile_flags is never required non-empty.
    std::string build_id;
};

// Live artifact SHA256 is the REQUIRED KEY (sec 7); path/size are metadata, not identity.
struct ArtifactIdentity {
    std::string path;
    std::string name;
    std::string sha256;                 // empty => unknown/unhashed. Does not by itself constrain
                                        // the ValidityState enum; it does make the record
                                        // permanently ineligible as calibration evidence.
    std::string container_version;      // .ninfer container/format version, empty if unknown.
    std::string target_id;
    std::string weights_id;
    std::uint64_t size_bytes = 0;       // metadata only, never a substitute for sha256.

    [[nodiscard]] bool sha256_known() const noexcept { return !is_unknown(sha256); }
};

// One participating device. `logical_slot` is the placement-owner slot (0/1/...); `physical_index`
// is the raw CUDA device ordinal -- the two MUST be kept distinct because a slot can be remapped to
// a different physical index across processes/hosts (CUDA_VISIBLE_DEVICES, multi-GPU rigs).
struct DeviceIdentity {
    int logical_slot     = -1;          // -1 => not applicable / not yet assigned.
    int physical_index   = -1;          // -1 => unknown.
    std::string uuid;                   // empty => unknown; UUID is the exact-match key (sec 11).
    std::string model_name;             // e.g. "NVIDIA GeForce RTX 5060 Ti"; metadata only.
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::uint64_t vram_bytes     = 0;

    [[nodiscard]] bool uuid_known() const noexcept { return !is_unknown(uuid); }
};

struct SoftwareIdentity {
    std::string driver_version;
    std::string cuda_runtime_version;
    std::string cuda_toolkit_version;
    std::vector<std::string> library_versions; // "name=version" pairs, caller's judgement on which.
};

// Everything an observation needs to be exactly reproducible or exactly compared against another
// observation for staleness (sec 11). `devices` supports multiple participating devices (a compute
// record has one; a transfer record has two; a future multi-GPU op could have more).
struct CalibrationIdentity {
    int schema_version = kSchemaVersion;
    SourceIdentity source;
    BuildIdentity build;
    ArtifactIdentity artifact;
    std::vector<DeviceIdentity> devices;
    SoftwareIdentity software;

    // Optional audit-only metadata (sec 7: OPTIONAL METADATA, not a compatibility key).
    std::string branch_name;
    std::string hostname;
    std::string os_description;
    std::string command_line;
};

// ---------------------------------------------------------------------------------------------
// Operation identity
// ---------------------------------------------------------------------------------------------

// Deliberately a small closed set of coarse operation families rather than one enumerator per
// future kernel: GDN_PREFILL/GDN_DECODE/GDN_STATE and FULL_ATTENTION_PREFILL/DECODE collapse to
// Gdn/Attention here, distinguished instead by `phase` + free-text `route` -- the doc's own
// families list (sec 8) crosses family with phase/route, so hard-enumerating every combination as
// a top-level tag would multiply cases without adding information the (family, phase, route) tuple
// doesn't already carry, and would need a schema bump for every new op the runtime grows.
enum class OperationFamily : std::uint8_t {
    Unknown = 0,
    Gdn,
    Attention,
    Projection,
    Mlp,
    FinalNorm,
    LmHead,
    Transfer,
    Endpoint,
};
[[nodiscard]] std::string_view operation_family_name(OperationFamily family);
[[nodiscard]] OperationFamily operation_family_from_name(std::string_view name);

enum class ExecutionPhase : std::uint8_t {
    Unknown = 0,
    Prefill,
    Decode,
    StateReadUpdate, // GDN recurrent-state read/update, distinct from a prefill/decode compute pass.
    Endpoint,
    Transfer,
};
[[nodiscard]] std::string_view execution_phase_name(ExecutionPhase phase);
[[nodiscard]] ExecutionPhase execution_phase_from_name(std::string_view name);

enum class WarmState : std::uint8_t { Unknown = 0, Cold, Warm, Cached };
[[nodiscard]] std::string_view warm_state_name(WarmState state);
[[nodiscard]] WarmState warm_state_from_name(std::string_view name);

enum class ExecutionMode : std::uint8_t { Unknown = 0, Eager, Graph };
[[nodiscard]] std::string_view execution_mode_name(ExecutionMode mode);
[[nodiscard]] ExecutionMode execution_mode_from_name(std::string_view name);

// Free-form shape dimensions covering current and future op families (attention B/T/H, GDN
// B/T/H/K/V/chunk/state, projection/MLP rows/cols). Fields left at 0 mean "not applicable to this
// op", not "measured as zero" -- callers only set the dims a given family uses.
struct ShapeDims {
    std::uint64_t b = 0; // batch
    std::uint64_t t = 0; // tokens (decode width or prefill chunk width)
    std::uint64_t h = 0; // heads
    std::uint64_t k = 0; // GDN key dim
    std::uint64_t v = 0; // GDN value dim
    std::uint64_t chunk_size  = 0;
    std::uint64_t context_len = 0; // attention context bucket, when relevant
    std::uint64_t rows = 0; // projection/MLP input rows
    std::uint64_t cols = 0; // projection/MLP output cols / vocab / hidden
};

struct OperationIdentity {
    OperationFamily family = OperationFamily::Unknown;
    ExecutionPhase phase   = ExecutionPhase::Unknown;
    int device_logical_slot = -1;       // which placement slot executed this op.
    ShapeDims shape;
    std::string dtype;                  // e.g. "bf16", "int8"; empty => unknown.
    std::string quant_route;            // e.g. "groupwise-int8", "none"; empty => unknown. "none"
                                        // is the valid explicit value for an unquantized route --
                                        // only emptiness itself is rejected by the gate.
    ExecutionMode execution_mode = ExecutionMode::Unknown; // CUDA-graph vs eager, tri-state.
    WarmState warm_state = WarmState::Unknown;
    int endpoint_owner_slot = -1;       // -1 => not applicable (not an endpoint-dependent op).
};

// ---------------------------------------------------------------------------------------------
// Transfer identity
// ---------------------------------------------------------------------------------------------

enum class TransferDirection : std::uint8_t { Unknown = 0, DeviceToDevice, HostToDevice, DeviceToHost };
[[nodiscard]] std::string_view transfer_direction_name(TransferDirection direction);
[[nodiscard]] TransferDirection transfer_direction_from_name(std::string_view name);

// Extensible by design: payload BYTES and COLUMNS are data, current geometry (10,240 / 496,640
// bytes/column) is NOT a schema constant -- it belongs to whatever M5.4b run produced it. The class
// tag stays a short closed enum because sec 7/11 require exact-match on payload CLASS for
// staleness, and "extensible" is covered by Other + a free-text label rather than an open string
// key (which would make exact-match comparison order-sensitive against typos).
enum class TransferPayloadClass : std::uint8_t {
    Unknown = 0,
    LayerBoundaryHidden,
    EndpointHidden,
    NormalizedHidden,
    LogitsResult,
    Other,
};
[[nodiscard]] std::string_view transfer_payload_class_name(TransferPayloadClass payload_class);
[[nodiscard]] TransferPayloadClass transfer_payload_class_from_name(std::string_view name);

enum class HostInvolvement : std::uint8_t {
    Unknown = 0,
    None,
    PinnedStaging,
    PageableStaging,
    DriverManagedStaging,
};
[[nodiscard]] std::string_view host_involvement_name(HostInvolvement involvement);
[[nodiscard]] HostInvolvement host_involvement_from_name(std::string_view name);

enum class TransferSync : std::uint8_t { Unknown = 0, Sync, Async };
[[nodiscard]] std::string_view transfer_sync_name(TransferSync sync);
[[nodiscard]] TransferSync transfer_sync_from_name(std::string_view name);

enum class TransferRoute : std::uint8_t {
    Unknown = 0,
    CopyEngine,
    Kernel,
    RuntimeManagedMemcpy,
};
[[nodiscard]] std::string_view transfer_route_name(TransferRoute route);
[[nodiscard]] TransferRoute transfer_route_from_name(std::string_view name);

enum class TransferOverlap : std::uint8_t { Unknown = 0, NoOverlap, Overlap };
[[nodiscard]] std::string_view transfer_overlap_name(TransferOverlap overlap);
[[nodiscard]] TransferOverlap transfer_overlap_from_name(std::string_view name);

struct TransferIdentity {
    TransferDirection direction = TransferDirection::Unknown;
    std::string source_device_uuid;
    std::string destination_device_uuid;
    TransferPayloadClass payload_class = TransferPayloadClass::Unknown;
    std::string payload_class_label;    // required when payload_class == Other.
    std::uint64_t payload_bytes = 0;
    std::string transfer_api;           // e.g. "cudaMemcpyPeerAsync"; empty => unknown.
    TransferSync sync = TransferSync::Unknown;
    TransferRoute route = TransferRoute::Unknown;   // runtime-managed memcpy vs proven route.
    HostInvolvement host_involvement = HostInvolvement::Unknown;
    TransferOverlap overlap = TransferOverlap::Unknown; // overlapped with compute or not.
};

// ---------------------------------------------------------------------------------------------
// GDN identity
// ---------------------------------------------------------------------------------------------

enum class GdnStage : std::uint8_t {
    Unknown = 0,
    ChunkedPrefill,
    RecurrentDecode,
    StateRead,
    StateUpdate,
    Projection,
    ShortConv,
};
[[nodiscard]] std::string_view gdn_stage_name(GdnStage stage);
[[nodiscard]] GdnStage gdn_stage_from_name(std::string_view name);

struct GdnIdentity {
    GdnStage stage = GdnStage::Unknown;
    ShapeDims shape;                    // B/T/H/K/V/chunk_size populated as applicable.
    std::string state_dtype;            // empty => unknown / not applicable.
    std::string activation_dtype;
    bool requires_synchronization = false;
    bool initial_state = false;         // true => op reads/writes the sequence-initial state.
    bool final_state   = false;         // true => op reads/writes the sequence-final state.
    std::uint64_t workspace_bytes = 0;
};

// ---------------------------------------------------------------------------------------------
// Qualification state
// ---------------------------------------------------------------------------------------------

// Two independent axes, kept as two enums rather than one merged state (ambiguity flagged in the
// M5.4a report: the task brief's evidence-class list -- MEASURED/MEASURED_ANCHOR/MODELED/
// PROJECTED/STALE/UNRELIABLE/INVALID/UNKNOWN -- and doc sec 11's identity-validity list --
// VALID/PARTIAL/STALE/UNQUALIFIED/INVALID -- describe different things: how a value was obtained
// vs whether its identity/correctness still holds. A record can be MEASURED (obtained by direct
// timing) yet unqualified-as-evidence (missing source commit) at the same time; merging the lists
// enum would force a value for the axis that doesn't apply, or silently pick one meaning over the
// other. Two orthogonal enums keep both test (c) -- "unresolved commit yields the unqualified
// state" -- and test (d) -- "an INVALID/UNRELIABLE record stays retrievable but ineligible" --
// individually true without special-casing.
enum class EvidenceClass : std::uint8_t {
    Unknown = 0,       // not yet classified / collector has not run.
    Measured,          // direct timing observation.
    MeasuredAnchor,     // a MEASURED point promoted to reference status by later analysis.
    Modeled,           // derived from a model/formula, not a direct timing.
    Projected,         // extrapolated beyond directly measured shapes/devices.
};
[[nodiscard]] std::string_view evidence_class_name(EvidenceClass evidence_class);
[[nodiscard]] EvidenceClass evidence_class_from_name(std::string_view name);

// Mirrors doc sec 11 exactly (identity/staleness axis).
enum class ValidityState : std::uint8_t {
    Unqualified = 0, // missing identity, unresolved source, insufficient samples, unsupported
                     // route, failed correctness, or incomplete provenance (sec 11 default).
                     // This enum is a DECLARED status, not a self-enforcing one: nothing stops a
                     // caller setting Valid on an incomplete record. Provenance is enforced by
                     // missing_requirements/is_qualified_calibration_evidence, never by this value.
    Valid,           // required identity complete, route supported, correctness qualified,
                     // sufficient samples retained, key matches current runtime.
    Partial,         // valid only for the operation/shape/device families it covers.
    Stale,           // previously valid but no longer compatible with current identity.
    Invalid,         // known-wrong record or failed qualification. Terminal: never requalifiable.
    Unreliable,      // the measurement ran and correctness may even have passed, but the values
                     // are not trustworthy (high variance, thermal throttling, device contention).
                     // Distinct from Invalid: an unreliable record may be requalified later by
                     // recollection with more samples. Never eligible as prediction evidence.
};
[[nodiscard]] std::string_view validity_state_name(ValidityState state);
[[nodiscard]] ValidityState validity_state_from_name(std::string_view name);

enum class CorrectnessStatus : std::uint8_t { NotRun = 0, Pass, Fail, NotApplicable };
[[nodiscard]] std::string_view correctness_status_name(CorrectnessStatus status);
[[nodiscard]] CorrectnessStatus correctness_status_from_name(std::string_view name);

// NECESSARY-BUT-NOT-SUFFICIENT status check: validity/correctness alone say nothing about
// provenance completeness or evidence class. This is NOT the authoritative record eligibility
// gate -- callers deciding whether a record may be used as prediction evidence must call
// `is_qualified_calibration_evidence(const CalibrationRecord&)` instead, never this alone.
[[nodiscard]] inline bool status_allows_prediction(ValidityState validity,
                                                    CorrectnessStatus correctness) noexcept {
    if (validity != ValidityState::Valid && validity != ValidityState::Partial) { return false; }
    return correctness == CorrectnessStatus::Pass;
}

struct QualificationState {
    EvidenceClass evidence     = EvidenceClass::Unknown;
    ValidityState validity     = ValidityState::Unqualified;
    CorrectnessStatus correctness = CorrectnessStatus::NotRun;

    // NECESSARY-BUT-NOT-SUFFICIENT status check: NOT the authoritative record eligibility gate.
    // Use the free function `is_qualified_calibration_evidence(const CalibrationRecord&)` for
    // the real gate -- it also checks provenance completeness and evidence class.
    [[nodiscard]] bool status_allows_prediction() const noexcept {
        return ::ninfer::targets::calibration::status_allows_prediction(validity, correctness);
    }
};

// ---------------------------------------------------------------------------------------------
// Raw sample contract (FIRST-CLASS; sec 5 -- do not store only aggregates)
// ---------------------------------------------------------------------------------------------

struct RawSample {
    std::string run_id;                 // groups samples from one measurement run.
    std::string observation_id;         // identifies the (identity, operation) this sample belongs to.
    std::uint64_t sample_index = 0;     // 0-based position within the run, for ordering/dedup.
    double value = 0.0;
    std::string unit;                   // e.g. "us", "GB/s"; REQUIRED, never implied by convention.
    std::optional<std::uint64_t> timestamp_unix_ms;
    CorrectnessStatus qualification = CorrectnessStatus::NotRun;
};

// ---------------------------------------------------------------------------------------------
// Observation record: one JSONL line. Combines identity + one of {operation, transfer, gdn} + the
// raw sample + qualification state. Exactly one of operation/transfer/gdn is populated per record;
// which one is indicated by `record_kind` (avoids a discriminated union needing its own hand-rolled
// tagged serialization -- every field just gets a fixed key prefix and unused blocks serialize as
// their defaults, which parse back byte-identical since they're never referenced).
// ---------------------------------------------------------------------------------------------

enum class RecordKind : std::uint8_t { Operation = 0, Transfer, Gdn };
[[nodiscard]] std::string_view record_kind_name(RecordKind kind);
[[nodiscard]] RecordKind record_kind_from_name(std::string_view name);

struct CalibrationRecord {
    RecordKind kind = RecordKind::Operation;
    CalibrationIdentity identity;
    OperationIdentity operation;   // meaningful when kind == Operation or Gdn (op wraps a family).
    TransferIdentity transfer;     // meaningful when kind == Transfer.
    GdnIdentity gdn;               // meaningful when kind == Gdn.
    RawSample sample;
    QualificationState qualification;
};

// The identity-only subset of missing_requirements: source/build/artifact/device/software
// provenance checks that read solely from a CalibrationIdentity, with no dependency on any sample,
// measurement, or kind-specific requirement. Extracted so a live probe that collects identity but
// takes no measurement (e.g. ninfer_calibration_probe) can gate on provenance completeness without
// fabricating a fake sample just to call missing_requirements(). An empty result does NOT mean the
// identity is "qualified calibration evidence" -- a record still needs its sample/correctness
// requirements to satisfy that; it only means the identity block itself is complete.
[[nodiscard]] std::vector<std::string> missing_identity_requirements(
    const CalibrationIdentity& identity);

// Empty result == record carries the complete identity/provenance its kind requires.
[[nodiscard]] std::vector<std::string> missing_requirements(const CalibrationRecord& record);

// THE authoritative calibration-evidence gate. A caller-supplied validity of Valid cannot bypass
// provenance completeness or evidence class -- see calibration.cpp for the full rule.
[[nodiscard]] bool is_qualified_calibration_evidence(const CalibrationRecord& record);

// ---------------------------------------------------------------------------------------------
// Serialization: JSONL, one FLAT single-level JSON object per line (see file header). Deterministic
// field order/naming; explicit units; no hidden defaults that change measurement meaning; append-
// safe (one line per observation, no shared trailing structure); a single corrupt line does not
// invalidate other lines when read through `parse_corpus_line_by_line`.
// ---------------------------------------------------------------------------------------------

[[nodiscard]] std::string serialize(const CalibrationRecord& record);

// Thrown by `parse_line` on any malformed line OR whenever schema_version is absent/unsupported
// (fail loudly, never default-and-continue).
struct ParseError : std::runtime_error {
    explicit ParseError(const std::string& message) : std::runtime_error(message) {}
};

[[nodiscard]] CalibrationRecord parse_line(std::string_view line);

// Convenience for a whole corpus: returns successfully parsed records and separately reports
// (0-based line index, error message) for lines that failed to parse, so one corrupt line never
// discards the rest of the file. Blank lines are skipped, not reported as errors.
struct CorpusParseResult {
    std::vector<CalibrationRecord> records;
    std::vector<std::pair<std::size_t, std::string>> line_errors;
};
[[nodiscard]] CorpusParseResult parse_corpus(std::string_view jsonl_text);

} // namespace ninfer::targets::calibration

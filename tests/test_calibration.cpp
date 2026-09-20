// M5.4a: host-only contract test for the calibration data schema (types.h data only, no CUDA, no
// artifact, no device). Covers the four required cases from the M5.4a checkpoint brief:
//   (a) serialize -> parse round trip is lossless and field-order deterministic
//   (b) unsupported schema_version (and schema_version absent) fails loudly
//   (c) missing/unresolved source commit yields the unqualified validity state
//   (d) an INVALID/UNRELIABLE record is retained but reports as ineligible for prediction
#include "targets/calibration.h"

#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

using namespace ninfer::targets::calibration;

CalibrationRecord make_sample_record() {
    CalibrationRecord record;
    record.kind                       = RecordKind::Operation;
    record.identity.schema_version    = kSchemaVersion;
    record.identity.source.commit     = "78d2b4d69fe35e61d151eb6aa189e5e366f3da7c";
    record.identity.source.state      = SourceState::Clean;
    record.identity.build.cuda_arch_list = "sm_86;sm_120a";
    record.identity.build.build_type     = "Release";
    record.identity.build.compile_flags  = {"-O3", "-DNDEBUG"};
    record.identity.build.build_id       = "build-fingerprint-0001";
    record.identity.artifact.path          = "/media/storage/artifacts/qwen3.8-27b.ninfer";
    record.identity.artifact.name          = "qwen3.8-27b";
    record.identity.artifact.sha256        = "deadbeef";
    record.identity.artifact.target_id     = "qwen3_6_27b";
    record.identity.artifact.weights_id    = "groupwise-int8";
    record.identity.artifact.size_bytes    = 27000000000ULL;

    DeviceIdentity gpu0;
    gpu0.logical_slot   = 0;
    gpu0.physical_index = 0;
    gpu0.uuid            = "GPU-aaaa0000";
    gpu0.model_name       = "NVIDIA GeForce RTX 5060 Ti";
    gpu0.compute_capability_major = 12;
    gpu0.compute_capability_minor = 0;
    gpu0.vram_bytes       = 16ULL * 1024 * 1024 * 1024;
    DeviceIdentity gpu1;
    gpu1.logical_slot   = 1;
    gpu1.physical_index = 1;
    gpu1.uuid            = "GPU-bbbb1111";
    gpu1.model_name       = "NVIDIA GeForce RTX 3060";
    gpu1.compute_capability_major = 8;
    gpu1.compute_capability_minor = 6;
    gpu1.vram_bytes       = 12ULL * 1024 * 1024 * 1024;
    record.identity.devices = {gpu0, gpu1};

    record.identity.software.driver_version      = "550.90.07";
    record.identity.software.cuda_runtime_version = "12.4";
    record.identity.software.cuda_toolkit_version = "12.4.131";
    record.identity.software.library_versions     = {"cudnn=9.1"};
    record.identity.branch_name    = "phase2-layer-parity-debug";
    record.identity.hostname       = "ninfer-lab";
    record.identity.os_description = "Linux 7.0.0-31-generic";
    record.identity.command_line   = "ninfer_bench --artifact x --n-prompt 512";

    record.operation.family = OperationFamily::Gdn;
    record.operation.phase  = ExecutionPhase::Decode;
    record.operation.device_logical_slot = 0;
    record.operation.shape.b = 1;
    record.operation.shape.t = 1;
    record.operation.shape.h = 16;
    record.operation.shape.k = 128;
    record.operation.shape.v = 128;
    record.operation.shape.chunk_size = 64;
    record.operation.dtype       = "bf16";
    record.operation.quant_route = "groupwise-int8";
    record.operation.execution_mode = ExecutionMode::Graph;
    record.operation.warm_state  = WarmState::Warm;
    record.operation.endpoint_owner_slot = -1;

    record.gdn.stage = GdnStage::RecurrentDecode;
    record.gdn.shape = record.operation.shape;
    record.gdn.state_dtype      = "fp32";
    record.gdn.activation_dtype = "bf16";
    record.gdn.requires_synchronization = true;
    record.gdn.final_state = true;

    record.sample.run_id         = "run-2026-09-19-001";
    record.sample.observation_id = "gdn_decode/slot0/warm/graph";
    record.sample.sample_index   = 42;
    // Value deliberately needs full double precision to make the round-trip test meaningful.
    record.sample.value = 1234.5678901234567;
    record.sample.unit  = "us";
    record.sample.timestamp_unix_ms = 1758000000000ULL;
    record.sample.qualification = CorrectnessStatus::Pass;

    record.qualification.evidence    = EvidenceClass::Measured;
    record.qualification.validity    = ValidityState::Valid;
    record.qualification.correctness = CorrectnessStatus::Pass;
    return record;
}

} // namespace

int main() {
    // (a) serialize -> parse -> serialize round trip is byte-identical: proves losslessness,
    // field-order determinism, and full double precision in one comparison.
    {
        const CalibrationRecord original = make_sample_record();
        const std::string first_pass     = serialize(original);
        const CalibrationRecord parsed   = parse_line(first_pass);
        const std::string second_pass    = serialize(parsed);
        check(first_pass == second_pass, "serialize->parse->serialize is not byte-identical");
        check(parsed.identity.source.commit == original.identity.source.commit,
              "source commit did not round trip");
        check(parsed.identity.devices.size() == 2, "device count did not round trip");
        check(parsed.identity.devices[1].uuid == "GPU-bbbb1111",
              "second device uuid did not round trip");
        check(parsed.sample.value == original.sample.value,
              "raw sample value lost precision across round trip");
        check(parsed.sample.timestamp_unix_ms.has_value() &&
                  *parsed.sample.timestamp_unix_ms == *original.sample.timestamp_unix_ms,
              "optional timestamp did not round trip");
        check(parsed.gdn.stage == GdnStage::RecurrentDecode, "gdn stage did not round trip");
        check(parsed.qualification.validity == ValidityState::Valid,
              "qualification validity did not round trip");
    }

    // (b) unsupported schema_version, and schema_version entirely absent, both fail loudly rather
    // than defaulting to the current version and silently continuing.
    {
        CalibrationRecord record = make_sample_record();
        std::string line = serialize(record);
        const std::string needle = "\"schema_version\":2";
        const std::size_t at = line.find(needle);
        check(at != std::string::npos, "schema_version needle not found");
        line.replace(at, needle.size(), "\"schema_version\":3");
        bool threw = false;
        try {
            (void)parse_line(line);
        } catch (const ParseError&) { threw = true; }
        check(threw, "schema_version=3 did not throw ParseError");

        const std::string no_version_line = "{\"artifact_type\":\"x\",\"record_kind\":\"operation\"}";
        bool threw_missing                = false;
        try {
            (void)parse_line(no_version_line);
        } catch (const ParseError&) { threw_missing = true; }
        check(threw_missing, "missing schema_version did not throw ParseError");
    }

    // (b2) v1 remains parseable and keeps old transfer enum semantics, but v2-only transfer
    // tokens cannot masquerade as v1.
    {
        CalibrationRecord v1 = make_sample_record();
        v1.kind = RecordKind::Transfer;
        v1.identity.schema_version = kSchemaVersionV1;
        v1.operation.phase = ExecutionPhase::Transfer;
        v1.transfer.direction = TransferDirection::DeviceToDevice;
        v1.transfer.payload_class = TransferPayloadClass::LogitsResult;
        v1.transfer.payload_bytes = 4096;
        v1.transfer.transfer_api = "cudaMemcpyPeerAsync";
        v1.transfer.sync = TransferSync::Async;
        v1.transfer.route = TransferRoute::CopyEngine;
        v1.transfer.host_involvement = HostInvolvement::None;
        v1.transfer.overlap = TransferOverlap::Overlap;
        v1.transfer.source_device_uuid = "GPU-aaaa0000";
        v1.transfer.destination_device_uuid = "GPU-bbbb1111";

        const std::string v1_line = serialize(v1);
        const CalibrationRecord parsed_v1 = parse_line(v1_line);
        check(parsed_v1.identity.schema_version == kSchemaVersionV1,
              "v1 record was not preserved as schema_version=1");
        check(parsed_v1.transfer.route == TransferRoute::CopyEngine,
              "v1 copy_engine semantics did not round trip");
        check(parsed_v1.transfer.host_involvement == HostInvolvement::None,
              "v1 host_involvement semantics did not round trip");

        CalibrationRecord mislabeled = v1;
        mislabeled.transfer.route = TransferRoute::RuntimeManagedMemcpy;
        bool serialize_threw = false;
        try {
            (void)serialize(mislabeled);
        } catch (const ParseError&) { serialize_threw = true; }
        check(serialize_threw, "serializer allowed v2-only transfer_route in schema v1");
        check(!is_qualified_calibration_evidence(mislabeled),
              "in-memory v1 record with v2-only route incorrectly qualified");

        std::string bad_line = v1_line;
        const std::string route_needle = "\"transfer_route\":\"copy_engine\"";
        const std::size_t route_at = bad_line.find(route_needle);
        check(route_at != std::string::npos, "v1 transfer_route needle not found");
        bad_line.replace(route_at, route_needle.size(),
                         "\"transfer_route\":\"runtime_managed_memcpy\"");
        bool parse_threw = false;
        try {
            (void)parse_line(bad_line);
        } catch (const ParseError&) { parse_threw = true; }
        check(parse_threw, "parser allowed v2-only transfer_route token in schema v1");
    }

    // (c) missing/unresolved source commit yields the UNQUALIFIED validity state -- QualificationState
    // defaults to Unqualified, and SourceIdentity::commit_known() makes the missing-commit condition
    // explicit for a caller building the record.
    {
        CalibrationRecord record = make_sample_record();
        record.identity.source.commit = kUnknownString;
        check(!record.identity.source.commit_known(), "commit_known() did not report unresolved commit");
        QualificationState qualification; // default-constructed: no explicit classification yet.
        check(qualification.validity == ValidityState::Unqualified,
              "default QualificationState validity is not Unqualified");
        record.qualification = qualification;
        const std::string line  = serialize(record);
        const CalibrationRecord parsed = parse_line(line);
        check(parsed.qualification.validity == ValidityState::Unqualified,
              "unqualified validity state did not round trip");
    }

    // (d) an INVALID/UNRELIABLE record is retained (parses fine, all fields readable) but reports
    // as ineligible for prediction via the explicit predicate -- ineligibility is never implied.
    {
        CalibrationRecord record            = make_sample_record();
        record.qualification.validity       = ValidityState::Invalid;
        record.qualification.correctness    = CorrectnessStatus::Fail;
        const std::string line              = serialize(record);
        const CalibrationRecord parsed      = parse_line(line);
        check(parsed.sample.value == record.sample.value,
              "invalid record's raw sample was not retained for forensics");
        check(!parsed.qualification.status_allows_prediction(),
              "invalid record incorrectly reported status_allows_prediction()");
        check(!status_allows_prediction(ValidityState::Invalid, CorrectnessStatus::Pass),
              "free predicate did not reject Invalid validity");
        check(!status_allows_prediction(ValidityState::Valid, CorrectnessStatus::Fail),
              "free predicate did not reject Fail correctness");
        check(status_allows_prediction(ValidityState::Valid, CorrectnessStatus::Pass),
              "free predicate rejected a genuinely eligible Valid+Pass combination");
    }

    // (e) UNRELIABLE is a distinct state, not a synonym for INVALID: the measurement ran and
    // correctness passed, but the values are not trustworthy. It must round trip under its own
    // name, stay readable for forensics/requalification, and still be ineligible as evidence.
    {
        CalibrationRecord record         = make_sample_record();
        record.qualification.validity    = ValidityState::Unreliable;
        record.qualification.correctness = CorrectnessStatus::Pass;
        const std::string line           = serialize(record);
        const CalibrationRecord parsed   = parse_line(line);
        check(parsed.qualification.validity == ValidityState::Unreliable,
              "unreliable validity state did not round trip");
        check(parsed.qualification.validity != ValidityState::Invalid,
              "unreliable record collapsed into the invalid state");
        check(parsed.sample.value == record.sample.value,
              "unreliable record's raw sample was not retained for requalification");
        check(!parsed.qualification.status_allows_prediction(),
              "unreliable record incorrectly reported status_allows_prediction()");
        check(validity_state_from_name("unreliable") == ValidityState::Unreliable,
              "unreliable name did not map back to the Unreliable state");
    }

    // Corpus-level: one corrupt line must not destroy the rest of the file.
    {
        const CalibrationRecord good = make_sample_record();
        const std::string corpus_text =
            serialize(good) + "\n" + "not even json\n" + "\n" + serialize(good) + "\n";
        const CorpusParseResult result = parse_corpus(corpus_text);
        check(result.records.size() == 2, "corpus parse did not retain both valid lines");
        check(result.line_errors.size() == 1, "corpus parse did not report exactly one corrupt line");
        if (!result.line_errors.empty()) {
            check(result.line_errors[0].first == 1, "corrupt line index was not reported correctly");
        }
    }

    // (1) Valid+Pass but missing provenance => NOT is_qualified_calibration_evidence, even though
    // the status predicate alone says yes. Built directly in C++ (not via a stripped JSON line --
    // that line would now be a ParseError, a separate concern from semantic ineligibility).
    {
        CalibrationRecord record = make_sample_record();
        record.identity.source.commit     = "";
        record.identity.artifact.sha256   = "";
        record.identity.devices.clear();
        check(record.qualification.status_allows_prediction(),
              "status axis alone should report eligible (proving it is insufficient)");
        check(!is_qualified_calibration_evidence(record),
              "record missing provenance incorrectly qualified as calibration evidence");
    }

    // (2) Evidence-class gate: Modeled/Projected/Unknown must never masquerade as raw measurements.
    {
        CalibrationRecord record = make_sample_record();
        check(is_qualified_calibration_evidence(record), "fully populated Measured record should qualify");
        record.qualification.evidence = EvidenceClass::MeasuredAnchor;
        check(is_qualified_calibration_evidence(record), "MeasuredAnchor should qualify");
        record.qualification.evidence = EvidenceClass::Modeled;
        check(!is_qualified_calibration_evidence(record), "Modeled evidence must not qualify");
        record.qualification.evidence = EvidenceClass::Projected;
        check(!is_qualified_calibration_evidence(record), "Projected evidence must not qualify");
        record.qualification.evidence = EvidenceClass::Unknown;
        check(!is_qualified_calibration_evidence(record), "Unknown evidence must not qualify");
    }

    // (3) Mechanical required-key coverage: default record's serialized line is structurally valid;
    // removing any single top-level key must throw ParseError. Generic over the actual key list so
    // it can't silently drift from the serializer.
    {
        const std::string line = serialize(CalibrationRecord{});
        bool default_ok         = true;
        try {
            (void)parse_line(line);
        } catch (const ParseError&) { default_ok = false; }
        check(default_ok, "serialize(CalibrationRecord{}) must parse successfully");

        // Walk the flat "key":value pairs and collect their [start, end) spans (including one
        // adjacent comma) so each can be excised independently.
        std::vector<std::pair<std::size_t, std::size_t>> spans; // [start, end)
        std::size_t pos = 1; // past '{'
        while (pos < line.size() && line[pos] != '}') {
            const std::size_t pair_start = pos;
            ++pos; // past opening quote of key
            while (line[pos] != '"') { ++pos; }
            ++pos; // past closing quote
            ++pos; // past ':'
            if (line[pos] == '"') {
                ++pos;
                while (line[pos] != '"') { ++pos; }
                ++pos;
            } else {
                while (pos < line.size() && line[pos] != ',' && line[pos] != '}') { ++pos; }
            }
            std::size_t pair_end = pos;
            if (pair_end < line.size() && line[pair_end] == ',') { ++pair_end; } // consume trailing comma
            spans.emplace_back(pair_start, pair_end);
            pos = pair_end;
        }

        int removal_failures = 0;
        for (const auto& [start, end] : spans) {
            std::string mutated = line.substr(0, start) + line.substr(end);
            // If removing the trailing comma-consuming span left a dangling leading comma right
            // after '{', strip it so we test presence-enforcement, not a comma-syntax artifact.
            if (mutated.size() > 1 && mutated[1] == ',') { mutated.erase(1, 1); }
            bool threw = false;
            try {
                (void)parse_line(mutated);
            } catch (const ParseError&) { threw = true; }
            if (!threw) { ++removal_failures; }
        }
        check(removal_failures == 0,
              "removing a required top-level key did not throw ParseError for " +
                  std::to_string(removal_failures) + " key(s)");
    }

    // (4) wrong artifact_type => ParseError.
    {
        CalibrationRecord record = make_sample_record();
        std::string line         = serialize(record);
        const std::string needle = "\"artifact_type\":\"" + std::string(kArtifactType) + '"';
        const std::size_t at     = line.find(needle);
        check(at != std::string::npos, "artifact_type key not found in serialized line");
        line.replace(at, needle.size(), "\"artifact_type\":\"something_else\"");
        bool threw = false;
        try {
            (void)parse_line(line);
        } catch (const ParseError&) { threw = true; }
        check(threw, "wrong artifact_type did not throw ParseError");
    }

    // (5)-(8): malformed literal / unknown-enum tokens each throw ParseError.
    {
        CalibrationRecord record = make_sample_record();
        const std::string base   = serialize(record);

        auto replace_one = [&](const std::string& needle, const std::string& replacement) {
            std::string line     = base;
            const std::size_t at = line.find(needle);
            check(at != std::string::npos, "needle not found: " + needle);
            line.replace(at, needle.size(), replacement);
            return line;
        };
        auto expect_throw = [&](const std::string& line, const std::string& what) {
            bool threw = false;
            try {
                (void)parse_line(line);
            } catch (const ParseError&) { threw = true; }
            check(threw, what);
        };

        expect_throw(replace_one("\"sample_index\":42", "\"sample_index\":12abc"),
                     "malformed integer did not throw ParseError");
        expect_throw(replace_one("\"gdn_requires_synchronization\":true",
                                  "\"gdn_requires_synchronization\":yes"),
                     "malformed boolean did not throw ParseError");
        expect_throw(replace_one("\"sample_value\":1234.5678901234567", "\"sample_value\":1.2.3"),
                     "malformed real did not throw ParseError");
        expect_throw(replace_one("\"record_kind\":\"operation\"", "\"record_kind\":\"bogus\""),
                     "unknown record_kind token did not throw ParseError");
        expect_throw(replace_one("\"validity_state\":\"valid\"", "\"validity_state\":\"bogus\""),
                     "unknown validity_state token did not throw ParseError");
    }

    // (9) Trailing garbage after '}' throws; a trailing '\r' (CRLF) still parses fine.
    {
        const std::string line = serialize(make_sample_record());
        bool threw              = false;
        try {
            (void)parse_line(line + " trailing garbage");
        } catch (const ParseError&) { threw = true; }
        check(threw, "trailing garbage after closing brace did not throw ParseError");

        bool crlf_ok = true;
        try {
            (void)parse_line(line + "\r");
        } catch (const ParseError&) { crlf_ok = false; }
        check(crlf_ok, "trailing \\r (CRLF) must still parse fine");
    }

    // (10) Explicit "unknown" token where the enum defines it: parses fine, record stays
    // unqualified/ineligible.
    {
        CalibrationRecord record      = make_sample_record();
        record.qualification.evidence = EvidenceClass::Unknown;
        const std::string line        = serialize(record);
        const CalibrationRecord parsed = parse_line(line);
        check(parsed.qualification.evidence == EvidenceClass::Unknown,
              "explicit unknown evidence_class token did not round trip");
        check(!is_qualified_calibration_evidence(parsed),
              "unknown-evidence record incorrectly qualified as calibration evidence");
    }

    // (11) Unreliable is distinct from Invalid: names differ, both round trip, both ineligible.
    {
        check(validity_state_name(ValidityState::Unreliable) !=
                  validity_state_name(ValidityState::Invalid),
              "Unreliable and Invalid must have distinct names");
        check(validity_state_from_name("unreliable") == ValidityState::Unreliable,
              "unreliable name did not round trip");
        check(validity_state_from_name("invalid") == ValidityState::Invalid,
              "invalid name did not round trip");
        check(!status_allows_prediction(ValidityState::Unreliable, CorrectnessStatus::Pass),
              "Unreliable must remain prediction-ineligible");
        check(!status_allows_prediction(ValidityState::Invalid, CorrectnessStatus::Pass),
              "Invalid must remain prediction-ineligible");
    }

    // (12) Fully populated Measured+Valid+Pass record with complete provenance IS qualified.
    {
        const CalibrationRecord record = make_sample_record();
        const std::vector<std::string> missing = missing_requirements(record);
        std::string joined;
        for (const std::string& reason : missing) { joined += reason + "; "; }
        check(missing.empty(), "make_sample_record() unexpectedly has missing requirements: " + joined);
        check(is_qualified_calibration_evidence(record),
              "fully populated sample record should be qualified calibration evidence");
    }

    // (13) Indexed validation: device_count declaring more devices than present, and a negative
    // count, both throw ParseError.
    {
        CalibrationRecord record = make_sample_record();
        const std::string base   = serialize(record);

        std::string over_count = base;
        const std::string needle = "\"device_count\":2";
        const std::size_t at     = over_count.find(needle);
        check(at != std::string::npos, "device_count key not found in serialized line");
        over_count.replace(at, needle.size(), "\"device_count\":3");
        bool threw_over = false;
        try {
            (void)parse_line(over_count);
        } catch (const ParseError&) { threw_over = true; }
        check(threw_over, "device_count declaring more devices than present did not throw");

        std::string negative_count = base;
        negative_count.replace(negative_count.find(needle), needle.size(), "\"device_count\":-1");
        bool threw_negative = false;
        try {
            (void)parse_line(negative_count);
        } catch (const ParseError&) { threw_negative = true; }
        check(threw_negative, "negative device_count did not throw ParseError");
    }

    // (14) Transfer-kind and Gdn-kind records each exercise their kind-specific requirements.
    {
        CalibrationRecord transfer_record;
        transfer_record.kind                                = RecordKind::Transfer;
        transfer_record.identity                             = make_sample_record().identity;
        transfer_record.operation.phase                      = ExecutionPhase::Transfer;
        transfer_record.sample                               = make_sample_record().sample;
        transfer_record.qualification                        = make_sample_record().qualification;
        transfer_record.transfer.direction                   = TransferDirection::DeviceToDevice;
        transfer_record.transfer.payload_class               = TransferPayloadClass::LogitsResult;
        transfer_record.transfer.payload_bytes               = 4096;
        transfer_record.transfer.transfer_api                = "cudaMemcpyPeerAsync";
        transfer_record.transfer.host_involvement            = HostInvolvement::None;
        transfer_record.transfer.source_device_uuid          = "GPU-aaaa0000";
        transfer_record.transfer.destination_device_uuid     = "GPU-bbbb1111";
        transfer_record.transfer.sync                        = TransferSync::Async;
        transfer_record.transfer.route                       = TransferRoute::CopyEngine;
        transfer_record.transfer.overlap                     = TransferOverlap::Overlap;
        check(is_qualified_calibration_evidence(transfer_record),
              "complete Transfer record should qualify");

        CalibrationRecord incomplete_transfer         = transfer_record;
        incomplete_transfer.transfer.transfer_api     = "";
        check(!is_qualified_calibration_evidence(incomplete_transfer),
              "Transfer record missing transfer_api should not qualify");

        // operation.phase is a common-core requirement for every record kind, not just Operation --
        // a Transfer record must not qualify while it is left Unknown (M5.4c-2 transfer-phase fix).
        CalibrationRecord transfer_unresolved_phase   = transfer_record;
        transfer_unresolved_phase.operation.phase     = ExecutionPhase::Unknown;
        check(!is_qualified_calibration_evidence(transfer_unresolved_phase),
              "Transfer record with unresolved operation_phase should not qualify");
        const std::vector<std::string> phase_reasons = missing_requirements(transfer_unresolved_phase);
        bool phase_reason_present = false;
        for (const std::string& reason : phase_reasons) {
            if (reason == "operation_phase unresolved") { phase_reason_present = true; }
        }
        check(phase_reason_present,
              "unresolved operation_phase on a Transfer record must be reported by missing_requirements");

        CalibrationRecord gdn_record                  = make_sample_record();
        gdn_record.kind                                = RecordKind::Gdn;
        check(is_qualified_calibration_evidence(gdn_record), "complete Gdn record should qualify");

        CalibrationRecord incomplete_gdn               = gdn_record;
        incomplete_gdn.gdn.activation_dtype            = "";
        check(!is_qualified_calibration_evidence(incomplete_gdn),
              "Gdn record missing activation_dtype should not qualify");
    }

    // (14b) Schema v2 can honestly represent the current driver/runtime-managed memcpy route.
    {
        CalibrationRecord transfer_record;
        transfer_record.kind                                = RecordKind::Transfer;
        transfer_record.identity                             = make_sample_record().identity;
        transfer_record.operation.phase                      = ExecutionPhase::Transfer;
        transfer_record.sample                               = make_sample_record().sample;
        transfer_record.qualification                        = make_sample_record().qualification;
        transfer_record.transfer.direction                   = TransferDirection::DeviceToDevice;
        transfer_record.transfer.payload_class               = TransferPayloadClass::LogitsResult;
        transfer_record.transfer.payload_bytes               = 4096;
        transfer_record.transfer.transfer_api =
            "cudaMemcpyAsync(cudaMemcpyDeviceToDevice,UVA)";
        transfer_record.transfer.host_involvement            = HostInvolvement::DriverManagedStaging;
        transfer_record.transfer.source_device_uuid          = "GPU-aaaa0000";
        transfer_record.transfer.destination_device_uuid     = "GPU-bbbb1111";
        transfer_record.transfer.sync                        = TransferSync::Async;
        transfer_record.transfer.route                       = TransferRoute::RuntimeManagedMemcpy;
        transfer_record.transfer.overlap                     = TransferOverlap::NoOverlap;
        check(is_qualified_calibration_evidence(transfer_record),
              "complete v2 driver-managed Transfer record should qualify");

        const std::string line = serialize(transfer_record);
        const CalibrationRecord parsed = parse_line(line);
        check(parsed.identity.schema_version == kSchemaVersionV2,
              "v2 transfer record did not serialize as schema v2");
        check(parsed.transfer.host_involvement == HostInvolvement::DriverManagedStaging,
              "driver_managed_staging did not round trip");
        check(parsed.transfer.route == TransferRoute::RuntimeManagedMemcpy,
              "runtime_managed_memcpy did not round trip");

        CalibrationRecord unknown_route = transfer_record;
        unknown_route.transfer.route = TransferRoute::Unknown;
        check(!is_qualified_calibration_evidence(unknown_route),
              "v2 transfer with unknown route should not qualify");
    }

    // (15) Focused negative-gate block: starting from a fully qualified good record, each ISOLATED
    // single-field mutation (applied to a fresh copy) must make the record NOT qualify. Table-
    // driven rather than 25 copy-pasted blocks.
    {
        const CalibrationRecord good = make_sample_record();
        check(is_qualified_calibration_evidence(good),
              "baseline good record for negative-gate block should qualify");

        struct Mutation {
            std::string label;
            std::function<void(CalibrationRecord&)> mutate;
        };
        const std::vector<Mutation> mutations = {
            {"unknown source state",
             [](CalibrationRecord& r) { r.identity.source.state = SourceState::Unknown; }},
            {"dirty source", [](CalibrationRecord& r) { r.identity.source.state = SourceState::Dirty; }},
            {"missing build_id", [](CalibrationRecord& r) { r.identity.build.build_id.clear(); }},
            {"missing cuda_toolkit_version",
             [](CalibrationRecord& r) { r.identity.software.cuda_toolkit_version.clear(); }},
            {"missing device model_name",
             [](CalibrationRecord& r) { r.identity.devices[0].model_name.clear(); }},
            {"zero device vram_bytes",
             [](CalibrationRecord& r) { r.identity.devices[0].vram_bytes = 0; }},
            {"zero compute_capability_major",
             [](CalibrationRecord& r) { r.identity.devices[0].compute_capability_major = 0; }},
            {"negative device physical_index",
             [](CalibrationRecord& r) { r.identity.devices[0].physical_index = -1; }},
            {"operation device slot not in inventory",
             [](CalibrationRecord& r) { r.operation.device_logical_slot = 7; }},
            {"unknown execution mode",
             [](CalibrationRecord& r) { r.operation.execution_mode = ExecutionMode::Unknown; }},
            {"unknown warm state",
             [](CalibrationRecord& r) { r.operation.warm_state = WarmState::Unknown; }},
            {"empty quant route", [](CalibrationRecord& r) { r.operation.quant_route.clear(); }},
            {"all-zero required operation shape",
             [](CalibrationRecord& r) { r.operation.shape = ShapeDims{}; }},
            {"sample qualification Fail",
             [](CalibrationRecord& r) { r.sample.qualification = CorrectnessStatus::Fail; }},
            {"sample qualification NotRun",
             [](CalibrationRecord& r) { r.sample.qualification = CorrectnessStatus::NotRun; }},
            {"missing timestamp",
             [](CalibrationRecord& r) { r.sample.timestamp_unix_ms.reset(); }},
            {"NaN sample value",
             [](CalibrationRecord& r) { r.sample.value = std::numeric_limits<double>::quiet_NaN(); }},
            {"+Inf sample value",
             [](CalibrationRecord& r) { r.sample.value = std::numeric_limits<double>::infinity(); }},
            {"-Inf sample value",
             [](CalibrationRecord& r) { r.sample.value = -std::numeric_limits<double>::infinity(); }},
            {"transfer source UUID not in devices",
             [](CalibrationRecord& r) {
                 r.kind                             = RecordKind::Transfer;
                 r.transfer.direction               = TransferDirection::DeviceToDevice;
                 r.transfer.payload_class           = TransferPayloadClass::LogitsResult;
                 r.transfer.payload_bytes            = 4096;
                 r.transfer.transfer_api             = "cudaMemcpyPeerAsync";
                 r.transfer.host_involvement         = HostInvolvement::None;
                 r.transfer.sync                     = TransferSync::Async;
                 r.transfer.route                    = TransferRoute::CopyEngine;
                 r.transfer.overlap                  = TransferOverlap::Overlap;
                 r.transfer.source_device_uuid       = "GPU-not-in-inventory";
                 r.transfer.destination_device_uuid  = "GPU-bbbb1111";
             }},
            {"transfer destination UUID not in devices",
             [](CalibrationRecord& r) {
                 r.kind                             = RecordKind::Transfer;
                 r.transfer.direction               = TransferDirection::DeviceToDevice;
                 r.transfer.payload_class           = TransferPayloadClass::LogitsResult;
                 r.transfer.payload_bytes            = 4096;
                 r.transfer.transfer_api             = "cudaMemcpyPeerAsync";
                 r.transfer.host_involvement         = HostInvolvement::None;
                 r.transfer.sync                     = TransferSync::Async;
                 r.transfer.route                    = TransferRoute::CopyEngine;
                 r.transfer.overlap                  = TransferOverlap::Overlap;
                 r.transfer.source_device_uuid       = "GPU-aaaa0000";
                 r.transfer.destination_device_uuid  = "GPU-not-in-inventory";
             }},
            {"unknown transfer sync",
             [](CalibrationRecord& r) {
                 r.kind                             = RecordKind::Transfer;
                 r.transfer.direction               = TransferDirection::DeviceToDevice;
                 r.transfer.payload_class           = TransferPayloadClass::LogitsResult;
                 r.transfer.payload_bytes            = 4096;
                 r.transfer.transfer_api             = "cudaMemcpyPeerAsync";
                 r.transfer.host_involvement         = HostInvolvement::None;
                 r.transfer.route                    = TransferRoute::CopyEngine;
                 r.transfer.overlap                  = TransferOverlap::Overlap;
                 r.transfer.source_device_uuid       = "GPU-aaaa0000";
                 r.transfer.destination_device_uuid  = "GPU-bbbb1111";
                 r.transfer.sync                     = TransferSync::Unknown;
             }},
            {"unknown transfer route",
             [](CalibrationRecord& r) {
                 r.kind                             = RecordKind::Transfer;
                 r.transfer.direction               = TransferDirection::DeviceToDevice;
                 r.transfer.payload_class           = TransferPayloadClass::LogitsResult;
                 r.transfer.payload_bytes            = 4096;
                 r.transfer.transfer_api             = "cudaMemcpyPeerAsync";
                 r.transfer.host_involvement         = HostInvolvement::None;
                 r.transfer.sync                     = TransferSync::Async;
                 r.transfer.overlap                  = TransferOverlap::Overlap;
                 r.transfer.source_device_uuid       = "GPU-aaaa0000";
                 r.transfer.destination_device_uuid  = "GPU-bbbb1111";
                 r.transfer.route                    = TransferRoute::Unknown;
             }},
            {"unknown transfer overlap",
             [](CalibrationRecord& r) {
                 r.kind                             = RecordKind::Transfer;
                 r.transfer.direction               = TransferDirection::DeviceToDevice;
                 r.transfer.payload_class           = TransferPayloadClass::LogitsResult;
                 r.transfer.payload_bytes            = 4096;
                 r.transfer.transfer_api             = "cudaMemcpyPeerAsync";
                 r.transfer.host_involvement         = HostInvolvement::None;
                 r.transfer.sync                     = TransferSync::Async;
                 r.transfer.route                    = TransferRoute::CopyEngine;
                 r.transfer.source_device_uuid       = "GPU-aaaa0000";
                 r.transfer.destination_device_uuid  = "GPU-bbbb1111";
                 r.transfer.overlap                  = TransferOverlap::Unknown;
             }},
            {"GDN state-accessing stage with empty state_dtype",
             [](CalibrationRecord& r) {
                 r.kind                  = RecordKind::Gdn;
                 r.gdn.stage             = GdnStage::RecurrentDecode;
                 r.gdn.state_dtype.clear();
             }},
        };

        for (const Mutation& mutation : mutations) {
            CalibrationRecord mutated = good;
            mutation.mutate(mutated);
            // NOTE: not round-tripped through serialize()/parse_line() -- put_num now refuses
            // non-finite values, so the NaN/Inf mutations would throw before the gate even runs.
            // Gate testing here is purely in-memory; parser-side rejection is covered separately.
            check(!is_qualified_calibration_evidence(mutated),
                  "mutation '" + mutation.label + "' incorrectly left the record qualified");
        }
    }

    // (16) Parse-level: a serialized line whose sample_value token is "nan", "inf", or overflows
    // double must throw ParseError (fail-closed parsing; JSON has no NaN/Inf literal).
    {
        const std::string base = serialize(make_sample_record());
        const std::string needle = "\"sample_value\":1234.5678901234567";
        auto with_sample_value = [&](const std::string& token) {
            std::string line = base;
            const std::size_t at = line.find(needle);
            check(at != std::string::npos, "sample_value needle not found");
            line.replace(at, needle.size(), "\"sample_value\":" + token);
            return line;
        };
        auto expect_parse_throw = [&](const std::string& token) {
            bool threw = false;
            try {
                (void)parse_line(with_sample_value(token));
            } catch (const ParseError&) { threw = true; }
            check(threw, "sample_value token \"" + token + "\" did not throw ParseError");
        };
        expect_parse_throw("nan");
        expect_parse_throw("inf");
        expect_parse_throw("1e999");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : failures;
}

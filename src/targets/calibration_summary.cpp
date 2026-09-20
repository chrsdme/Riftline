#include "targets/calibration_summary.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <sstream>

namespace ninfer::targets::calibration_summary {
namespace {

using ninfer::targets::calibration::CalibrationIdentity;
using ninfer::targets::calibration::CalibrationRecord;
using ninfer::targets::calibration::DeviceIdentity;

// Builds the canonical logical_slot -> uuid mapping for `devices`. Returns std::nullopt (fail
// closed) if any two devices share the same logical_slot -- a duplicate slot is not a well-formed
// mapping and must never be compared as if it were (sec 6: "fail closed on duplicate logical
// slots"), which classify_compatibility below turns into UnknownUnqualified via the existing
// identity-completeness gate (a slot collision means this identity cannot even be considered
// complete enough to compare).
[[nodiscard]] std::optional<std::map<int, std::string>> canonical_device_mapping(
    const std::vector<DeviceIdentity>& devices) {
    std::map<int, std::string> mapping;
    for (const DeviceIdentity& device : devices) {
        const auto [it, inserted] = mapping.emplace(device.logical_slot, device.uuid);
        if (!inserted) { return std::nullopt; }
    }
    return mapping;
}

// Device key per sec 6/14: the (logical_slot -> uuid) MAPPING, compared as a canonical
// slot-keyed map so two vectors describing the identical mapping in different insertion order
// compare equal, while a slot remap (same UUID set, different slot ownership) or a changed UUID
// for a given slot correctly compares unequal. Duplicate logical slots in either side fail closed
// (never silently compatible, never silently incompatible -- see canonical_device_mapping).
[[nodiscard]] bool devices_match_semantic(const std::vector<DeviceIdentity>& a,
                                           const std::vector<DeviceIdentity>& b) {
    const std::optional<std::map<int, std::string>> mapping_a = canonical_device_mapping(a);
    const std::optional<std::map<int, std::string>> mapping_b = canonical_device_mapping(b);
    if (!mapping_a.has_value() || !mapping_b.has_value()) { return false; }
    return *mapping_a == *mapping_b;
}

// sec 5: full identity-level qualification completeness, applied to BOTH current and stored
// identity before a positive comparison. `compute_capability_minor == 0` remains valid (many GPUs
// legitimately report a 0 minor version). Deliberately does NOT check branch_name/hostname/
// os_description/command_line -- those are audit-only and must never become compatibility keys.
[[nodiscard]] bool identity_complete_enough_to_compare(const CalibrationIdentity& identity) {
    if (identity.source.commit.empty()) { return false; }
    if (identity.source.state != ninfer::targets::calibration::SourceState::Clean) { return false; }
    if (identity.build.build_id.empty()) { return false; }
    if (identity.build.cuda_arch_list.empty()) { return false; }
    if (identity.build.build_type.empty()) { return false; }
    if (identity.artifact.sha256.empty()) { return false; }
    if (identity.artifact.target_id.empty()) { return false; }
    if (identity.artifact.weights_id.empty()) { return false; }
    if (identity.devices.empty()) { return false; }
    for (const DeviceIdentity& d : identity.devices) {
        if (d.logical_slot < 0) { return false; }
        if (d.physical_index < 0) { return false; }
        if (d.uuid.empty()) { return false; }
        if (d.model_name.empty()) { return false; }
        if (d.compute_capability_major <= 0) { return false; }
        if (d.vram_bytes == 0) { return false; }
    }
    // Duplicate logical slots are not a well-formed mapping at all (sec 6): fail closed here so
    // the result is UnknownUnqualified, not a StaleIncompatible fallthrough from a failed
    // semantic-mapping comparison.
    if (!canonical_device_mapping(identity.devices).has_value()) { return false; }
    if (identity.software.driver_version.empty()) { return false; }
    if (identity.software.cuda_runtime_version.empty()) { return false; }
    if (identity.software.cuda_toolkit_version.empty()) { return false; }
    return true;
}

} // namespace

CompatibilityResult classify_compatibility(const CalibrationIdentity& stored,
                                            const CalibrationIdentity& current) {
    CompatibilityResult result;

    // Current-identity completeness is checked FIRST (see header rationale): an incomplete
    // current identity fails closed to UnknownUnqualified regardless of what stored looks like,
    // never StaleIncompatible and never Compatible.
    if (!identity_complete_enough_to_compare(current)) {
        result.status = Compatibility::UnknownUnqualified;
        result.reasons.push_back("current live identity is incomplete");
        return result;
    }
    // A stored identity that was never itself complete cannot be positively compared either.
    if (!identity_complete_enough_to_compare(stored)) {
        result.status = Compatibility::UnknownUnqualified;
        result.reasons.push_back("stored identity is incomplete");
        return result;
    }

    std::vector<std::string> mismatches;
    if (stored.source.commit != current.source.commit) { mismatches.push_back("source_commit"); }
    if (stored.build.build_id != current.build.build_id) { mismatches.push_back("build_id"); }
    if (stored.artifact.sha256 != current.artifact.sha256) { mismatches.push_back("artifact_sha256"); }
    if (stored.artifact.target_id != current.artifact.target_id) {
        mismatches.push_back("artifact_target_id");
    }
    if (stored.artifact.weights_id != current.artifact.weights_id) {
        mismatches.push_back("artifact_weights_id");
    }
    if (!devices_match_semantic(stored.devices, current.devices)) {
        mismatches.push_back("device_uuid_mapping");
    }
    if (stored.software.driver_version != current.software.driver_version) {
        mismatches.push_back("driver_version");
    }
    if (stored.software.cuda_runtime_version != current.software.cuda_runtime_version) {
        mismatches.push_back("cuda_runtime_version");
    }
    if (stored.software.cuda_toolkit_version != current.software.cuda_toolkit_version) {
        mismatches.push_back("cuda_toolkit_version");
    }
    // Audit-only fields (branch_name/hostname/os_description/command_line) are deliberately never
    // compared here (sec 14: they MUST NOT stale).

    if (mismatches.empty()) {
        result.status = Compatibility::Compatible;
        return result;
    }
    result.status = Compatibility::StaleIncompatible;
    result.reasons = std::move(mismatches);
    return result;
}

double percentile(std::vector<double> samples, double fraction) {
    if (samples.empty()) { throw std::invalid_argument("cannot take a percentile of an empty sample set"); }
    std::sort(samples.begin(), samples.end());
    const std::size_t index = std::min(
        samples.size() - 1,
        static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1)));
    return samples[index];
}

CalibrationRecord group_key_record(const CalibrationRecord& record) {
    CalibrationRecord key = record;
    // Per-sample fields: never part of measurement semantics.
    key.sample.run_id.clear();
    key.sample.observation_id.clear();
    key.sample.sample_index = 0;
    key.sample.value = 0.0;
    key.sample.timestamp_unix_ms.reset();
    key.sample.qualification = ninfer::targets::calibration::CorrectnessStatus::NotRun;
    key.qualification = {};
    // Audit-only identity metadata: sec 14 says these must not stale, and symmetrically must not
    // split a summary group either.
    key.identity.branch_name.clear();
    key.identity.hostname.clear();
    key.identity.os_description.clear();
    key.identity.command_line.clear();
    // Artifact path/name/size_bytes are metadata, not identity (header: "never a substitute for
    // sha256"); sha256/target_id/weights_id/container_version stay in the key.
    key.identity.artifact.path.clear();
    key.identity.artifact.name.clear();
    key.identity.artifact.size_bytes = 0;
    // sec 6: normalize the device mapping's VECTOR ORDER in this key copy only (never in the raw
    // record, whose serialization order must not change) so two records with the identical
    // logical_slot->uuid mapping supplied in different insertion order hash to the same group_key
    // instead of splitting into two groups.
    std::sort(key.identity.devices.begin(), key.identity.devices.end(),
              [](const auto& a, const auto& b) { return a.logical_slot < b.logical_slot; });
    return key;
}

std::vector<CalibrationSummary> derive_summaries(const std::vector<CalibrationRecord>& records,
                                                  const CalibrationIdentity& current_identity,
                                                  std::uint64_t min_sample_count) {
    struct Group {
        CalibrationRecord key_record;
        std::vector<double> values;
        std::vector<std::uint64_t> timestamps;
    };
    std::map<std::string, Group> groups; // std::map: deterministic iteration order by key string.

    for (const CalibrationRecord& record : records) {
        if (!ninfer::targets::calibration::is_qualified_calibration_evidence(record)) { continue; }
        if (record.qualification.evidence != ninfer::targets::calibration::EvidenceClass::Measured &&
            record.qualification.evidence !=
                ninfer::targets::calibration::EvidenceClass::MeasuredAnchor) {
            continue;
        }
        if (classify_compatibility(record.identity, current_identity).status !=
            Compatibility::Compatible) {
            continue;
        }

        const CalibrationRecord key_record = group_key_record(record);
        const std::string serialized_key = ninfer::targets::calibration::serialize(key_record);
        const std::string key = ninfer::targets::qwen3_6::frontend_internal::sha256_hex(
            ninfer::targets::qwen3_6::frontend_internal::sha256(std::string_view(serialized_key)));
        Group& group = groups[key];
        group.key_record = key_record;
        group.values.push_back(record.sample.value);
        if (record.sample.timestamp_unix_ms.has_value()) {
            group.timestamps.push_back(*record.sample.timestamp_unix_ms);
        }
    }

    std::vector<CalibrationSummary> summaries;
    summaries.reserve(groups.size());
    for (auto& [key, group] : groups) {
        CalibrationSummary summary;
        summary.group_key = key;
        summary.kind = group.key_record.kind;
        summary.unit = group.key_record.sample.unit;
        summary.stats.count = group.values.size();
        summary.stats.min = *std::min_element(group.values.begin(), group.values.end());
        summary.stats.max = *std::max_element(group.values.begin(), group.values.end());
        summary.stats.p50 = percentile(group.values, 0.50);
        summary.stats.p95 = percentile(group.values, 0.95);
        summary.stats.p99 = percentile(group.values, 0.99);
        if (!group.timestamps.empty()) {
            summary.stats.first_timestamp_unix_ms =
                *std::min_element(group.timestamps.begin(), group.timestamps.end());
            summary.stats.last_timestamp_unix_ms =
                *std::max_element(group.timestamps.begin(), group.timestamps.end());
        }
        summary.qualification = summary.stats.count >= min_sample_count
                                     ? SummaryQualification::Qualified
                                     : SummaryQualification::PartialInsufficientSamples;
        summaries.push_back(std::move(summary));
    }
    return summaries;
}

namespace {

// Minimal hand-rolled flat-JSON writer/reader for the summary format only (NOT a copy of
// calibration.cpp's parser -- deliberately independent per decision: "write your own copy in the
// summary module; the schema boundary says don't touch calibration.cpp"). Only the small field set
// a CalibrationSummary needs.

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

std::string number(double value) {
    std::ostringstream out;
    out.precision(17);
    out << value;
    return out.str();
}

std::string_view record_kind_token(ninfer::targets::calibration::RecordKind kind) {
    return ninfer::targets::calibration::record_kind_name(kind);
}

// One raw token as it appeared in the flat object: `raw` is the unescaped-if-quoted or verbatim-
// if-bare text between the colon and the terminator; `was_quoted` distinguishes a string value
// (needs unescaping, must be treated as a string) from a bare numeric/literal token.
struct RawField {
    std::string raw;
    bool was_quoted = false;
};

// Unescapes the small fixed set serialize_summary()/json_escape() produce: \" \\ \n \r \t.
// Any other backslash escape (including a bare trailing backslash) is rejected outright -- never
// silently pass through raw backslash-escaped bytes as the logical string (sec 10).
std::string unescape_summary_string(std::string_view escaped) {
    std::string out;
    out.reserve(escaped.size());
    for (std::size_t i = 0; i < escaped.size(); ++i) {
        if (escaped[i] != '\\') {
            out += escaped[i];
            continue;
        }
        if (i + 1 >= escaped.size()) {
            throw SummaryParseError("unterminated escape sequence in summary string value");
        }
        switch (escaped[i + 1]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        default:
            throw SummaryParseError("unsupported escape sequence in summary string value: \\" +
                                     std::string(1, escaped[i + 1]));
        }
        ++i;
    }
    return out;
}

// Strict single left-to-right pass over a flat `{"k":v,"k":v,...}` object: collects key -> raw
// value into `fields`, erroring on any structural defect (sec 10 fail-closed list) rather than
// merely finding SOME occurrence of a needle anywhere in the line, which cannot detect duplicate
// keys or trailing garbage.
std::map<std::string, RawField> parse_flat_object_strict(std::string_view line) {
    std::size_t pos = 0;
    const auto skip_ws = [&]() {
        while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) { ++pos; }
    };
    skip_ws();
    if (pos >= line.size() || line[pos] != '{') {
        throw SummaryParseError("summary line does not start with '{'");
    }
    ++pos;

    std::map<std::string, RawField> fields;
    bool first_field = true;
    for (;;) {
        skip_ws();
        if (pos < line.size() && line[pos] == '}' && first_field && fields.empty()) {
            throw SummaryParseError("summary object must not be empty");
        }
        if (pos >= line.size() || line[pos] != '"') {
            throw SummaryParseError("expected a quoted key in summary object");
        }
        ++pos;
        const std::size_t key_start = pos;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\') { ++pos; } // skip escaped char, including an escaped quote.
            ++pos;
        }
        if (pos >= line.size()) { throw SummaryParseError("unterminated key in summary object"); }
        const std::string key = unescape_summary_string(line.substr(key_start, pos - key_start));
        ++pos; // closing quote.

        skip_ws();
        if (pos >= line.size() || line[pos] != ':') {
            throw SummaryParseError("expected ':' after key \"" + key + "\"");
        }
        ++pos;
        skip_ws();

        RawField field;
        if (pos < line.size() && line[pos] == '"') {
            ++pos;
            const std::size_t value_start = pos;
            while (pos < line.size() && line[pos] != '"') {
                if (line[pos] == '\\') { ++pos; }
                ++pos;
            }
            if (pos >= line.size()) {
                throw SummaryParseError("unterminated string value for key \"" + key + "\"");
            }
            field.raw = unescape_summary_string(line.substr(value_start, pos - value_start));
            field.was_quoted = true;
            ++pos; // closing quote.
        } else {
            const std::size_t value_start = pos;
            while (pos < line.size() && line[pos] != ',' && line[pos] != '}' && line[pos] != ' ' &&
                   line[pos] != '\t') {
                ++pos;
            }
            if (pos == value_start) {
                throw SummaryParseError("missing value for key \"" + key + "\"");
            }
            field.raw = std::string(line.substr(value_start, pos - value_start));
            field.was_quoted = false;
        }

        if (!fields.emplace(key, std::move(field)).second) {
            throw SummaryParseError("duplicate key \"" + key + "\" in summary object");
        }
        first_field = false;

        skip_ws();
        if (pos >= line.size()) { throw SummaryParseError("unterminated summary object"); }
        if (line[pos] == ',') {
            ++pos;
            continue;
        }
        if (line[pos] == '}') {
            ++pos;
            break;
        }
        throw SummaryParseError("expected ',' or '}' in summary object");
    }

    skip_ws();
    if (pos != line.size()) {
        throw SummaryParseError("trailing non-whitespace after summary object");
    }
    return fields;
}

const RawField& require_field(const std::map<std::string, RawField>& fields, std::string_view key) {
    const auto it = fields.find(std::string(key));
    if (it == fields.end()) {
        throw SummaryParseError("missing required key \"" + std::string(key) + "\"");
    }
    return it->second;
}

// Strict complete-token signed-integer parse: std::from_chars must consume the ENTIRE token, so
// "1junk" (which std::stoi would silently accept as 1) is rejected.
long long parse_strict_integer(const std::string& token, std::string_view field_name) {
    if (token.empty()) {
        throw SummaryParseError("empty integer value for \"" + std::string(field_name) + "\"");
    }
    long long value = 0;
    const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size()) {
        throw SummaryParseError("malformed integer value for \"" + std::string(field_name) +
                                 "\": " + token);
    }
    return value;
}

// Strict complete-token double parse via strtod + endptr-consumed-all + isfinite: strtod alone
// accepts "nan"/"inf"/"infinity" tokens, which isfinite() then rejects (sec 10: NaN/Inf statistics
// must never pass).
double parse_strict_finite_double(const std::string& token, std::string_view field_name) {
    if (token.empty()) {
        throw SummaryParseError("empty numeric value for \"" + std::string(field_name) + "\"");
    }
    errno = 0;
    char* endptr = nullptr;
    const double value = std::strtod(token.c_str(), &endptr);
    if (endptr != token.c_str() + token.size() || endptr == token.c_str()) {
        throw SummaryParseError("malformed numeric value for \"" + std::string(field_name) +
                                 "\": " + token);
    }
    if (!std::isfinite(value)) {
        throw SummaryParseError("non-finite numeric value for \"" + std::string(field_name) +
                                 "\": " + token);
    }
    return value;
}

bool is_lowercase_hex_group_key(std::string_view value) {
    if (value.size() != 64) { return false; }
    for (char c : value) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!ok) { return false; }
    }
    return true;
}

} // namespace

std::string serialize_summary(const CalibrationSummary& summary) {
    std::ostringstream out;
    out << '{';
    out << "\"artifact_type\":\"" << kSummaryArtifactType << "\",";
    out << "\"summary_schema_version\":" << kSummarySchemaVersion << ',';
    out << "\"group_key\":\"" << json_escape(summary.group_key) << "\",";
    out << "\"record_kind\":\"" << record_kind_token(summary.kind) << "\",";
    out << "\"unit\":\"" << json_escape(summary.unit) << "\",";
    out << "\"count\":" << summary.stats.count << ',';
    out << "\"min\":" << number(summary.stats.min) << ',';
    out << "\"max\":" << number(summary.stats.max) << ',';
    out << "\"p50\":" << number(summary.stats.p50) << ',';
    out << "\"p95\":" << number(summary.stats.p95) << ',';
    out << "\"p99\":" << number(summary.stats.p99) << ',';
    out << "\"first_timestamp_unix_ms\":"
        << (summary.stats.first_timestamp_unix_ms.has_value()
                ? std::to_string(*summary.stats.first_timestamp_unix_ms)
                : std::string("-1"))
        << ',';
    out << "\"last_timestamp_unix_ms\":"
        << (summary.stats.last_timestamp_unix_ms.has_value()
                ? std::to_string(*summary.stats.last_timestamp_unix_ms)
                : std::string("-1"))
        << ',';
    out << "\"qualification\":\""
        << (summary.qualification == SummaryQualification::Qualified ? "qualified" : "partial")
        << "\"";
    out << '}';
    return out.str();
}

CalibrationSummary parse_summary_line(std::string_view line) {
    const std::map<std::string, RawField> fields = parse_flat_object_strict(line);

    const RawField& artifact_type_field = require_field(fields, "artifact_type");
    if (!artifact_type_field.was_quoted ||
        artifact_type_field.raw != std::string(kSummaryArtifactType)) {
        throw SummaryParseError("wrong/missing artifact_type for a calibration summary line");
    }

    const RawField& schema_field = require_field(fields, "summary_schema_version");
    if (schema_field.was_quoted) {
        throw SummaryParseError("summary_schema_version must be a bare numeric token");
    }
    const long long schema_value =
        parse_strict_integer(schema_field.raw, "summary_schema_version");
    if (schema_value != kSummarySchemaVersion) {
        throw SummaryParseError("unsupported summary_schema_version: " + schema_field.raw);
    }

    const RawField& group_key_field = require_field(fields, "group_key");
    if (!group_key_field.was_quoted || !is_lowercase_hex_group_key(group_key_field.raw)) {
        throw SummaryParseError("group_key must be exactly 64 lowercase hex characters");
    }

    const RawField& record_kind_field = require_field(fields, "record_kind");
    if (!record_kind_field.was_quoted) {
        throw SummaryParseError("record_kind must be a quoted string");
    }
    const std::optional<ninfer::targets::calibration::RecordKind> record_kind =
        record_kind_field.raw == "operation"
            ? std::optional(ninfer::targets::calibration::RecordKind::Operation)
        : record_kind_field.raw == "transfer"
            ? std::optional(ninfer::targets::calibration::RecordKind::Transfer)
        : record_kind_field.raw == "gdn" ? std::optional(ninfer::targets::calibration::RecordKind::Gdn)
                                          : std::nullopt;
    if (!record_kind.has_value()) {
        throw SummaryParseError("unknown record_kind: " + record_kind_field.raw);
    }

    const RawField& unit_field = require_field(fields, "unit");
    if (!unit_field.was_quoted || unit_field.raw.empty()) {
        throw SummaryParseError("unit must be a non-empty quoted string");
    }

    const RawField& count_field = require_field(fields, "count");
    if (count_field.was_quoted) { throw SummaryParseError("count must be a bare numeric token"); }
    const long long count_value = parse_strict_integer(count_field.raw, "count");
    if (count_value <= 0) {
        throw SummaryParseError("count must be > 0, got: " + count_field.raw);
    }

    const auto require_finite = [&](std::string_view key) {
        const RawField& field = require_field(fields, key);
        if (field.was_quoted) {
            throw SummaryParseError(std::string(key) + " must be a bare numeric token");
        }
        return parse_strict_finite_double(field.raw, key);
    };
    const double min_value = require_finite("min");
    const double max_value = require_finite("max");
    const double p50_value = require_finite("p50");
    const double p95_value = require_finite("p95");
    const double p99_value = require_finite("p99");
    if (!(min_value <= p50_value && p50_value <= p95_value && p95_value <= p99_value &&
          p99_value <= max_value)) {
        throw SummaryParseError("inconsistent statistic ordering: require min<=p50<=p95<=p99<=max");
    }

    // -1 is the ONLY valid "absent" sentinel serialize_summary() ever writes (see
    // std::optional<std::uint64_t>::has_value() ? to_string(*v) : "-1" above); any other negative
    // value is a malformed timestamp and must be rejected, not silently treated as absent.
    const RawField& first_ts_field = require_field(fields, "first_timestamp_unix_ms");
    if (first_ts_field.was_quoted) {
        throw SummaryParseError("first_timestamp_unix_ms must be a bare numeric token");
    }
    const long long first_ts = parse_strict_integer(first_ts_field.raw, "first_timestamp_unix_ms");
    if (first_ts < -1) {
        throw SummaryParseError("malformed first_timestamp_unix_ms: " + first_ts_field.raw);
    }

    const RawField& last_ts_field = require_field(fields, "last_timestamp_unix_ms");
    if (last_ts_field.was_quoted) {
        throw SummaryParseError("last_timestamp_unix_ms must be a bare numeric token");
    }
    const long long last_ts = parse_strict_integer(last_ts_field.raw, "last_timestamp_unix_ms");
    if (last_ts < -1) {
        throw SummaryParseError("malformed last_timestamp_unix_ms: " + last_ts_field.raw);
    }
    if (first_ts >= 0 && last_ts >= 0 && first_ts > last_ts) {
        throw SummaryParseError("first_timestamp_unix_ms must not exceed last_timestamp_unix_ms");
    }

    const RawField& qualification_field = require_field(fields, "qualification");
    if (!qualification_field.was_quoted) {
        throw SummaryParseError("qualification must be a quoted string");
    }
    SummaryQualification qualification;
    if (qualification_field.raw == "qualified") {
        qualification = SummaryQualification::Qualified;
    } else if (qualification_field.raw == "partial") {
        qualification = SummaryQualification::PartialInsufficientSamples;
    } else {
        throw SummaryParseError("unknown qualification token: " + qualification_field.raw);
    }

    CalibrationSummary summary;
    summary.group_key = group_key_field.raw;
    summary.kind = *record_kind;
    summary.unit = unit_field.raw;
    summary.stats.count = static_cast<std::uint64_t>(count_value);
    summary.stats.min = min_value;
    summary.stats.max = max_value;
    summary.stats.p50 = p50_value;
    summary.stats.p95 = p95_value;
    summary.stats.p99 = p99_value;
    if (first_ts >= 0) { summary.stats.first_timestamp_unix_ms = static_cast<std::uint64_t>(first_ts); }
    if (last_ts >= 0) { summary.stats.last_timestamp_unix_ms = static_cast<std::uint64_t>(last_ts); }
    summary.qualification = qualification;
    return summary;
}

} // namespace ninfer::targets::calibration_summary

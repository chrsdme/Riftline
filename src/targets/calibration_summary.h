#pragma once

// M5.4b sec 14-16: staleness/compatibility classifier, deterministic summary derivation, and
// summary storage. Host-only, CUDA-free. Percentile convention documented at percentile() below.

#include "targets/calibration.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::calibration_summary {

// ---------------------------------------------------------------------------------------------
// sec 14: staleness / compatibility classifier
// ---------------------------------------------------------------------------------------------

enum class Compatibility : std::uint8_t { Compatible = 0, StaleIncompatible, UnknownUnqualified };

struct CompatibilityResult {
    Compatibility status = Compatibility::UnknownUnqualified;
    std::vector<std::string> reasons; // empty iff status == Compatible.
};

// Fail-closed comparison of `stored` (an identity captured with a raw observation) against
// `current` (freshly collected live identity). Does not mutate either argument -- this is a pure
// derived classification, never a destructive rewrite of a stored record (sec 14: "do not mutate
// historical raw observations destructively").
//
// Order of checks matters for the "current identity incomplete" test case: current-identity
// completeness is checked FIRST. An incomplete current identity yields UnknownUnqualified
// immediately, before any field-by-field comparison runs -- otherwise an incomplete-and-mismatched
// current identity would misreport as StaleIncompatible instead of UnknownUnqualified.
//
// Exact-match compatibility keys (sec 14): source commit, build_id, artifact sha256, target_id,
// weights_id, participating device UUID mapping (compared as a canonical logical_slot -> uuid MAP,
// order-independent -- so a slot remap that changes which slot owns which UUID is still a
// mismatch even when the UUID set is unchanged, while the same mapping supplied in a different
// vector order is NOT a mismatch; duplicate logical slots fail closed via the identity-
// completeness gate below, sec 6), driver version, CUDA runtime version, CUDA toolkit version.
//
// Audit-only fields that MUST NOT affect the result: branch_name, hostname, os_description,
// command_line.
[[nodiscard]] CompatibilityResult classify_compatibility(
    const ninfer::targets::calibration::CalibrationIdentity& stored,
    const ninfer::targets::calibration::CalibrationIdentity& current);

// ---------------------------------------------------------------------------------------------
// sec 15: deterministic summary derivation
// ---------------------------------------------------------------------------------------------

// Nearest-rank percentile with no interpolation, adopted from bench/ops/ninfer_bench_common.h's
// `summarize_timings` (its FIRST convention, ~line 129-136: `samples[min(n-1, floor(f*(n-1)))]` on
// a copy sorted ascending) rather than its second convention (~line 311-315:
// `sorted[min(n-1, floor(q*n))]`) -- ninfer_bench_common.h is a bench-target header this host-only
// module must not include (decision 4), so the formula is reproduced here, not the dependency.
// p99 extends the same formula (not present in the original helper). `samples` need not be
// pre-sorted; this function sorts a local copy.
[[nodiscard]] double percentile(std::vector<double> samples, double fraction);

struct SummaryStats {
    std::uint64_t count = 0;
    double min = 0.0;
    double max = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    std::optional<std::uint64_t> first_timestamp_unix_ms;
    std::optional<std::uint64_t> last_timestamp_unix_ms;
};

enum class SummaryQualification : std::uint8_t { Qualified = 0, PartialInsufficientSamples };

struct CalibrationSummary {
    // Canonical grouping key: sha256_hex(serialize(key_record)), where key_record is a copy of one
    // member record with every per-sample field cleared (run_id/observation_id/sample_index/
    // value/timestamp/sample-and-record qualification state) and every audit-only field cleared
    // (branch_name/hostname/os_description/command_line, and artifact path/name/size_bytes --
    // header: metadata, not identity). Unit is DELIBERATELY KEPT in the key: it is what makes
    // mixed units land in different groups instead of one combined group (sec 15/24: never combine
    // mixed units). Reusing serialize() as the input means the key can never drift out of sync
    // with what "everything that changes measurement semantics" actually contains -- see
    // group_key_record() in calibration_summary.cpp for the exact clearing rule. Hashed (rather
    // than stored as the raw serialized string) so the summary line never embeds
    // record-shaped/record-typed content (sec 16: must not masquerade as a raw record, and must
    // not duplicate the raw corpus inside the summary).
    std::string group_key;
    ninfer::targets::calibration::RecordKind kind = ninfer::targets::calibration::RecordKind::Operation;
    std::string unit;
    SummaryStats stats;
    SummaryQualification qualification = SummaryQualification::PartialInsufficientSamples;
};

// Builds the canonical grouping-key record for `record` (public so tests can assert grouping-key
// separation directly without needing hundreds of near-duplicate fixture records; see decision
// note above for exactly which fields get cleared).
[[nodiscard]] ninfer::targets::calibration::CalibrationRecord group_key_record(
    const ninfer::targets::calibration::CalibrationRecord& record);

// Default input filter (sec 15): is_qualified_calibration_evidence(record) == true AND
// EvidenceClass in {Measured, MeasuredAnchor} AND classify_compatibility(record.identity,
// current_identity).status == Compatible. Records failing the filter are silently excluded from
// every group (they remain retrievable from the raw corpus untouched -- this function never
// mutates its input). Groups with mixed units cannot occur here because unit is part of the key;
// a caller who groups upstream in some other, non-serialize-based way and colides two different
// units into one bucket is a caller bug this module does not need to guard against internally.
//
// A group below `min_sample_count` is marked PartialInsufficientSamples (stats are still computed
// and returned -- sec 23: raw records stay retained, and the summary is not discarded, only
// flagged). `min_sample_count` has no built-in default here (sec 23: do not invent a statistically
// authoritative global minimum) -- callers must supply one explicitly.
[[nodiscard]] std::vector<CalibrationSummary> derive_summaries(
    const std::vector<ninfer::targets::calibration::CalibrationRecord>& records,
    const ninfer::targets::calibration::CalibrationIdentity& current_identity,
    std::uint64_t min_sample_count);

// ---------------------------------------------------------------------------------------------
// sec 16: summary storage format
// ---------------------------------------------------------------------------------------------

inline constexpr std::string_view kSummaryArtifactType = "ninfer_calibration_summary";
inline constexpr int kSummarySchemaVersion = 1;

// Deterministic single-line flat-JSON serialization, explicitly NOT artifact_type=
// ninfer_calibration_record (sec 16: must not masquerade as a raw M5.4a record). Does not
// duplicate the raw corpus -- only group_key/kind/unit/stats/qualification are stored.
[[nodiscard]] std::string serialize_summary(const CalibrationSummary& summary);

struct SummaryParseError : std::runtime_error {
    explicit SummaryParseError(const std::string& message) : std::runtime_error(message) {}
};

[[nodiscard]] CalibrationSummary parse_summary_line(std::string_view line);

} // namespace ninfer::targets::calibration_summary

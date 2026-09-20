#pragma once

// M5.4b sec 12-13: raw JSONL writer + corpus reader. Host-only, CUDA-free. Wraps the accepted
// M5.4a serialize()/parse_line() -- this module owns file I/O and naming only, never the record
// schema.

#include "targets/calibration.h"
#include "targets/calibration_summary.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::calibration_io {

// Appends `line` to `fd` (an O_APPEND-opened, single-owner file) as one atomic-from-the-caller's-
// -view row: retries on EINTR, loops on short writes, and on a hard failure after a partial row
// truncates the file back to its pre-row size so no partial row is ever left behind. `fd` must be
// O_APPEND (so a plain write() always appends regardless of any concurrent lseek from elsewhere in
// this process) -- because it is O_APPEND, `lseek(fd, 0, SEEK_CUR)` does NOT reflect the row's
// start offset, so the pre-row offset is captured via fstat().st_size instead. Throws
// std::runtime_error on any I/O failure (including the rollback ftruncate itself failing, which is
// reported but does not mask the original write error). Single-owner-file safe only: no advisory
// locking, no protection against another process/fd touching the same file concurrently.
void write_full_line(int fd, std::string_view line);

// ---------------------------------------------------------------------------------------------
// sec 12: raw JSONL writer
// ---------------------------------------------------------------------------------------------

// One open run file under `root_dir` (default ninfer::targets::calibration::kDefaultCalibrationRawDir,
// caller-overridable for tests -- tests MUST always pass a temp dir). File name is
// `{unix_ms_timestamp}_{sanitized_run_id}.jsonl`. Construction creates `root_dir` (recursively) if
// missing and opens the file with O_CREAT|O_EXCL semantics: if a file with that exact name already
// exists (same millisecond + same run_id -- vanishingly unlikely but not impossible, and exactly
// what "repeated use must not silently overwrite" requires covering), construction throws rather
// than truncating/appending to a prior run's data.
class RawWriter {
public:
    // Throws std::runtime_error if: run_id is empty/invalid (must be non-empty and contain only
    // [A-Za-z0-9_.-], rejecting '/' and '..' outright to block path traversal into root_dir),
    // root_dir cannot be created, or the run file already exists.
    RawWriter(std::string root_dir, std::string run_id);

    // Appends one M5.4a serialize()'d line for `record` (whatever its ValidityState/EvidenceClass
    // -- invalid/unqualified/unreliable records are preserved for audit, never filtered here), via
    // write_full_line() (EINTR-retrying, short-write-looping, rolls back to the pre-row offset on
    // hard failure). Throws std::runtime_error on serialization failure or write failure (fail
    // loudly, no partial/fabricated line left behind). `records_written()` is only incremented
    // after a complete row has been written.
    void append(const ninfer::targets::calibration::CalibrationRecord& record);

    // Flushes and closes the file explicitly. Throws std::runtime_error if flush/close reports an
    // error. Safe to call more than once (subsequent calls are a no-op). The destructor also
    // flushes/closes but swallows errors (destructors must not throw) -- callers who need to
    // observe flush/close failures MUST call close() explicitly rather than relying on the
    // destructor.
    void close();

    ~RawWriter();

    RawWriter(const RawWriter&) = delete;
    RawWriter& operator=(const RawWriter&) = delete;
    RawWriter(RawWriter&&) = delete;
    RawWriter& operator=(RawWriter&&) = delete;

    [[nodiscard]] const std::string& file_path() const noexcept { return file_path_; }
    [[nodiscard]] std::uint64_t records_written() const noexcept { return records_written_; }

private:
    std::string file_path_;
    int fd_ = -1;
    std::uint64_t records_written_ = 0;
    bool closed_ = false;
};

// Validates a run_id per the same rule RawWriter enforces (non-empty, [A-Za-z0-9_.-] only, no
// path separators). Exposed so callers/tests can pre-check without constructing a writer.
[[nodiscard]] bool is_valid_run_id(const std::string& run_id) noexcept;

// ---------------------------------------------------------------------------------------------
// sec 16: summary persistence
// ---------------------------------------------------------------------------------------------

// One open summary file under `root_dir` (default
// ninfer::targets::calibration::kDefaultCalibrationSummariesDir, caller-overridable -- tests MUST
// always pass a temp dir). File name is `{unix_ms_timestamp}_{sanitized_run_id}.summary.jsonl`.
// Construction creates `root_dir` (recursively) if missing and opens the file with O_CREAT|O_EXCL
// semantics: no silent overwrite of a prior run's summaries. One serialize_summary() line per
// summary; rows are appended via the same write_full_line() helper RawWriter uses (EINTR-retrying,
// short-write-safe, rolls back a partial row).
class SummaryWriter {
public:
    // Throws std::runtime_error if: run_id is empty/invalid, root_dir cannot be created, or the
    // summary file already exists.
    SummaryWriter(std::string root_dir, std::string run_id);

    // Appends one serialize_summary()'d line. Throws std::runtime_error on serialization or write
    // failure. `summaries_written()` is only incremented after a complete row is written.
    void append(const ninfer::targets::calibration_summary::CalibrationSummary& summary);

    // Flushes and closes the file explicitly. Throws std::runtime_error if flush/close reports an
    // error. Safe to call more than once. The destructor also flushes/closes but swallows errors.
    void close();

    ~SummaryWriter();

    SummaryWriter(const SummaryWriter&) = delete;
    SummaryWriter& operator=(const SummaryWriter&) = delete;
    SummaryWriter(SummaryWriter&&) = delete;
    SummaryWriter& operator=(SummaryWriter&&) = delete;

    [[nodiscard]] const std::string& file_path() const noexcept { return file_path_; }
    [[nodiscard]] std::uint64_t summaries_written() const noexcept { return summaries_written_; }

private:
    std::string file_path_;
    int fd_ = -1;
    std::uint64_t summaries_written_ = 0;
    bool closed_ = false;
};

// ---------------------------------------------------------------------------------------------
// sec 13: corpus reader / corrupt-line isolation
// ---------------------------------------------------------------------------------------------

// Reads the JSONL file at `path` line by line through the accepted M5.4a parser. One corrupt line
// never discards valid neighbors: `records` holds every successfully parsed line (in file order,
// skipping blanks) and `line_errors` holds (0-based line index, error message) for every line that
// failed to parse. Throws std::runtime_error only if the file itself cannot be opened/read (a
// missing/unreadable corpus file is a harder failure than one corrupt line inside it).
[[nodiscard]] ninfer::targets::calibration::CorpusParseResult read_corpus_file(
    const std::string& path);

} // namespace ninfer::targets::calibration_io

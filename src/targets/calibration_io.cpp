#include "targets/calibration_io.h"

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::targets::calibration_io {
namespace {

namespace fs = std::filesystem;

std::uint64_t now_unix_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

} // namespace

void write_full_line(int fd, std::string_view line) {
    // O_APPEND means lseek(fd, 0, SEEK_CUR) does not reflect where THIS row will land -- capture
    // the pre-row size via fstat() so a hard failure mid-row can roll back to it.
    struct stat pre_stat{};
    if (::fstat(fd, &pre_stat) != 0) {
        throw std::runtime_error("fstat failed before writing row");
    }
    const off_t pre_row_offset = pre_stat.st_size;

    std::size_t written_total = 0;
    while (written_total < line.size()) {
        const ssize_t written =
            ::write(fd, line.data() + written_total, line.size() - written_total);
        if (written < 0) {
            if (errno == EINTR) { continue; }
            // Best-effort rollback; original write error is authoritative regardless of whether
            // the rollback itself succeeds.
            if (::ftruncate(fd, pre_row_offset) != 0) { /* best effort; original error is authoritative */ }
            throw std::runtime_error("write failed for row (rolled back to pre-row offset)");
        }
        if (written == 0) {
            // No forward progress and not EINTR: treat as a hard failure rather than spin.
            if (::ftruncate(fd, pre_row_offset) != 0) { /* best effort; original error is authoritative */ }
            throw std::runtime_error("write made no progress for row (rolled back to pre-row offset)");
        }
        written_total += static_cast<std::size_t>(written);
    }
}

bool is_valid_run_id(const std::string& run_id) noexcept {
    if (run_id.empty()) { return false; }
    for (char c : run_id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '_' || c == '.' || c == '-';
        if (!ok) { return false; }
    }
    // Reject a bare "." or ".." component outright even though the charset above already excludes
    // '/' (so no traversal across directories is possible either way) -- belt and suspenders
    // against a run_id of exactly "." or "..".
    if (run_id == "." || run_id == "..") { return false; }
    return true;
}

RawWriter::RawWriter(std::string root_dir, std::string run_id) {
    if (!is_valid_run_id(run_id)) {
        throw std::runtime_error("invalid run_id (must be non-empty [A-Za-z0-9_.-]): " + run_id);
    }
    std::error_code fs_error;
    fs::create_directories(root_dir, fs_error);
    if (fs_error && !fs::is_directory(root_dir)) {
        throw std::runtime_error("failed to create raw calibration directory: " + root_dir + ": " +
                                  fs_error.message());
    }

    const std::string file_name = std::to_string(now_unix_ms()) + "_" + run_id + ".jsonl";
    fs::path path = fs::path(root_dir) / file_name;
    file_path_ = path.string();

    // O_CREAT|O_EXCL: fails if the file already exists rather than truncating/appending, which is
    // what "repeated use must not silently overwrite a prior run file" requires. Given the file
    // name embeds a millisecond timestamp, a collision only occurs if the same run_id is reused
    // within the same millisecond -- exactly the case this guard must not silently paper over.
    fd_ = ::open(file_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::runtime_error("failed to create raw calibration run file (already exists?): " +
                                  file_path_);
    }
}

void RawWriter::append(const ninfer::targets::calibration::CalibrationRecord& record) {
    if (closed_) { throw std::runtime_error("append() called on a closed RawWriter: " + file_path_); }
    std::string line;
    try {
        line = ninfer::targets::calibration::serialize(record);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to serialize calibration record: " + std::string(e.what()));
    }
    line.push_back('\n');
    try {
        write_full_line(fd_, line);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to write row to raw calibration run file: " + file_path_ +
                                  ": " + e.what());
    }
    ++records_written_;
}

void RawWriter::close() {
    if (closed_) { return; }
    closed_ = true;
    if (fd_ < 0) { return; }
    if (::fsync(fd_) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fsync failed for raw calibration run file: " + file_path_);
    }
    const int rc = ::close(fd_);
    fd_ = -1;
    if (rc != 0) {
        throw std::runtime_error("close failed for raw calibration run file: " + file_path_);
    }
}

RawWriter::~RawWriter() {
    if (!closed_ && fd_ >= 0) {
        ::fsync(fd_);
        ::close(fd_);
    }
}

SummaryWriter::SummaryWriter(std::string root_dir, std::string run_id) {
    if (!is_valid_run_id(run_id)) {
        throw std::runtime_error("invalid run_id (must be non-empty [A-Za-z0-9_.-]): " + run_id);
    }
    std::error_code fs_error;
    fs::create_directories(root_dir, fs_error);
    if (fs_error && !fs::is_directory(root_dir)) {
        throw std::runtime_error("failed to create calibration summaries directory: " + root_dir +
                                  ": " + fs_error.message());
    }

    const std::string file_name = std::to_string(now_unix_ms()) + "_" + run_id + ".summary.jsonl";
    fs::path path = fs::path(root_dir) / file_name;
    file_path_ = path.string();

    // O_CREAT|O_EXCL: no silent overwrite of a prior run's summaries.
    fd_ = ::open(file_path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_APPEND, 0644);
    if (fd_ < 0) {
        throw std::runtime_error("failed to create calibration summary file (already exists?): " +
                                  file_path_);
    }
}

void SummaryWriter::append(const ninfer::targets::calibration_summary::CalibrationSummary& summary) {
    if (closed_) {
        throw std::runtime_error("append() called on a closed SummaryWriter: " + file_path_);
    }
    std::string line;
    try {
        line = ninfer::targets::calibration_summary::serialize_summary(summary);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to serialize calibration summary: " + std::string(e.what()));
    }
    line.push_back('\n');
    try {
        write_full_line(fd_, line);
    } catch (const std::exception& e) {
        throw std::runtime_error("failed to write row to calibration summary file: " + file_path_ +
                                  ": " + e.what());
    }
    ++summaries_written_;
}

void SummaryWriter::close() {
    if (closed_) { return; }
    closed_ = true;
    if (fd_ < 0) { return; }
    if (::fsync(fd_) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fsync failed for calibration summary file: " + file_path_);
    }
    const int rc = ::close(fd_);
    fd_ = -1;
    if (rc != 0) {
        throw std::runtime_error("close failed for calibration summary file: " + file_path_);
    }
}

SummaryWriter::~SummaryWriter() {
    if (!closed_ && fd_ >= 0) {
        ::fsync(fd_);
        ::close(fd_);
    }
}

ninfer::targets::calibration::CorpusParseResult read_corpus_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("failed to open calibration corpus file: " + path); }

    // Genuine line-by-line streaming: O(max_line_size) memory, not O(corpus_size). Mirrors
    // parse_corpus()'s exact semantics: blank test is `find_first_not_of(" \t\r") == npos`;
    // line_index is 0-based and increments for blank lines too (they are skipped for parsing but
    // still consume an index); getline() does NOT strip a trailing '\r' on CRLF input, matching
    // parse_corpus() operating on raw (unstripped) lines.
    ninfer::targets::calibration::CorpusParseResult result;
    std::string line;
    std::size_t line_index = 0;
    while (std::getline(file, line)) {
        const bool blank = line.find_first_not_of(" \t\r") == std::string::npos;
        if (!blank) {
            try {
                result.records.push_back(ninfer::targets::calibration::parse_line(line));
            } catch (const ninfer::targets::calibration::ParseError& error) {
                result.line_errors.emplace_back(line_index, error.what());
            }
        }
        ++line_index;
    }
    if (file.bad()) { throw std::runtime_error("failed to read calibration corpus file: " + path); }
    return result;
}

} // namespace ninfer::targets::calibration_io

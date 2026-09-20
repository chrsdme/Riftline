#include "targets/calibration_collect.h"

#include "targets/calibration_build_id.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace ninfer::targets::calibration_collect {
namespace {

namespace fs = std::filesystem;
using ninfer::targets::calibration::ArtifactIdentity;
using ninfer::targets::calibration::SourceIdentity;
using ninfer::targets::calibration::SourceState;

// Runs `argv` (no shell involved -- fork/exec directly, so there is no shell-metacharacter
// injection surface regardless of what `argv` contains) with a fixed working directory, captures
// stdout, and reports the exit status. `ok` is false on fork/exec/wait failure; a non-zero exit
// from the child is reported via `exit_code` and is NOT itself an `ok=false` (the caller decides
// what a non-zero git exit means).
struct ProcessResult {
    bool ok = false;
    int exit_code = -1;
    std::string stdout_text;
};

ProcessResult run_process(const std::vector<std::string>& argv, const std::string& cwd) {
    ProcessResult result;
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) { return result; }

    const pid_t pid = fork();
    if (pid < 0) {
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        return result;
    }
    if (pid == 0) {
        // Child: redirect stdout to the pipe, discard stderr (git's warnings are not our
        // authority), exec directly -- no shell. Neutralize the developer's global/system git
        // config here, in the actual production child, so identity collection can never be
        // influenced by e.g. `[status] showUntrackedFiles = no` in ~/.gitconfig or /etc/gitconfig
        // (sec 3). fork() copies the parent's env, so mutating it here (after fork, before exec)
        // only affects this child process, never the parent.
        setenv("GIT_CONFIG_GLOBAL", "/dev/null", 1);
        setenv("GIT_CONFIG_SYSTEM", "/dev/null", 1);
        dup2(pipe_fds[1], STDOUT_FILENO);
        close(pipe_fds[0]);
        close(pipe_fds[1]);
        if (std::freopen("/dev/null", "w", stderr) == nullptr) { /* best-effort; child continues */ }
        if (chdir(cwd.c_str()) != 0) { _exit(127); }
        std::vector<char*> c_argv;
        c_argv.reserve(argv.size() + 1);
        for (const std::string& arg : argv) { c_argv.push_back(const_cast<char*>(arg.c_str())); }
        c_argv.push_back(nullptr);
        execvp(c_argv[0], c_argv.data());
        _exit(127); // exec failed
    }

    // Parent.
    close(pipe_fds[1]);
    std::array<char, 4096> buffer{};
    ssize_t n = 0;
    while ((n = read(pipe_fds[0], buffer.data(), buffer.size())) > 0) {
        result.stdout_text.append(buffer.data(), static_cast<std::size_t>(n));
    }
    close(pipe_fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) { return result; }
    result.ok = true;
    result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return result;
}

std::string trim(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) { s.pop_back(); }
    return s;
}

[[nodiscard]] bool looks_like_git_sha(std::string_view s) {
    if (s.size() != 40) { return false; }
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) { return false; }
    }
    return true;
}

} // namespace

SourceIdentity collect_source_identity(const std::string& source_root) {
    SourceIdentity identity; // defaults to Unknown, empty commit.

    std::error_code fs_error;
    const fs::path canonical_root = fs::weakly_canonical(fs::path(source_root), fs_error);
    if (fs_error || !fs::exists(canonical_root)) { return identity; }

    // Guard: source_root's own git toplevel must equal source_root itself (realpath-compared).
    // This is what actually stops an outer/parent repository (e.g. the lab-wide
    // /media/storage/ninfer-lab/.git) from being silently treated as authority for a nested
    // source_root that is not itself a repo root -- `git -C <subdir> rev-parse HEAD` would
    // otherwise happily resolve against the parent repo.
    const ProcessResult toplevel =
        run_process({"git", "rev-parse", "--show-toplevel"}, canonical_root.string());
    if (!toplevel.ok || toplevel.exit_code != 0) { return identity; }
    std::error_code toplevel_fs_error;
    const fs::path reported_toplevel =
        fs::weakly_canonical(fs::path(trim(toplevel.stdout_text)), toplevel_fs_error);
    if (toplevel_fs_error || reported_toplevel != canonical_root) { return identity; }

    const ProcessResult head = run_process({"git", "rev-parse", "HEAD"}, canonical_root.string());
    if (!head.ok || head.exit_code != 0) { return identity; }
    const std::string commit = trim(head.stdout_text);
    if (!looks_like_git_sha(commit)) { return identity; }
    identity.commit = commit;

    // status --porcelain --untracked-files=all: any output (tracked or untracked) means Dirty.
    // Fail-closed: an untracked file is exactly the kind of provenance gap qualified evidence must
    // reject, so it counts toward Dirty rather than being ignored. `--untracked-files=all` makes
    // the invocation itself immune to a hostile `[status] showUntrackedFiles=no`/`normal` config
    // (on top of the GIT_CONFIG_GLOBAL/GIT_CONFIG_SYSTEM neutralization in run_process()'s child):
    // an explicit CLI flag always overrides that config setting.
    const ProcessResult status = run_process(
        {"git", "status", "--porcelain", "--untracked-files=all"}, canonical_root.string());
    if (!status.ok || status.exit_code != 0) {
        // Commit resolved but state could not be determined: state stays Unknown (never assume
        // Clean), commit is still recorded (it IS known).
        identity.state = SourceState::Unknown;
        return identity;
    }
    identity.state = trim(status.stdout_text).empty() ? SourceState::Clean : SourceState::Dirty;
    return identity;
}

std::string build_fingerprint_input() { return kNinferBuildFingerprintInput; }

std::string build_id_from_input(const std::string& fingerprint_input) {
    using ninfer::targets::qwen3_6::frontend_internal::sha256;
    using ninfer::targets::qwen3_6::frontend_internal::sha256_hex;
    return sha256_hex(sha256(std::string_view(fingerprint_input)));
}

std::string parse_nvidia_driver_release_text(std::string_view text) {
    auto valid_release = [](std::string_view token) {
        int dots = 0;
        bool previous_dot = true;
        for (char c : token) {
            if (c == '.') {
                if (previous_dot) { return false; }
                previous_dot = true;
                ++dots;
            } else if (std::isdigit(static_cast<unsigned char>(c))) {
                previous_dot = false;
            } else {
                return false;
            }
        }
        return dots >= 2 && !previous_dot;
    };

    std::size_t start = 0;
    while (start < text.size()) {
        while (start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[start]))) {
            ++start;
        }
        const std::size_t end = text.find_first_of(" \t\r\n", start);
        const std::string_view token =
            end == std::string_view::npos ? text.substr(start) : text.substr(start, end - start);
        if (valid_release(token)) { return std::string(token); }
        if (end == std::string_view::npos) { break; }
        start = end + 1;
    }
    return {};
}

std::string collect_nvidia_driver_release() {
    const auto read_file = [](const char* path) -> std::string {
        std::ifstream in(path);
        if (!in) { return {}; }
        std::ostringstream out;
        out << in.rdbuf();
        return out.str();
    };
    if (std::string release =
            parse_nvidia_driver_release_text(read_file("/sys/module/nvidia/version"));
        !release.empty()) {
        return release;
    }
    return parse_nvidia_driver_release_text(read_file("/proc/driver/nvidia/version"));
}

// Splits the ";"-joined deterministic audit string produced by CMake (see
// calibration_build_id.h.in / CMakeLists.txt sec 4) into individual tokens. Empty input yields an
// empty vector -- compile_flags may legitimately be empty when no extra flags are set.
std::vector<std::string> split_compile_flags_audit(std::string_view audit) {
    std::vector<std::string> flags;
    std::size_t start = 0;
    while (start <= audit.size()) {
        const std::size_t sep = audit.find(';', start);
        const std::string_view token =
            sep == std::string_view::npos ? audit.substr(start) : audit.substr(start, sep - start);
        if (!token.empty()) { flags.emplace_back(token); }
        if (sep == std::string_view::npos) { break; }
        start = sep + 1;
    }
    return flags;
}

ninfer::targets::calibration::BuildIdentity collect_build_identity() {
    ninfer::targets::calibration::BuildIdentity build;
    build.compile_flags = split_compile_flags_audit(kNinferCompileFlagsAudit);
#if defined(NINFER_SM8X_COMPAT)
    build.compile_flags.push_back("NINFER_SM8X_COMPAT=1");
#endif
    build.cuda_arch_list = kNinferCudaArchList;
    build.build_type = kNinferBuildType;
    build.build_id = build_id_from_input(build_fingerprint_input());
    return build;
}

std::string sha256_hex_of_file(const std::string& path) {
    std::error_code fs_error;
    if (!fs::is_regular_file(path, fs_error) || fs_error) {
        throw std::runtime_error("artifact path is not a readable regular file: " + path);
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("failed to open artifact for hashing: " + path); }

    // Bounded-memory streaming hash: O(chunk) host memory regardless of artifact size (sec 2).
    // Fixed 1 MiB chunk buffer, read to EOF with explicit read-error handling, fail closed.
    using ninfer::targets::qwen3_6::frontend_internal::Sha256Context;
    using ninfer::targets::qwen3_6::frontend_internal::sha256_hex;
    constexpr std::size_t kChunkSize = 1ULL << 20;
    std::vector<char> chunk(kChunkSize);
    Sha256Context ctx;
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const std::streamsize got = file.gcount();
        if (got > 0) {
            ctx.update(std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(chunk.data()), static_cast<std::size_t>(got)));
        }
        if (file.bad()) { throw std::runtime_error("failed to read artifact for hashing: " + path); }
        if (file.eof()) { break; }
    }
    return sha256_hex(ctx.finish());
}

ArtifactIdentity collect_artifact_identity(const std::string& path, std::string target_id,
                                            std::string weights_id, std::string container_version) {
    std::error_code fs_error;
    if (!fs::is_regular_file(path, fs_error) || fs_error) {
        throw std::runtime_error("artifact path is not a readable regular file: " + path);
    }
    const std::uintmax_t size = fs::file_size(path, fs_error);
    if (fs_error) { throw std::runtime_error("failed to stat artifact: " + path); }

    ArtifactIdentity artifact;
    artifact.path = path;
    artifact.name = fs::path(path).filename().string();
    artifact.size_bytes = static_cast<std::uint64_t>(size);
    artifact.sha256 = sha256_hex_of_file(path); // throws on failure -- fail closed, no partial id.
    artifact.target_id = std::move(target_id);
    artifact.weights_id = std::move(weights_id);
    artifact.container_version = std::move(container_version);
    return artifact;
}

IdentitySnapshotResult assemble_identity_snapshot(
    const std::string& source_root, const std::string& artifact_path, std::string target_id,
    std::string weights_id, std::string container_version,
    std::vector<ninfer::targets::calibration::DeviceIdentity> devices,
    std::vector<CollectionDiagnostic> device_diagnostics,
    ninfer::targets::calibration::SoftwareIdentity software,
    std::vector<CollectionDiagnostic> software_diagnostics, std::string branch_name,
    std::string hostname, std::string os_description, std::string command_line) {
    IdentitySnapshotResult result;

    result.identity.source = collect_source_identity(source_root);
    if (!result.identity.source.commit_known()) {
        result.diagnostics.push_back(
            {"source", "git commit unresolved for source_root=" + source_root, true});
    } else if (result.identity.source.state == SourceState::Dirty) {
        result.diagnostics.push_back(
            {"source", "source tree is dirty; cannot be qualified calibration evidence", false});
    } else if (result.identity.source.state == SourceState::Unknown) {
        result.diagnostics.push_back(
            {"source", "source state could not be determined", true});
    }

    result.identity.build = collect_build_identity();
    if (result.identity.build.build_id.empty()) {
        result.diagnostics.push_back({"build", "build_id collector produced an empty id", true});
    }

    try {
        result.identity.artifact =
            collect_artifact_identity(artifact_path, std::move(target_id), std::move(weights_id),
                                       std::move(container_version));
    } catch (const std::exception& e) {
        result.diagnostics.push_back({"artifact", e.what(), true});
    }

    result.identity.devices = std::move(devices);
    for (std::size_t i = 0; i < result.identity.devices.size(); ++i) {
        if (!result.identity.devices[i].uuid_known()) {
            result.diagnostics.push_back(
                {"device[" + std::to_string(i) + "]", "device uuid unavailable", true});
        }
    }
    for (CollectionDiagnostic& d : device_diagnostics) { result.diagnostics.push_back(std::move(d)); }

    result.identity.software = std::move(software);
    if (result.identity.software.driver_version.empty() ||
        result.identity.software.cuda_runtime_version.empty() ||
        result.identity.software.cuda_toolkit_version.empty()) {
        result.diagnostics.push_back(
            {"software", "one or more software identity fields unavailable", true});
    }
    for (CollectionDiagnostic& d : software_diagnostics) { result.diagnostics.push_back(std::move(d)); }

    result.identity.branch_name = std::move(branch_name);
    result.identity.hostname = std::move(hostname);
    result.identity.os_description = std::move(os_description);
    result.identity.command_line = std::move(command_line);

    return result;
}

} // namespace ninfer::targets::calibration_collect

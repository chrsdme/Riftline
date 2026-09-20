#pragma once

// M5.4b sec 7-9, 11: host-only (CUDA-free) collectors for source/build/artifact identity, plus the
// identity-snapshot assembler that combines them with caller-supplied device/software identity
// (sec 10, CUDA-only, lives in calibration_collect_cuda.cu) without this header ever depending on
// CUDA. Every collector here is fail-closed: a collector failure never produces apparently-valid
// empty metadata -- see CollectionDiagnostics.

#include "targets/calibration.h"

#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::calibration_collect {

// ---------------------------------------------------------------------------------------------
// sec 7: live source identity collector
// ---------------------------------------------------------------------------------------------

// Collects SourceIdentity for the explicitly supplied `source_root`. Never uses any other
// directory as authority (in particular, never the outer /media/storage/ninfer-lab/.git) --
// enforced by comparing `git -C source_root rev-parse --show-toplevel` against the realpath of
// `source_root` itself; a mismatch (e.g. source_root is a subdirectory of some unrelated repo, or
// not a repo at all) yields Unknown with no commit, never a borrowed parent identity.
//
// Untracked files count toward Dirty (fail-closed reading of `git status --porcelain`): an
// uncommitted-but-untracked file is exactly the kind of provenance gap qualified evidence must
// reject.
[[nodiscard]] ninfer::targets::calibration::SourceIdentity collect_source_identity(
    const std::string& source_root);

// ---------------------------------------------------------------------------------------------
// sec 8: build identity collector
// ---------------------------------------------------------------------------------------------

// Returns the canonicalized build-fingerprint input string baked in at configure time (see
// src/targets/calibration_build_id.h.in). Exposed separately from build_id() so tests can assert
// the two are related by sha256_hex without needing a live CMake reconfigure.
[[nodiscard]] std::string build_fingerprint_input();

// build_id = sha256_hex(canonical build-fingerprint input string). Deterministic, no timestamp/
// UUID, differs whenever a kernel-affecting configure input (compiler id/version, CUDA compiler
// version, sorted CUDA arch list, build type, NINFER_SM8X_COMPAT) changes. Exposed as a pure
// function of an input string (not only of the compiled-in constant) so tests can assert
// change-sensitivity without reconfiguring.
[[nodiscard]] std::string build_id_from_input(const std::string& fingerprint_input);

// Convenience: build_id_from_input(build_fingerprint_input()) for this compiled binary.
[[nodiscard]] ninfer::targets::calibration::BuildIdentity collect_build_identity();

// Parses an installed NVIDIA kernel-driver release from Linux driver-version text. Accepts exact
// NVIDIA release tokens like "595.91.07" and rejects CUDA API compatibility versions like "13.2".
[[nodiscard]] std::string parse_nvidia_driver_release_text(std::string_view text);

// Linux-only NVIDIA kernel-driver release collector. Empty string means fail closed: the exact
// release could not be obtained and callers must emit a failed diagnostic.
[[nodiscard]] std::string collect_nvidia_driver_release();

// ---------------------------------------------------------------------------------------------
// sec 9: artifact identity + SHA256
// ---------------------------------------------------------------------------------------------

// Whole-file-read SHA256: the artifact file is read into memory in one pass and hashed with the
// existing ninfer::targets::qwen3_6::frontend_internal::sha256(). Chosen over an incremental
// streaming variant for simplicity (decision doc rung 6/7: reuse before inventing an incremental
// API the reused implementation doesn't offer) -- calibration artifacts here are the .ninfer
// container files this lab currently exercises (multi-GB but not so large that a single
// contiguous read is unreasonable on this workstation's RAM). If M5.4c later hashes artifacts far
// larger than fits comfortably in memory, add a streaming variant then; this function's signature
// (path in, hash out) does not need to change to do that.
//
// Fails closed: throws std::runtime_error with a descriptive message if the path does not exist,
// is not a regular file, or cannot be read in full. Never returns a partial/best-effort hash.
[[nodiscard]] std::string sha256_hex_of_file(const std::string& path);

// Populates ArtifactIdentity.{path,name,size_bytes,sha256} from the live file at `path`, plus the
// caller-supplied target_id/weights_id/container_version (NEVER inferred from the filename --
// caller's responsibility). Throws std::runtime_error on any read/hash failure; never returns a
// record with path+size but no verified sha256.
[[nodiscard]] ninfer::targets::calibration::ArtifactIdentity collect_artifact_identity(
    const std::string& path, std::string target_id, std::string weights_id,
    std::string container_version);

// ---------------------------------------------------------------------------------------------
// sec 11: identity snapshot assembler
// ---------------------------------------------------------------------------------------------

// One collection diagnostic: which identity component, what happened, and why. `failed` marks a
// collector that ran and hit an error (e.g. artifact file unreadable); a collector that simply has
// nothing to report (e.g. no devices were requested) is not a failure and does not appear here.
struct CollectionDiagnostic {
    std::string component; // e.g. "source", "build", "artifact", "device[0]", "software".
    std::string detail;    // human-readable reason.
    bool failed = false;   // true => collector error; false => informational (field genuinely
                           // unavailable, e.g. commit unresolved on an Unknown source state).
};

struct IdentitySnapshotResult {
    ninfer::targets::calibration::CalibrationIdentity identity;
    std::vector<CollectionDiagnostic> diagnostics;

    // True iff no diagnostic in `diagnostics` has failed == true. Does NOT mean the identity
    // qualifies as calibration evidence -- callers must still check missing_requirements() /
    // is_qualified_calibration_evidence() on the record built from this identity. This only
    // answers "did every collector that ran succeed," which is a narrower, more mechanical
    // question than qualification.
    [[nodiscard]] bool all_collectors_succeeded() const noexcept {
        for (const CollectionDiagnostic& d : diagnostics) {
            if (d.failed) { return false; }
        }
        return true;
    }
};

// Assembles a CalibrationIdentity from already-collected pieces (decision 3: this function takes
// no CUDA dependency and calls no CUDA API itself -- device/software identity must be collected by
// the caller, e.g. via calibration_collect_cuda.cu's collect_device_identity/
// collect_software_identity, and passed in here). This is what lets the host test drive the
// three-way collector-failed / genuinely-unavailable / collected-successfully distinction with
// plain fixtures instead of a CUDA fake.
//
// `devices` and `software` are accepted as already-collected values plus their own diagnostics
// (device_diagnostics indexed the same as `devices`); this function does not second-guess them,
// it only folds them into the returned diagnostics list under component names "device[N]" /
// "software".
[[nodiscard]] IdentitySnapshotResult assemble_identity_snapshot(
    const std::string& source_root, const std::string& artifact_path, std::string target_id,
    std::string weights_id, std::string container_version,
    std::vector<ninfer::targets::calibration::DeviceIdentity> devices,
    std::vector<CollectionDiagnostic> device_diagnostics,
    ninfer::targets::calibration::SoftwareIdentity software,
    std::vector<CollectionDiagnostic> software_diagnostics, std::string branch_name = {},
    std::string hostname = {}, std::string os_description = {}, std::string command_line = {});

} // namespace ninfer::targets::calibration_collect

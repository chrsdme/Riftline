// M5.4b: host-only test for the calibration collection infrastructure (sec 27's allowed test
// list). Covers: source collector (temp git repo clean/dirty/non-git), deterministic build
// identity, SHA256 known vectors + temp file, writer temp-dir behavior incl. no-overwrite, corpus
// corrupt-line isolation, staleness classifier matrix, summary percentiles + grouping-key
// separation + mixed-unit rejection, payload geometry derivation + overflow safety, record emitter
// refusing to fabricate Valid. No CUDA, no CTest, no live transfer/bench execution.

#include "targets/calibration.h"
#include "targets/calibration_collect.h"
#include "targets/calibration_io.h"
#include "targets/calibration_measurement.h"
#include "targets/calibration_summary.h"
#include "targets/qwen3_6/impl/frontend/digest.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

namespace fs = std::filesystem;

fs::path make_temp_dir(const std::string& label) {
    fs::path dir = fs::temp_directory_path() /
                   ("ninfer_m54b_" + label + "_" + std::to_string(::getpid()) + "_" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(&label)));
    fs::create_directories(dir);
    return dir;
}

int run_shell(const std::string& command) { return std::system(command.c_str()); }

using namespace ninfer::targets::calibration;
using namespace ninfer::targets::calibration_collect;
using namespace ninfer::targets::calibration_io;
using namespace ninfer::targets::calibration_summary;
using namespace ninfer::targets::calibration_measurement;

// ---- fixtures shared across the staleness/summary sections -----------------------------------

CalibrationIdentity make_complete_identity(const std::string& commit = "aa11bb22cc33dd44ee55ff667788990011223344",
                                            const std::string& build_id = "build-fingerprint-aaa",
                                            const std::string& artifact_sha = "sha-aaa",
                                            const std::string& device_uuid = "GPU-aaaa") {
    CalibrationIdentity identity;
    identity.source.commit = commit;
    identity.source.state = SourceState::Clean;
    identity.build.cuda_arch_list = "sm_86;sm_120a";
    identity.build.build_type = "Release";
    identity.build.build_id = build_id;
    identity.artifact.path = "/tmp/fake.ninfer";
    identity.artifact.name = "fake.ninfer";
    identity.artifact.sha256 = artifact_sha;
    identity.artifact.target_id = "qwen3_6_27b";
    identity.artifact.weights_id = "groupwise-int8";
    identity.artifact.size_bytes = 123;
    DeviceIdentity device;
    device.logical_slot = 0;
    device.physical_index = 0;
    device.uuid = device_uuid;
    device.model_name = "NVIDIA GeForce RTX 5060 Ti";
    device.compute_capability_major = 12;
    device.compute_capability_minor = 0;
    device.vram_bytes = 16ULL * 1024 * 1024 * 1024;
    identity.devices = {device};
    identity.software.driver_version = "550.90.07";
    identity.software.cuda_runtime_version = "12.4";
    identity.software.cuda_toolkit_version = "12.4.131";
    identity.branch_name = "phase2-layer-parity-debug";
    identity.hostname = "ninfer-lab";
    identity.command_line = "ninfer_bench";
    return identity;
}

CalibrationRecord make_measured_record(const CalibrationIdentity& identity, double value,
                                        std::uint64_t timestamp_ms, const std::string& unit = "us") {
    ObservationInput input;
    input.identity = identity;
    input.kind = RecordKind::Operation;
    input.operation.family = OperationFamily::Mlp;
    input.operation.phase = ExecutionPhase::Prefill;
    input.operation.device_logical_slot = 0;
    input.operation.shape.rows = 128;
    input.operation.shape.cols = 5120;
    input.operation.dtype = "bf16";
    input.operation.quant_route = "none";
    input.operation.execution_mode = ExecutionMode::Eager;
    input.operation.warm_state = WarmState::Warm;
    input.run_id = "run-1";
    input.observation_id = "obs-1";
    input.sample_index = 0;
    input.value = value;
    input.unit = unit;
    input.timestamp_unix_ms = timestamp_ms;
    input.evidence = EvidenceClass::Measured;
    input.requested_validity = ValidityState::Valid;
    input.correctness = CorrectnessStatus::Pass;
    return emit_record(input).record;
}

} // namespace

int main() {
    // sec 3: deliberately NOT setting GIT_CONFIG_GLOBAL/GIT_CONFIG_SYSTEM here in main(). Doing so
    // used to mask the fact that production `run_process()` did not neutralize the git child's
    // environment itself -- every test below only proved the TEST HARNESS's env was hermetic, not
    // that the collector was. Production `run_process()` now sets GIT_CONFIG_GLOBAL/
    // GIT_CONFIG_SYSTEM on the child right after fork(), so `collect_source_identity()` is
    // hermetic on its own; the adversarial test below (sec 3) proves that directly.
    // `run_shell()` git commands below still run through this process's ordinary (unmodified)
    // environment/shell git config, which is fine for the plain clean/dirty/non-git fixtures.

    // ---- sec 7: source collector -------------------------------------------------------------
    {
        const fs::path repo = make_temp_dir("git_clean");
        run_shell("git -C '" + repo.string() +
                  "' -c init.defaultBranch=main init -q "
                  "&& git -C '" + repo.string() +
                  "' -c user.email=t@t -c user.name=t -c commit.gpgsign=false "
                  "commit --allow-empty -q -m init");
        const SourceIdentity clean = collect_source_identity(repo.string());
        check(clean.commit_known(), "clean temp git repo should resolve a commit");
        check(clean.state == SourceState::Clean, "clean temp git repo should report Clean");

        std::ofstream(repo / "README.md") << "tracked file\n";
        run_shell("git -C '" + repo.string() +
                  "' -c user.email=t@t -c user.name=t add README.md && git -C '" + repo.string() +
                  "' -c user.email=t@t -c user.name=t -c commit.gpgsign=false commit -q -m add");
        std::ofstream(repo / "README.md", std::ios::app) << "modified\n";
        const SourceIdentity dirty = collect_source_identity(repo.string());
        check(dirty.commit_known(), "dirty temp git repo should still resolve a commit");
        check(dirty.state == SourceState::Dirty, "modified tracked file should report Dirty");

        const fs::path non_git = make_temp_dir("non_git");
        const SourceIdentity unknown = collect_source_identity(non_git.string());
        check(!unknown.commit_known(), "non-git root must not fabricate a commit");
        check(unknown.state == SourceState::Unknown, "non-git root must report Unknown");

        // Outer-.git guard: a SUBDIRECTORY of the temp repo is not itself a repo root. Without the
        // toplevel-realpath check in collect_source_identity, `git -C <subdir> rev-parse HEAD`
        // would happily resolve the PARENT repo's commit -- exactly the "do not use an outer .git
        // as authority" case sec 7 requires guarding against. Assert it stays Unknown.
        const fs::path nested = repo / "sub";
        fs::create_directories(nested);
        const SourceIdentity nested_identity = collect_source_identity(nested.string());
        check(!nested_identity.commit_known(),
              "a subdirectory of a repo that is not itself a repo root must not borrow the parent's commit");
        check(nested_identity.state == SourceState::Unknown,
              "a non-repo-root subdirectory must report Unknown, not the parent repo's state");

        fs::remove_all(repo);
        fs::remove_all(non_git);
    }

    // ---- sec 3: adversarial hostile-git-config test --------------------------------------------
    // Proves collect_source_identity() is hermetic ON ITS OWN, not merely because this test
    // process's ambient environment happens to be clean. A hostile global git config with
    // `[status] showUntrackedFiles = no` can hide an untracked file from a plain
    // `git status --porcelain` call; production must still report Dirty because run_process()
    // neutralizes GIT_CONFIG_GLOBAL/GIT_CONFIG_SYSTEM in its OWN child, unconditionally, regardless
    // of what this test process's environment contains.
    {
        const fs::path repo = make_temp_dir("git_hostile");
        run_shell("git -C '" + repo.string() +
                  "' -c init.defaultBranch=main init -q "
                  "&& git -C '" + repo.string() +
                  "' -c user.email=t@t -c user.name=t -c commit.gpgsign=false "
                  "commit --allow-empty -q -m init");
        std::ofstream(repo / "untracked.txt") << "not tracked, not ignored\n";

        // Hostile global config that would suppress untracked files in a normal `git status`.
        const fs::path hostile_home = make_temp_dir("hostile_home");
        const fs::path hostile_config = hostile_home / "hostile.gitconfig";
        std::ofstream(hostile_config) << "[status]\n    showUntrackedFiles = no\n";

        // Arrange THIS PROCESS's environment (and therefore std::system()'s child, and any
        // run_process() call that did NOT itself override GIT_CONFIG_GLOBAL) so the hostile config
        // would apply -- this is the "hostile ambient environment" sec 3 asks us to construct.
        const std::string saved_global = [] {
            const char* v = std::getenv("GIT_CONFIG_GLOBAL");
            return v != nullptr ? std::string(v) : std::string();
        }();
        const bool had_saved_global = std::getenv("GIT_CONFIG_GLOBAL") != nullptr;
        ::setenv("GIT_CONFIG_GLOBAL", hostile_config.c_str(), 1);

        // Positive control FIRST, with the hostile config explicitly neutralized: proves git
        // actually runs successfully here and genuinely sees an untracked, non-ignored file --
        // without this, an empty negative-control result below would be indistinguishable from
        // "git silently failed/produced nothing for an unrelated reason." Written OUTSIDE the repo
        // so it never becomes a second untracked file in the very repo under test.
        const std::string positive_control_path = (hostile_home / "positive_control.txt").string();
        const int positive_rc =
            run_shell("GIT_CONFIG_GLOBAL=/dev/null git -C '" + repo.string() +
                      "' status --porcelain > '" + positive_control_path + "' 2>/dev/null");
        check(positive_rc == 0, "positive control: neutralized `git status --porcelain` must succeed");
        std::ifstream positive_stream(positive_control_path);
        std::ostringstream positive_buffer;
        positive_buffer << positive_stream.rdbuf();
        check(!positive_buffer.str().empty(),
              "positive control: with the hostile config neutralized, `git status --porcelain` "
              "must report the untracked file (proves the fixture itself is valid)");

        // Negative control: a direct, unprotected `git status --porcelain` (via this hostile
        // environment, through the shell) must indeed hide the untracked file -- proving the
        // hostile config is actually effective, now that the positive control has ruled out a
        // vacuously-empty result from git simply not running.
        const std::string control_output_path = (hostile_home / "control_status.txt").string();
        const int control_rc = run_shell("git -C '" + repo.string() + "' status --porcelain > '" +
                                          control_output_path + "' 2>/dev/null");
        check(control_rc == 0, "negative control: `git status --porcelain` invocation itself must succeed");
        std::ifstream control_stream(control_output_path);
        std::ostringstream control_buffer;
        control_buffer << control_stream.rdbuf();
        check(control_buffer.str().empty(),
              "negative control: hostile global git config must hide the untracked file from plain "
              "`git status --porcelain` (test setup invariant)");

        // Production path: collect_source_identity() spawns its OWN child via run_process(), which
        // must set GIT_CONFIG_GLOBAL=/dev/null in that child regardless of the hostile value this
        // test process just set -- so it must still see and report the untracked file as Dirty.
        const SourceIdentity hostile_result = collect_source_identity(repo.string());
        check(hostile_result.commit_known(),
              "hostile-config repo should still resolve a commit");
        check(hostile_result.state == SourceState::Dirty,
              "collect_source_identity() must neutralize the hostile global git config itself and "
              "still report Dirty for an untracked non-ignored file");

        if (had_saved_global) {
            ::setenv("GIT_CONFIG_GLOBAL", saved_global.c_str(), 1);
        } else {
            ::unsetenv("GIT_CONFIG_GLOBAL");
        }
        fs::remove_all(repo);
        fs::remove_all(hostile_home);
    }

    // ---- sec 8: deterministic build identity ---------------------------------------------------
    {
        const std::string input_a = build_fingerprint_input();
        const std::string id_a1 = build_id_from_input(input_a);
        const std::string id_a2 = build_id_from_input(input_a);
        check(id_a1 == id_a2, "build_id must be stable across repeated calls with the same input");
        check(!id_a1.empty(), "build_id must not be empty");

        const std::string input_b = input_a + ";ninfer_sm8x_compat=CHANGED";
        const std::string id_b = build_id_from_input(input_b);
        check(id_b != id_a1, "build_id must differ when a kernel-affecting input changes");

        const BuildIdentity collected = collect_build_identity();
        check(collected.build_id == build_id_from_input(build_fingerprint_input()),
              "collect_build_identity should hash the live fingerprint input");
        check(!collected.cuda_arch_list.empty(),
              "collect_build_identity must populate cuda_arch_list, not leave it unresolved");
        check(!collected.build_type.empty(),
              "collect_build_identity must populate build_type, not leave it unresolved");
    }

    // ---- M5.4c-1 driver identity semantic fix --------------------------------------------------
    {
        check(parse_nvidia_driver_release_text("595.91.07\n") == "595.91.07",
              "exact NVIDIA driver release should parse from /sys-style text");
        check(parse_nvidia_driver_release_text(
                  "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.91.07  Release Build") ==
                  "595.91.07",
              "exact NVIDIA driver release should parse from /proc-style text");
        check(parse_nvidia_driver_release_text("").empty(),
              "empty NVIDIA driver release source must fail closed");
        check(parse_nvidia_driver_release_text("NVRM version: unknown\n").empty(),
              "malformed NVIDIA driver release source must fail closed");
        check(parse_nvidia_driver_release_text("13.2\n").empty(),
              "CUDA API compatibility version must not be accepted as driver_version");
    }

    // ---- sec 9: SHA256 known vectors + temp file ------------------------------------------------
    {
        using ninfer::targets::qwen3_6::frontend_internal::sha256;
        using ninfer::targets::qwen3_6::frontend_internal::sha256_hex;
        const std::string empty_hash = sha256_hex(sha256(std::string_view("")));
        check(empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "sha256(empty) known-vector mismatch");
        const std::string abc_hash = sha256_hex(sha256(std::string_view("abc")));
        check(abc_hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
              "sha256(\"abc\") known-vector mismatch");

        const fs::path temp_dir = make_temp_dir("sha_file");
        const fs::path temp_file = temp_dir / "artifact.bin";
        std::ofstream(temp_file, std::ios::binary) << "abc";
        const std::string file_hash = sha256_hex_of_file(temp_file.string());
        check(file_hash == abc_hash, "sha256_hex_of_file(\"abc\" file) must match the in-memory hash");

        const ArtifactIdentity artifact =
            collect_artifact_identity(temp_file.string(), "qwen3_6_27b", "groupwise-int8", "v1");
        check(artifact.sha256 == abc_hash, "collect_artifact_identity sha256 mismatch");
        check(artifact.size_bytes == 3, "collect_artifact_identity size_bytes mismatch");
        check(artifact.target_id == "qwen3_6_27b", "artifact target_id must be caller-supplied verbatim");

        bool threw = false;
        try {
            (void)sha256_hex_of_file((temp_dir / "missing.bin").string());
        } catch (const std::exception&) { threw = true; }
        check(threw, "sha256_hex_of_file on a missing file must throw, not fail closed silently");

        // sec 2: a file larger than one SHA256 block (64 bytes) exercises the multi-block path.
        {
            std::string bigger_than_one_block(200, 'x');
            const fs::path bigger_file = temp_dir / "bigger.bin";
            std::ofstream(bigger_file, std::ios::binary) << bigger_than_one_block;
            const std::string streamed = sha256_hex_of_file(bigger_file.string());
            const std::string in_memory =
                sha256_hex(sha256(std::string_view(bigger_than_one_block)));
            check(streamed == in_memory,
                  "sec2: file >1 SHA256 block must match streaming vs whole-buffer hash");
        }

        // sec 2: multi-megabyte temp file, streamed (sha256_hex_of_file, chunked reads spanning
        // multiple 1 MiB chunk-buffer refills) vs whole-buffer (sha256(string_view)) must agree,
        // and a one-byte mutation must change the digest.
        {
            constexpr std::size_t kMultiMbSize = 3 * 1024 * 1024 + 777; // not chunk-aligned.
            std::string multi_mb(kMultiMbSize, '\0');
            for (std::size_t i = 0; i < multi_mb.size(); ++i) {
                multi_mb[i] = static_cast<char>(i % 251);
            }
            const fs::path multi_mb_file = temp_dir / "multi_mb.bin";
            std::ofstream(multi_mb_file, std::ios::binary).write(multi_mb.data(), static_cast<std::streamsize>(multi_mb.size()));
            const std::string streamed_hash = sha256_hex_of_file(multi_mb_file.string());
            const std::string in_memory_hash = sha256_hex(sha256(std::string_view(multi_mb)));
            check(streamed_hash == in_memory_hash,
                  "sec2: multi-MB streamed hash must equal the whole-buffer hash");

            std::string mutated = multi_mb;
            mutated[kMultiMbSize / 2] = static_cast<char>(mutated[kMultiMbSize / 2] ^ 0xFF);
            const std::string mutated_hash = sha256_hex(sha256(std::string_view(mutated)));
            check(mutated_hash != in_memory_hash,
                  "sec2: a single-byte mutation must change the digest");
        }

        fs::remove_all(temp_dir);
    }

    // ---- sec 12: raw JSONL writer temp-dir behavior + no-overwrite -----------------------------
    {
        const fs::path root = make_temp_dir("writer");
        const CalibrationIdentity identity = make_complete_identity();
        const CalibrationRecord record = make_measured_record(identity, 12.5, 1000);

        RawWriter writer(root.string(), "run-alpha");
        writer.append(record);
        writer.append(record);
        writer.close();
        check(writer.records_written() == 2, "writer should report 2 records written");
        check(fs::exists(writer.file_path()), "writer's reported file_path must exist");

        bool threw_on_bad_run_id = false;
        try {
            RawWriter bad(root.string(), "../escape");
        } catch (const std::exception&) { threw_on_bad_run_id = true; }
        check(threw_on_bad_run_id, "run_id with path-traversal characters must be rejected");
        check(!is_valid_run_id("../escape"), "is_valid_run_id must reject '..' segments");
        check(is_valid_run_id("run-alpha_1.2"), "is_valid_run_id must accept alnum/_/./- run ids");

        const std::uint64_t first_size = fs::file_size(writer.file_path());
        check(first_size > 0, "writer's run file must be non-empty after append+close");

        // No-overwrite, tested deterministically on the writer's OWN reported path (not a second
        // writer racing on timestamp+run_id): re-attempting O_CREAT|O_EXCL at that exact filename
        // must fail, which is precisely the guard RawWriter's constructor relies on to refuse
        // clobbering a prior run file.
        const int dup_fd = ::open(writer.file_path().c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        check(dup_fd < 0, "O_CREAT|O_EXCL must refuse to recreate an existing run file");
        if (dup_fd >= 0) { ::close(dup_fd); }
        check(fs::file_size(writer.file_path()) == first_size,
              "the existing run file's contents must be untouched after the guard check");

        fs::remove_all(root);
    }

    // ---- sec 13: corpus reader / corrupt-line isolation -----------------------------------------
    {
        const fs::path root = make_temp_dir("corpus");
        const fs::path corpus_path = root / "corpus.jsonl";
        const CalibrationIdentity identity = make_complete_identity();
        const CalibrationRecord good_a = make_measured_record(identity, 1.0, 1000);
        const CalibrationRecord good_b = make_measured_record(identity, 2.0, 1001);
        {
            std::ofstream out(corpus_path, std::ios::binary);
            out << serialize(good_a) << '\n';
            out << "{not valid json at all\n";
            out << serialize(good_b) << '\n';
        }
        const CorpusParseResult parsed = read_corpus_file(corpus_path.string());
        check(parsed.records.size() == 2, "corrupt line must not discard the two valid neighbors");
        check(parsed.line_errors.size() == 1, "exactly one corrupt line must be reported");
        if (!parsed.line_errors.empty()) {
            check(parsed.line_errors[0].first == 1, "corrupt line index should be line 1 (0-based)");
        }
        fs::remove_all(root);
    }

    // ---- sec 14: staleness / compatibility classifier matrix -----------------------------------
    {
        const CalibrationIdentity base = make_complete_identity();

        check(classify_compatibility(base, base).status == Compatibility::Compatible,
              "exact identity match must be Compatible");

        auto mutate_and_check = [&](auto mutator, const char* label) {
            CalibrationIdentity current = base;
            mutator(current);
            const CompatibilityResult result = classify_compatibility(base, current);
            check(result.status == Compatibility::StaleIncompatible,
                  std::string(label) + " changed must be StaleIncompatible");
        };
        mutate_and_check([](CalibrationIdentity& id) { id.source.commit = "different-commit-000000000000000000000000"; },
                         "source_commit");
        mutate_and_check([](CalibrationIdentity& id) { id.build.build_id = "different-build-id"; }, "build_id");
        mutate_and_check([](CalibrationIdentity& id) { id.artifact.sha256 = "different-sha"; }, "artifact_sha256");
        mutate_and_check([](CalibrationIdentity& id) { id.devices[0].uuid = "GPU-remapped"; }, "device_uuid");
        mutate_and_check([](CalibrationIdentity& id) { id.software.driver_version = "999.0"; }, "driver_version");
        mutate_and_check([](CalibrationIdentity& id) { id.software.cuda_runtime_version = "9.9"; },
                         "cuda_runtime_version");
        mutate_and_check([](CalibrationIdentity& id) { id.software.cuda_toolkit_version = "9.9.9"; },
                         "cuda_toolkit_version");

        // Device slot->uuid REMAP with the same uuid set: two devices, swap which slot owns which
        // uuid. Must be caught as stale (pairing, not set).
        {
            CalibrationIdentity two_device_base = base;
            DeviceIdentity second = two_device_base.devices[0];
            second.logical_slot = 1;
            second.uuid = "GPU-bbbb";
            two_device_base.devices.push_back(second);
            CalibrationIdentity remapped = two_device_base;
            std::swap(remapped.devices[0].uuid, remapped.devices[1].uuid);
            const CompatibilityResult result = classify_compatibility(two_device_base, remapped);
            check(result.status == Compatibility::StaleIncompatible,
                  "device logical_slot->uuid remap with same uuid SET must still be stale");
        }

        // Audit-only fields changed alone: still Compatible.
        {
            CalibrationIdentity current = base;
            current.branch_name = "some-other-branch";
            current.hostname = "some-other-host";
            current.command_line = "some other command line";
            const CompatibilityResult result = classify_compatibility(base, current);
            check(result.status == Compatibility::Compatible,
                  "branch/hostname/command_line-only changes must remain Compatible");
        }

        // Incomplete current identity: Unknown/Unqualified, never Compatible.
        {
            CalibrationIdentity incomplete_current; // default-constructed: everything unresolved.
            const CompatibilityResult result = classify_compatibility(base, incomplete_current);
            check(result.status == Compatibility::UnknownUnqualified,
                  "incomplete current identity must be UnknownUnqualified, not Compatible or Stale");
        }

        // sec 5: mutate each required current-identity completeness dimension independently and
        // assert UnknownUnqualified (never Compatible, never StaleIncompatible-via-fallthrough).
        auto require_unqualified = [&](auto mutator, const char* label) {
            CalibrationIdentity current = base;
            mutator(current);
            const CompatibilityResult result = classify_compatibility(base, current);
            check(result.status == Compatibility::UnknownUnqualified,
                  std::string("sec5: incomplete current.") + label + " must be UnknownUnqualified");
        };
        require_unqualified([](CalibrationIdentity& id) { id.source.commit.clear(); }, "source.commit");
        require_unqualified([](CalibrationIdentity& id) { id.source.state = SourceState::Dirty; },
                             "source.state=Dirty");
        require_unqualified([](CalibrationIdentity& id) { id.source.state = SourceState::Unknown; },
                             "source.state=Unknown");
        require_unqualified([](CalibrationIdentity& id) { id.build.build_id.clear(); }, "build.build_id");
        require_unqualified([](CalibrationIdentity& id) { id.build.cuda_arch_list.clear(); },
                             "build.cuda_arch_list");
        require_unqualified([](CalibrationIdentity& id) { id.build.build_type.clear(); },
                             "build.build_type");
        require_unqualified([](CalibrationIdentity& id) { id.artifact.sha256.clear(); },
                             "artifact.sha256");
        require_unqualified([](CalibrationIdentity& id) { id.artifact.target_id.clear(); },
                             "artifact.target_id");
        require_unqualified([](CalibrationIdentity& id) { id.artifact.weights_id.clear(); },
                             "artifact.weights_id");
        require_unqualified([](CalibrationIdentity& id) { id.devices.clear(); }, "devices (empty)");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].logical_slot = -1; },
                             "device.logical_slot");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].physical_index = -1; },
                             "device.physical_index");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].uuid.clear(); }, "device.uuid");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].model_name.clear(); },
                             "device.model_name");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].compute_capability_major = 0; },
                             "device.compute_capability_major");
        require_unqualified([](CalibrationIdentity& id) { id.devices[0].vram_bytes = 0; },
                             "device.vram_bytes");
        require_unqualified([](CalibrationIdentity& id) { id.software.driver_version.clear(); },
                             "software.driver_version");
        require_unqualified([](CalibrationIdentity& id) { id.software.cuda_runtime_version.clear(); },
                             "software.cuda_runtime_version");
        require_unqualified([](CalibrationIdentity& id) { id.software.cuda_toolkit_version.clear(); },
                             "software.cuda_toolkit_version");

        // compute_capability_minor == 0 remains valid: must NOT force UnknownUnqualified.
        {
            CalibrationIdentity current = base;
            current.devices[0].compute_capability_minor = 0;
            const CompatibilityResult result = classify_compatibility(base, current);
            check(result.status == Compatibility::Compatible,
                  "compute_capability_minor == 0 must remain a valid/complete identity");
        }

        // Equivalent completeness applies to STORED identity too.
        {
            CalibrationIdentity incomplete_stored = base;
            incomplete_stored.source.state = SourceState::Dirty;
            const CompatibilityResult result = classify_compatibility(incomplete_stored, base);
            check(result.status == Compatibility::UnknownUnqualified,
                  "sec5: incomplete stored identity must also be UnknownUnqualified");
        }

        // sec 6: device mapping is order-independent -- same slot->uuid mapping, different vector
        // insertion order, must be Compatible (not falsely StaleIncompatible).
        {
            CalibrationIdentity two_device = base;
            DeviceIdentity second = two_device.devices[0];
            second.logical_slot = 1;
            second.uuid = "GPU-bbbb";
            two_device.devices.push_back(second);
            CalibrationIdentity reordered = two_device;
            std::swap(reordered.devices[0], reordered.devices[1]);
            const CompatibilityResult result = classify_compatibility(two_device, reordered);
            check(result.status == Compatibility::Compatible,
                  "sec6: identical logical_slot->uuid mapping in different vector order must be Compatible");
        }

        // sec 6: duplicate logical slot must fail closed to UnknownUnqualified.
        {
            CalibrationIdentity duplicate_slot = base;
            DeviceIdentity dup = duplicate_slot.devices[0];
            dup.uuid = "GPU-duplicate-slot";
            duplicate_slot.devices.push_back(dup); // same logical_slot as devices[0].
            const CompatibilityResult result = classify_compatibility(base, duplicate_slot);
            check(result.status == Compatibility::UnknownUnqualified,
                  "sec6: duplicate logical_slot in an identity must fail closed to UnknownUnqualified");
        }
    }

    // ---- sec 15: summary percentiles (exact expected values) -----------------------------------
    {
        std::vector<double> values;
        for (int i = 1; i <= 10; ++i) { values.push_back(static_cast<double>(i)); }
        check(percentile(values, 0.50) == 5.0, "p50 of 1..10 should be 5 (nearest-rank, 0-based idx 4)");
        check(percentile(values, 0.95) == 9.0, "p95 of 1..10 should be 9 (idx floor(0.95*9)=8 -> value 9)");
        check(percentile(values, 0.99) == 9.0, "p99 of 1..10 should equal p95 under nearest-rank-no-interp");

        std::vector<double> single = {42.0};
        check(percentile(single, 0.50) == 42.0, "percentile of a single-sample vector is that sample");
        check(percentile(single, 0.99) == 42.0, "percentile of a single-sample vector is that sample (p99)");
    }

    // ---- sec 15: grouping-key separation + mixed-unit rejection --------------------------------
    {
        const CalibrationIdentity identity = make_complete_identity();
        std::vector<CalibrationRecord> records;
        for (int i = 0; i < 5; ++i) {
            records.push_back(make_measured_record(identity, 10.0 + i, 1000 + i, "us"));
        }
        // A record identical except for shape.cols: must land in a DIFFERENT group.
        CalibrationRecord different_shape = make_measured_record(identity, 99.0, 2000, "us");
        different_shape.operation.shape.cols = 4096;
        records.push_back(different_shape);
        // A record identical except for unit: must land in a different group (never merged).
        CalibrationRecord different_unit = make_measured_record(identity, 5.0, 3000, "GB/s");
        records.push_back(different_unit);
        // Two records differing only in audit-only fields (branch_name): must land in the SAME
        // group as the first five.
        CalibrationIdentity identity_diff_branch = identity;
        identity_diff_branch.branch_name = "other-branch";
        records.push_back(make_measured_record(identity_diff_branch, 11.0, 4000, "us"));

        const std::vector<CalibrationSummary> summaries = derive_summaries(records, identity, 1);
        check(summaries.size() == 3, "expected 3 groups: base(6 samples), different_shape, different_unit");

        std::uint64_t base_group_count = 0;
        for (const CalibrationSummary& s : summaries) {
            if (s.unit == "GB/s") {
                check(s.stats.count == 1, "GB/s group must be split out, count 1");
            } else if (s.stats.count == 6) {
                base_group_count = s.stats.count;
            }
        }
        check(base_group_count == 6,
              "audit-only branch_name difference must NOT split the group (expected 6-sample group)");

        // Sample-sufficiency: below min_sample_count => PartialInsufficientSamples.
        const std::vector<CalibrationSummary> strict = derive_summaries(records, identity, 100);
        for (const CalibrationSummary& s : strict) {
            check(s.qualification == SummaryQualification::PartialInsufficientSamples,
                  "every group must be Partial when min_sample_count is unreachable");
        }
    }

    // ---- sec 15b: raw schema v1/v2 transfer records never share a summary group ----------------
    {
        CalibrationIdentity identity = make_complete_identity();
        DeviceIdentity gpu1 = identity.devices.front();
        gpu1.logical_slot = 1;
        gpu1.physical_index = 1;
        gpu1.uuid = "GPU-bbbb1111";
        identity.devices.push_back(gpu1);

        auto make_transfer = [&](int schema_version, HostInvolvement host,
                                 TransferRoute route, double value) {
            ObservationInput input;
            input.identity = identity;
            input.identity.schema_version = schema_version;
            input.kind = RecordKind::Transfer;
            input.operation.phase = ExecutionPhase::Transfer;
            input.transfer.direction = TransferDirection::DeviceToDevice;
            input.transfer.source_device_uuid = "GPU-aaaa";
            input.transfer.destination_device_uuid = "GPU-bbbb1111";
            input.transfer.payload_class = TransferPayloadClass::LogitsResult;
            input.transfer.payload_bytes = 4096;
            input.transfer.transfer_api = "cudaMemcpyAsync(cudaMemcpyDeviceToDevice,UVA)";
            input.transfer.sync = TransferSync::Async;
            input.transfer.host_involvement = host;
            input.transfer.route = route;
            input.transfer.overlap = TransferOverlap::NoOverlap;
            input.run_id = "run-transfer";
            input.observation_id = "d2d";
            input.value = value;
            input.unit = "us";
            input.timestamp_unix_ms = 5000;
            input.evidence = EvidenceClass::Measured;
            input.requested_validity = ValidityState::Valid;
            input.correctness = CorrectnessStatus::Pass;
            return emit_record(input).record;
        };

        const CalibrationRecord v1 =
            make_transfer(kSchemaVersionV1, HostInvolvement::None, TransferRoute::CopyEngine, 1.0);
        const CalibrationRecord v2 = make_transfer(kSchemaVersionV2,
                                                   HostInvolvement::DriverManagedStaging,
                                                   TransferRoute::RuntimeManagedMemcpy, 2.0);
        check(is_qualified_calibration_evidence(v1), "v1 transfer fixture should qualify");
        check(is_qualified_calibration_evidence(v2), "v2 transfer fixture should qualify");

        const std::vector<CalibrationSummary> summaries = derive_summaries({v1, v2}, identity, 1);
        check(summaries.size() == 2,
              "otherwise-identical v1/v2 transfer records must produce distinct summary groups");
    }

    // ---- sec 15: Modeled/Projected must never enter the trusted measured distribution ----------
    {
        const CalibrationIdentity identity = make_complete_identity();
        // Five genuinely measured samples, plus one Modeled and one Projected record that are
        // otherwise IDENTICAL (same group key, same unit, same shape). Only the measured five may
        // be summarized: sec 15 forbids mixing derived/extrapolated evidence into the same
        // distribution, and a wrong value here would silently corrupt every M5.5 input.
        std::vector<CalibrationRecord> records;
        for (int i = 0; i < 5; ++i) {
            records.push_back(make_measured_record(identity, 10.0, 1000 + i, "us"));
        }
        CalibrationRecord modeled = make_measured_record(identity, 999.0, 2000, "us");
        modeled.qualification.evidence = EvidenceClass::Modeled;
        records.push_back(modeled);
        CalibrationRecord projected = make_measured_record(identity, 999.0, 2001, "us");
        projected.qualification.evidence = EvidenceClass::Projected;
        records.push_back(projected);

        const std::vector<CalibrationSummary> summaries = derive_summaries(records, identity, 1);
        check(summaries.size() == 1, "Modeled/Projected must not create their own trusted group");
        check(summaries.front().stats.count == 5,
              "only the 5 Measured samples may be summarized (Modeled/Projected excluded)");
        check(summaries.front().stats.max == 10.0,
              "the 999.0 Modeled/Projected values must not reach the measured distribution");
    }

    // ---- sec 16: summary storage round trip -----------------------------------------------------
    {
        const CalibrationIdentity identity = make_complete_identity();
        std::vector<CalibrationRecord> records = {make_measured_record(identity, 1.0, 1000),
                                                  make_measured_record(identity, 2.0, 1001),
                                                  make_measured_record(identity, 3.0, 1002)};
        const std::vector<CalibrationSummary> summaries = derive_summaries(records, identity, 1);
        check(summaries.size() == 1, "expected exactly 1 group for this fixture");
        const std::string line = serialize_summary(summaries[0]);
        check(line.find("\"artifact_type\":\"ninfer_calibration_summary\"") != std::string::npos,
              "summary line must carry the distinct summary artifact_type");
        check(line.find("ninfer_calibration_record") == std::string::npos,
              "summary line must never masquerade as a raw calibration record");
        const CalibrationSummary parsed = parse_summary_line(line);
        check(parsed.stats.count == summaries[0].stats.count, "round-trip count mismatch");
        check(parsed.stats.p50 == summaries[0].stats.p50, "round-trip p50 mismatch");
        check(parsed.group_key == summaries[0].group_key, "round-trip group_key mismatch");
        check(parsed.qualification == summaries[0].qualification, "round-trip qualification mismatch");
        check(parsed.kind == summaries[0].kind, "round-trip record_kind mismatch");
        check(parsed.unit == summaries[0].unit, "round-trip unit mismatch");
        check(parsed.stats.min == summaries[0].stats.min, "round-trip min mismatch");
        check(parsed.stats.max == summaries[0].stats.max, "round-trip max mismatch");
        check(parsed.stats.p95 == summaries[0].stats.p95, "round-trip p95 mismatch");
        check(parsed.stats.p99 == summaries[0].stats.p99, "round-trip p99 mismatch");
        check(parsed.stats.first_timestamp_unix_ms == summaries[0].stats.first_timestamp_unix_ms,
              "round-trip first_timestamp_unix_ms mismatch");
        check(parsed.stats.last_timestamp_unix_ms == summaries[0].stats.last_timestamp_unix_ms,
              "round-trip last_timestamp_unix_ms mismatch");
    }

    // ---- sec 10: summary parser must fail closed on malformed input -----------------------------
    {
        const CalibrationIdentity identity = make_complete_identity();
        const CalibrationRecord good_record = make_measured_record(identity, 7.0, 3000);
        const std::vector<CalibrationSummary> good_summaries =
            derive_summaries({good_record}, identity, 1);
        check(good_summaries.size() == 1, "expected exactly 1 group for the sec 10 fixture");
        const std::string good_line = serialize_summary(good_summaries[0]);

        // Sanity: the valid line must still parse (byte-stable happy path alongside the strict
        // rewrite).
        bool valid_parsed = true;
        try {
            (void)parse_summary_line(good_line);
        } catch (const SummaryParseError&) { valid_parsed = false; }
        check(valid_parsed, "sec10: a genuinely valid summary line must still parse");

        auto expect_rejected = [&](const std::string& malformed_line, const char* label) {
            bool threw = false;
            try {
                (void)parse_summary_line(malformed_line);
            } catch (const SummaryParseError&) { threw = true; }
            check(threw, std::string("sec10: must reject ") + label);
        };

        auto replace_one = [](std::string line, const std::string& from, const std::string& to) {
            const std::size_t pos = line.find(from);
            if (pos != std::string::npos) { line.replace(pos, from.size(), to); }
            return line;
        };
        auto number_for_test = [](double value) {
            std::ostringstream out;
            out.precision(17);
            out << value;
            return out.str();
        };

        expect_rejected(replace_one(good_line, "\"artifact_type\":\"ninfer_calibration_summary\"",
                                     "\"artifact_type\":\"wrong_type\""),
                         "wrong artifact_type");
        expect_rejected(replace_one(good_line, "\"summary_schema_version\":1",
                                     "\"summary_schema_version\":2"),
                         "wrong schema version");
        expect_rejected(replace_one(good_line, "\"summary_schema_version\":1",
                                     "\"summary_schema_version\":1junk"),
                         "schema numeric token with trailing garbage");
        expect_rejected(replace_one(good_line, "\"group_key\":\"" + good_summaries[0].group_key + "\"",
                                     "\"group_key\":\"\""),
                         "empty group_key");
        expect_rejected(replace_one(good_line, "\"group_key\":\"" + good_summaries[0].group_key + "\"",
                                     "\"group_key\":\"not-hex-and-wrong-length\""),
                         "non-hex/wrong-length group_key");
        expect_rejected(replace_one(good_line, "\"record_kind\":\"operation\"",
                                     "\"record_kind\":\"not_a_real_kind\""),
                         "unknown record_kind");
        expect_rejected(replace_one(good_line, "\"unit\":\"" + good_summaries[0].unit + "\"",
                                     "\"unit\":\"\""),
                         "empty unit");
        expect_rejected(replace_one(good_line, "\"count\":1", "\"count\":0"), "count == 0");
        expect_rejected(replace_one(good_line, "\"count\":1", "\"count\":-1"),
                         "negative count");
        expect_rejected(replace_one(good_line, "\"count\":1", "\"count\":1x"),
                         "malformed count with suffix");
        expect_rejected(replace_one(good_line, "\"min\":" + number_for_test(good_summaries[0].stats.min),
                                     "\"min\":nan"),
                         "NaN min");
        expect_rejected(replace_one(good_line, "\"max\":" + number_for_test(good_summaries[0].stats.max),
                                     "\"max\":inf"),
                         "Inf max");
        {
            // Force an ordering violation: min > p50.
            const std::string min_token = number_for_test(good_summaries[0].stats.min);
            const std::string tampered =
                replace_one(good_line, "\"min\":" + min_token, "\"min\":999999");
            expect_rejected(tampered, "min > p50 ordering violation");
        }
        expect_rejected(good_line + "trailing_garbage",
                         "trailing non-whitespace after object");
        expect_rejected(replace_one(good_line, "\"qualification\":\"qualified\"",
                                     "\"qualification\":\"qualified\",\"qualification\":\"qualified\""),
                         "duplicate required key");
        expect_rejected(replace_one(good_line, "\"qualification\":\"qualified\"",
                                     "\"qualification\":\"bogus_token\""),
                         "unknown qualification token");

        // Malformed timestamp ordering: first > last.
        {
            std::string tampered = good_line;
            tampered = replace_one(tampered, "\"first_timestamp_unix_ms\":3000",
                                    "\"first_timestamp_unix_ms\":9999999999");
            expect_rejected(tampered, "first_timestamp_unix_ms > last_timestamp_unix_ms");
        }
    }

    // ---- sec 7/16: actual summary persistence (SummaryWriter), temp dirs only -------------------
    {
        const fs::path root = make_temp_dir("summaries");
        const CalibrationIdentity identity = make_complete_identity();
        std::vector<CalibrationRecord> records = {make_measured_record(identity, 5.0, 2000),
                                                  make_measured_record(identity, 6.0, 2001)};
        const std::vector<CalibrationSummary> summaries = derive_summaries(records, identity, 1);
        check(summaries.size() == 1, "expected exactly 1 group for this fixture");

        SummaryWriter writer(root.string(), "run-summary-alpha");
        writer.append(summaries[0]);
        writer.append(summaries[0]);
        writer.close();
        check(writer.summaries_written() == 2, "SummaryWriter should report 2 summaries written");
        check(fs::exists(writer.file_path()), "SummaryWriter's reported file_path must exist");

        {
            std::ifstream in(writer.file_path());
            std::string first_line;
            std::getline(in, first_line);
            check(!first_line.empty(), "persisted summary file must contain a non-empty first line");
            const CalibrationSummary read_back = parse_summary_line(first_line);
            check(read_back.group_key == summaries[0].group_key,
                  "persisted-then-parsed summary must round-trip group_key");
        }

        // No silent overwrite: O_CREAT|O_EXCL at the writer's own reported path must fail.
        const int dup_fd = ::open(writer.file_path().c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        check(dup_fd < 0, "SummaryWriter's O_CREAT|O_EXCL must refuse to recreate an existing file");
        if (dup_fd >= 0) { ::close(dup_fd); }

        bool threw_on_bad_run_id = false;
        try {
            SummaryWriter bad(root.string(), "../escape");
        } catch (const std::exception&) { threw_on_bad_run_id = true; }
        check(threw_on_bad_run_id, "SummaryWriter must reject a path-traversal run_id");

        fs::remove_all(root);
    }

    // ---- sec 18: payload geometry derivation + overflow safety ---------------------------------
    {
        const PayloadGeometry hidden = current_layer_boundary_hidden_geometry();
        check(hidden.bytes_per_column == 10240, "hidden BF16 bytes/column must derive to 10240");
        const PayloadGeometry endpoint = current_endpoint_hidden_geometry();
        check(endpoint.bytes_per_column == 10240, "endpoint hidden bytes/column must also be 10240");
        const PayloadGeometry normalized = current_normalized_hidden_geometry();
        check(normalized.bytes_per_column == 10240, "normalized hidden bytes/column must also be 10240");
        const PayloadGeometry logits = current_logits_result_geometry();
        check(logits.bytes_per_column == 496640, "logits BF16 bytes/column must derive to 496640");

        check(hidden.total_bytes(1) == 10240, "total_bytes(1) must equal bytes_per_column");
        check(hidden.total_bytes(1000) == 10240ULL * 1000ULL, "total_bytes must scale exactly");
        // Large but NON-overflowing input: 10240 * 2^40 fits well within uint64_t (~1.1e16 << ~1.8e19
        // max), so this only proves 64-bit math is used, not that overflow is detected.
        const std::uint64_t huge_columns = 1ULL << 40;
        check(hidden.total_bytes(huge_columns) / huge_columns == hidden.bytes_per_column,
              "total_bytes must not overflow for a large NON-overflowing column count (64-bit math)");

        // sec 11: first genuinely-overflowing input must throw, not wrap.
        bool total_bytes_threw = false;
        try {
            const std::uint64_t overflowing_columns =
                std::numeric_limits<std::uint64_t>::max() / hidden.bytes_per_column + 1ULL;
            (void)hidden.total_bytes(overflowing_columns);
        } catch (const std::overflow_error&) { total_bytes_threw = true; }
        check(total_bytes_threw, "total_bytes must throw std::overflow_error on the first overflowing input");

        bool bf16_bytes_threw = false;
        try {
            (void)bf16_bytes_per_column(std::numeric_limits<std::uint64_t>::max());
        } catch (const std::overflow_error&) { bf16_bytes_threw = true; }
        check(bf16_bytes_threw,
              "bf16_bytes_per_column must throw std::overflow_error on the first overflowing input");

        // Current hidden/logit geometry must be unchanged by the overflow-checking change.
        check(hidden.columns_width == 5120, "hidden columns_width must remain 5120 (TextConfig::hidden)");
        check(logits.columns_width == 248320, "logits columns_width must remain the vocab width");
    }

    // ---- sec 17: record emitter refuses to fabricate Valid --------------------------------------
    {
        ObservationInput input;
        input.identity = CalibrationIdentity{}; // deliberately incomplete: default-constructed.
        input.kind = RecordKind::Operation;
        input.operation.family = OperationFamily::Mlp;
        input.operation.phase = ExecutionPhase::Prefill;
        input.operation.device_logical_slot = 0;
        input.run_id = "run-x";
        input.observation_id = "obs-x";
        input.value = 1.0;
        input.unit = "us";
        input.timestamp_unix_ms = 1000;
        input.evidence = EvidenceClass::Measured;
        input.requested_validity = ValidityState::Valid; // caller ASKS for Valid...
        input.correctness = CorrectnessStatus::Pass;

        const EmittedRecord emitted = emit_record(input);
        check(!emitted.missing_requirements.empty(),
              "an incomplete identity must produce non-empty missing_requirements");
        check(emitted.record.qualification.validity == ValidityState::Unqualified,
              "emitter must NOT honor requested_validity=Valid when requirements are missing");
        check(!emitted.is_qualified_evidence,
              "an incomplete record must never be qualified calibration evidence");

        // Now a complete identity + full operation metadata: requested Valid should be honored.
        const CalibrationIdentity identity = make_complete_identity();
        const CalibrationRecord complete = make_measured_record(identity, 5.0, 1000);
        check(complete.qualification.validity == ValidityState::Valid,
              "a fully-specified observation should be honored as Valid");
        check(is_qualified_calibration_evidence(complete),
              "a fully-specified Measured record should qualify as calibration evidence");
    }

    // ---- sec 19: transfer harness plumbing (compile/shape only, no live transfer) --------------
    {
        TransferIntent intent;
        intent.source_logical_slot = 0;
        intent.destination_logical_slot = 1;
        intent.payload_class = TransferPayloadClass::LayerBoundaryHidden;
        const TransferIdentity skeleton = transfer_identity_skeleton(intent);
        check(skeleton.payload_class == TransferPayloadClass::LayerBoundaryHidden,
              "transfer skeleton must carry the requested payload class");
        check(skeleton.direction == TransferDirection::DeviceToDevice,
              "transfer skeleton may derive DeviceToDevice from two distinct requested slots");
        check(skeleton.route == TransferRoute::Unknown,
              "transfer skeleton must never assume copy-engine vs kernel route");
        check(skeleton.overlap == TransferOverlap::Unknown,
              "transfer skeleton must never assume overlap");
    }

    // ---- sec 20: compute bench adapter shape/identity expansion ---------------------------------
    {
        BenchAdapterRequest request;
        request.family = BenchFamily::GdnDecode;
        request.shape.b = 1;
        request.shape.t = 1;
        request.device_logical_slot = 0;
        request.execution_mode = ExecutionMode::Graph;
        request.warm_state = WarmState::Warm;
        const BenchOperationExpansion expansion = expand_bench_request(request);
        check(expansion.is_gdn, "GdnDecode must expand to a GDN identity");
        check(expansion.gdn.stage == GdnStage::RecurrentDecode, "GdnDecode must map to RecurrentDecode stage");
        check(expansion.operation.family == OperationFamily::Gdn, "GdnDecode operation family must be Gdn");
        check(expansion.operation.phase == ExecutionPhase::Decode, "GdnDecode operation phase must be Decode");
        check(expansion.operation.execution_mode == ExecutionMode::Graph,
              "graph/eager must be populated explicitly from the request");
        check(expansion.operation.warm_state == WarmState::Warm,
              "warm/cold must be populated explicitly from the request");

        BenchAdapterRequest attn_request;
        attn_request.family = BenchFamily::FullAttentionPrefill;
        const BenchOperationExpansion attn_expansion = expand_bench_request(attn_request);
        check(!attn_expansion.is_gdn, "FullAttentionPrefill must NOT expand to a GDN identity");
        check(attn_expansion.operation.family == OperationFamily::Attention,
              "FullAttentionPrefill operation family must be Attention");
        check(attn_expansion.operation.phase == ExecutionPhase::Prefill,
              "FullAttentionPrefill operation phase must be Prefill");
    }

    if (failures == 0) {
        std::cout << "ninfer_calibration_collect_test: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}

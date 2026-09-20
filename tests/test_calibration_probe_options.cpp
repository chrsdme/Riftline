// Host-only test for ninfer_calibration_probe's argument parser (apps/calibration_probe/options.*).
// No CUDA, no artifact, no device -- exercises only the flag-parsing layer, same pattern as
// apps/cli's rope/options tests (tests/test_cli_options.cpp) compiling the options.cpp directly.

#include "options.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Builds an argv-shaped vector<char*> from string args (argv[0] is a fixed program name).
std::vector<char*> make_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& s : storage) { argv.push_back(s.data()); }
    return argv;
}

} // namespace

int main() {
    using namespace ninfer::calibration_probe;

    // Fully valid argv parses into the expected fields.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/some/src",
            "--artifact", "/some/artifact.ninfer",
            "--target-id", "target-a",
            "--weights-id", "weights-b",
            "--container-version", "v3",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.mode == ProbeMode::Identity, "valid argv: mode == Identity");
        check(options.source_root == "/some/src", "valid argv: source_root");
        check(options.artifact == "/some/artifact.ninfer", "valid argv: artifact");
        check(options.target_id == "target-a", "valid argv: target_id");
        check(options.weights_id == "weights-b", "valid argv: weights_id");
        check(options.container_version == "v3", "valid argv: container_version");
        check(options.slot0_cuda_index == 0, "valid argv: slot0_cuda_index");
        check(options.slot1_cuda_index == 1, "valid argv: slot1_cuda_index");
    }

    // transfer modes parse their own sample controls and default run ids.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "transfer-smoke",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
            "--warmup", "2",
            "--samples", "3",
            "--run-id", "rid",
            "--raw-dir", "/raw",
            "--summary-dir", "/summary",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.mode == ProbeMode::TransferSmoke, "transfer-smoke mode parsed");
        check(options.warmup == 2, "transfer warmup parsed");
        check(options.samples == 3, "transfer samples parsed");
        check(options.run_id == "rid", "transfer run id parsed");
        check(options.raw_dir == "/raw", "transfer raw dir parsed");
        check(options.summary_dir == "/summary", "transfer summary dir parsed");
    }

    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "transfer-route",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.mode == ProbeMode::TransferRoute, "transfer-route mode parsed");
        check(options.run_id.empty(), "transfer-route should not default a run id");
    }

    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "transfer-run",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.mode == ProbeMode::TransferRun, "transfer-run mode parsed");
        check(options.run_id == "m5_4c2_transfer", "transfer-run default run id");
    }

    // target_id/weights_id must come only from their own flags, never derived from --artifact.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/src",
            "--artifact", "/models/some-target-weights-name.ninfer",
            "--target-id", "explicit-target",
            "--weights-id", "explicit-weights",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.target_id == "explicit-target",
              "target_id taken from flag, not filename");
        check(options.weights_id == "explicit-weights",
              "weights_id taken from flag, not filename");
    }

    // container-version is optional.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.container_version.empty(), "container_version optional and empty");
    }

    // Missing required flag rejected.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "missing --source-root rejected");
    }

    // Unknown flag rejected.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "1",
            "--bogus-flag", "x",
        };
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "unknown flag rejected");
    }

    // Non-integer slot index rejected.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "not-a-number",
            "--slot1-cuda-index", "1",
        };
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "non-integer slot0-cuda-index rejected");
    }

    // Negative slot index rejected.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root", "/src",
            "--artifact", "/a.ninfer",
            "--target-id", "t",
            "--weights-id", "w",
            "--slot0-cuda-index", "0",
            "--slot1-cuda-index", "-1",
        };
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "negative slot1-cuda-index rejected");
    }

    // Unknown mode rejected.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "bogus-mode",
        };
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "unknown mode rejected");
    }

    // Missing mode entirely rejected.
    {
        std::vector<std::string> args = {"ninfer_calibration_probe"};
        std::vector<char*> argv = make_argv(args);
        bool threw = false;
        try {
            (void)parse_options(static_cast<int>(argv.size()), argv.data());
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "missing mode rejected");
    }

    // --name=value inline form also works.
    {
        std::vector<std::string> args = {
            "ninfer_calibration_probe", "identity",
            "--source-root=/src",
            "--artifact=/a.ninfer",
            "--target-id=t",
            "--weights-id=w",
            "--slot0-cuda-index=0",
            "--slot1-cuda-index=1",
        };
        std::vector<char*> argv = make_argv(args);
        Options options = parse_options(static_cast<int>(argv.size()), argv.data());
        check(options.source_root == "/src", "inline = form: source_root");
        check(options.slot1_cuda_index == 1, "inline = form: slot1_cuda_index");
    }

    if (failures == 0) {
        std::cout << "ninfer_calibration_probe_options_test: all checks passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}

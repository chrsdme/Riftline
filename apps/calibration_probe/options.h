#pragma once

// Argument parsing for ninfer_calibration_probe, split into its own translation unit (mirroring
// apps/cli/options.h) so a host-only test can exercise flag parsing without linking CUDA or the
// rest of main.cpp.

#include <string>

namespace ninfer::calibration_probe {

enum class ProbeMode { Identity, TransferRoute, TransferSmoke, TransferRun };

struct Options {
    ProbeMode mode = ProbeMode::Identity;

    std::string source_root;
    std::string artifact;
    std::string target_id;
    std::string weights_id;
    std::string container_version; // optional, may be empty.
    int slot0_cuda_index = -1;
    int slot1_cuda_index = -1;
    int warmup = -1;  // mode default when negative.
    int samples = -1; // mode default when negative.
    std::string run_id;
    std::string raw_dir;
    std::string summary_dir;
};

// Throws std::invalid_argument (with a message suitable for stderr) on any missing/unknown flag,
// unknown mode, or non-integer/negative CUDA index. target_id/weights_id are taken ONLY from
// --target-id/--weights-id -- never derived from --artifact's filename.
[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::calibration_probe

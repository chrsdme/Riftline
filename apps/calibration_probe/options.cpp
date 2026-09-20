#include "options.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace ninfer::calibration_probe {
namespace {

int parse_nonnegative_int(const std::string& text, std::string_view label) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<int>(value);
}

// Splits "--name=value" into (name, value); returns std::nullopt if arg has no '='.
std::optional<std::pair<std::string, std::string>> split_inline_value(const std::string& arg) {
    const std::size_t eq = arg.find('=');
    if (eq == std::string::npos) { return std::nullopt; }
    return std::make_pair(arg.substr(0, eq), arg.substr(eq + 1));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " identity|transfer-route|transfer-smoke|transfer-run --source-root <dir> --artifact <path> --target-id <id>\n"
           "       --weights-id <id> --slot0-cuda-index <n> --slot1-cuda-index <n>\n"
           "       [--container-version <ver>] [--warmup <n>] [--samples <n>]\n"
           "       [--run-id <id>] [--raw-dir <dir>] [--summary-dir <dir>]\n"
           "\n"
           "identity mode: collects live source/build/artifact/device/software identity via the\n"
           "M5.4b collectors and reports the identity gate. No timing benchmark, no writes.\n"
           "transfer-route mode: reports cudaDeviceCanAccessPeer without timing or writes.\n"
           "transfer modes: collect qualified transfer samples and summaries for current semantic\n"
           "payloads using the production UVA cudaMemcpyAsync DeviceToDevice route.\n";
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) { throw std::invalid_argument("missing mode"); }

    Options options;
    const std::string mode_arg = argv[1];
    if (mode_arg == "identity") {
        options.mode = ProbeMode::Identity;
    } else if (mode_arg == "transfer-route") {
        options.mode = ProbeMode::TransferRoute;
    } else if (mode_arg == "transfer-smoke") {
        options.mode = ProbeMode::TransferSmoke;
    } else if (mode_arg == "transfer-run") {
        options.mode = ProbeMode::TransferRun;
    } else {
        throw std::invalid_argument(
            "unknown mode: " + mode_arg +
            " (expected identity|transfer-route|transfer-smoke|transfer-run)");
    }

    bool have_source_root = false;
    bool have_artifact = false;
    bool have_target_id = false;
    bool have_weights_id = false;
    bool have_slot0 = false;
    bool have_slot1 = false;

    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        std::string name;
        std::string value;
        bool have_value = false;

        if (auto split = split_inline_value(arg)) {
            name = split->first;
            value = split->second;
            have_value = true;
        } else {
            name = arg;
        }

        auto next_value = [&]() -> std::string {
            if (have_value) { return value; }
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            ++i;
            return argv[i];
        };

        if (name == "--source-root") {
            options.source_root = next_value();
            have_source_root = true;
        } else if (name == "--artifact") {
            options.artifact = next_value();
            have_artifact = true;
        } else if (name == "--target-id") {
            options.target_id = next_value();
            have_target_id = true;
        } else if (name == "--weights-id") {
            options.weights_id = next_value();
            have_weights_id = true;
        } else if (name == "--container-version") {
            options.container_version = next_value();
        } else if (name == "--warmup") {
            options.warmup = parse_nonnegative_int(next_value(), "warmup");
        } else if (name == "--samples") {
            options.samples = parse_nonnegative_int(next_value(), "samples");
        } else if (name == "--run-id") {
            options.run_id = next_value();
        } else if (name == "--raw-dir") {
            options.raw_dir = next_value();
        } else if (name == "--summary-dir") {
            options.summary_dir = next_value();
        } else if (name == "--slot0-cuda-index") {
            options.slot0_cuda_index = parse_nonnegative_int(next_value(), "slot0-cuda-index");
            have_slot0 = true;
        } else if (name == "--slot1-cuda-index") {
            options.slot1_cuda_index = parse_nonnegative_int(next_value(), "slot1-cuda-index");
            have_slot1 = true;
        } else {
            throw std::invalid_argument("unknown flag: " + name);
        }
        ++i;
    }

    if (!have_source_root) { throw std::invalid_argument("missing required flag: --source-root"); }
    if (!have_artifact) { throw std::invalid_argument("missing required flag: --artifact"); }
    if (!have_target_id) { throw std::invalid_argument("missing required flag: --target-id"); }
    if (!have_weights_id) { throw std::invalid_argument("missing required flag: --weights-id"); }
    if (!have_slot0) { throw std::invalid_argument("missing required flag: --slot0-cuda-index"); }
    if (!have_slot1) { throw std::invalid_argument("missing required flag: --slot1-cuda-index"); }
    if (options.mode == ProbeMode::TransferSmoke || options.mode == ProbeMode::TransferRun) {
        if (options.samples == 0) { throw std::invalid_argument("--samples must be > 0"); }
        if (options.run_id.empty()) {
            options.run_id = options.mode == ProbeMode::TransferSmoke ? "m5_4c2_transfer_smoke"
                                                                       : "m5_4c2_transfer";
        }
    }

    return options;
}

} // namespace ninfer::calibration_probe

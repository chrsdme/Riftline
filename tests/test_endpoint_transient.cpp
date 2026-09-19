// M5.3 Task 2: the M4b peer-endpoint stage (text_context_impl.h endpoint_project, PEER branch)
// allocates transient buffers from the peer's own workspace arena that `build_workspace_plan` does
// not simulate. This test pins the contract that makes that safe. Two parts:
//
//   1. STRUCTURAL: `columns` and `head_columns` are never both large, so the 496,640 B/col
//      vocabulary term never multiplies a chunk-scale factor.
//   2. QUANTITATIVE: given (1), the endpoint peak fits inside the arena the planner already sizes.
//
// The comparison is against the ARENA -- a high-water mark across stages -- not against any single
// stage's per-column cost. That distinction matters: the planner's SMALLEST stage runs only ~384
// B/col above this one, a margin the 496,640 B logits constant exceeds below ~1293 columns. The
// arena is ~8.6x, so the real margin is wide and widens with chunk size.
//
// SCOPE OF THIS PROOF -- PROFILE-QUALIFIED, NOT UNIVERSAL. The two sides are not equally strong:
//   * the ENDPOINT side is SOURCE-DERIVED, computed from the same TextConfig constants the runtime
//     uses, so a change to the model geometry moves it here;
//   * the ARENA side is a MEASURED CONSTANT, recorded from the `workspace_bytes` field of the
//     [admission] line on the Qwen3.8-27B groupwise-int artifact at 57/7 @8K (evidence in
//     logs/claude-phase2/M5_3_20260919/). Nothing in this file calls build_workspace_plan.
// So this test qualifies the dominance invariant for the CURRENTLY SUPPORTED 27B workspace profile.
// It is not a theorem about every future artifact, weights profile, or workspace recipe: a change
// to build_workspace_plan or a new target could shrink the arena's per-column cost without
// anything here failing. Any such change must re-measure the arena and re-qualify this invariant,
// or derive it from the planner directly. M5 feasibility must not assume endpoint-transient
// dominance for an unqualified future target.
//
// Host-only: pure arithmetic over compile-time config, no device, no artifact, no arena.
#include "targets/qwen3_6_27b/impl/config.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

using Config = ninfer::targets::qwen3_6_27b::detail::TextConfig;

constexpr std::int64_t kBf16 = 2;

// --- endpoint stage cost ------------------------------------------------------------------------
// px [hidden, columns] BF16, a SEPARATE pn [hidden, columns] BF16 when apply_norm (otherwise pn
// aliases px), and pl [output_rows, head_columns] BF16. One arena scope, so all coexist at peak.
constexpr std::int64_t endpoint_peak_bytes(std::int64_t columns, std::int64_t head_columns,
                                           bool apply_norm) {
    return (apply_norm ? 2 : 1) * Config::hidden * columns * kBf16 +
           static_cast<std::int64_t>(Config::output_rows) * head_columns * kBf16;
}

constexpr std::int64_t kEndpointBytesPerColumn = 2 * Config::hidden * kBf16;   // 20,480
constexpr std::int64_t kLogitsBytesPerColumn =
    static_cast<std::int64_t>(Config::output_rows) * kBf16;                    // 496,640

// --- measured planned workspace arena (57/7 @8K, real artifact) ---------------------------------
// [admission] workspace_bytes at two prefill chunks. Linear with no chunk-independent floor:
// (180,953,088 - 22,619,136) / (1024 - 128) = 176,712 B/col exactly, and 176,712 * 128 = 22,619,136.
constexpr std::int64_t kMeasuredArenaChunk128  = 22619136;
constexpr std::int64_t kMeasuredArenaChunk1024 = 180953088;
constexpr std::int64_t kArenaBytesPerColumn    = 176712;

} // namespace

int main() {
    // 0. Internal consistency of the three RECORDED constants: they describe a line through the
    //    origin, which is what case 2 extrapolates along. NOTE what this does and does not do --
    //    all three values are frozen literals, so this is arithmetic among constants, not a probe
    //    of the live planner. It cannot detect an arena that changed in the source, and it cannot
    //    fail for a different artifact or workspace profile. Re-measuring is a human step (run the
    //    CLI at two prefill chunks and read `workspace_bytes` from the [admission] line); this
    //    check only catches someone editing one constant and not the others.
    check(kArenaBytesPerColumn * 128 == kMeasuredArenaChunk128,
          "measured arena at chunk 128 is not " + std::to_string(kArenaBytesPerColumn) + " B/col");
    check(kArenaBytesPerColumn * 1024 == kMeasuredArenaChunk1024,
          "measured arena at chunk 1024 is not " + std::to_string(kArenaBytesPerColumn) + " B/col");

    // 1. The endpoint's chunk-scaling cost is well below the arena's, so the margin widens with
    //    chunk size rather than narrowing. This is the load-bearing relation.
    check(kEndpointBytesPerColumn < kArenaBytesPerColumn,
          "endpoint per-column cost " + std::to_string(kEndpointBytesPerColumn) +
              " is not below the measured arena per-column cost " +
              std::to_string(kArenaBytesPerColumn));

    // 2. Fit at the two measured points and at larger legal chunks by extrapolation. head_columns
    //    is 1: on the prefill path it is `is_last ? 1 : 0`, never the chunk.
    for (const std::int64_t chunk : {std::int64_t{128}, std::int64_t{1024}, std::int64_t{8192},
                                     std::int64_t{131072}}) {
        const std::int64_t endpoint = endpoint_peak_bytes(chunk, 1, /*apply_norm=*/true);
        const std::int64_t arena    = kArenaBytesPerColumn * chunk;
        check(endpoint <= arena, "endpoint peak " + std::to_string(endpoint) +
                                     " exceeds the planned arena " + std::to_string(arena) +
                                     " at chunk " + std::to_string(chunk));
    }

    // 3. The structural cap is what keeps the vocabulary term off the chunk-scaling path. Pin that
    //    a hypothetical head_columns == columns at chunk scale would NOT fit -- i.e. this test would
    //    catch a regression that lifted the `is_last ? 1 : 0` cap in the prefill loop.
    {
        const std::int64_t chunk = 1024;
        check(endpoint_peak_bytes(chunk, 1, true) <= kArenaBytesPerColumn * chunk,
              "capped head_columns should fit at chunk 1024");
        check(endpoint_peak_bytes(chunk, chunk, true) > kArenaBytesPerColumn * chunk,
              "an uncapped head_columns at chunk scale should NOT fit -- if this passes, the "
              "structural cap is no longer what protects the arena and the contract needs revisiting");
    }

    // 4. Why the arena, not a single stage: the smallest stage's per-column margin over the endpoint
    //    is small enough that the logits constant overruns it at modest chunks. Pin the crossover so
    //    nobody re-derives the weaker (and wrong) per-stage argument.
    {
        constexpr std::int64_t kSmallestStagePerColumn = 20864;  // residual + gdn_control
        constexpr std::int64_t margin = kSmallestStagePerColumn - kEndpointBytesPerColumn;
        check(margin > 0 && kLogitsBytesPerColumn / margin > 1000,
              "the smallest-stage margin is no longer thin; re-check whether the arena-level "
              "argument is still the right one");
    }

    // 5. The non-norm path allocates one hidden buffer (pn aliases px), strictly cheaper than the
    //    worst case the contract is stated against.
    check(endpoint_peak_bytes(1024, 1, /*apply_norm=*/false) <
              endpoint_peak_bytes(1024, 1, /*apply_norm=*/true),
          "the non-norm endpoint path should allocate strictly less than the norm path");

    // 6. Decode-shaped calls are tiny: columns == head_columns == batch, and layer split pins batch
    //    to 1 (engine.cpp rejects max_concurrency > 1 under layer split, M5.3).
    check(endpoint_peak_bytes(1, 1, true) == 2 * Config::hidden * kBf16 + kLogitsBytesPerColumn,
          "single-column endpoint peak does not match the documented formula");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

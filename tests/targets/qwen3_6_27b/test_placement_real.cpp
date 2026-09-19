// M5.1 F / G: candidate weight residency on the REAL Qwen3.8-27B artifact, through the real
// materialization planner (Package::plan_load with an explicit PlacementSpec). Host-only: the
// binder never touches a device, so this needs the artifact but no GPU.
//
// The accepted M4a endpoint delta (1,350,871,040 bytes = output_head 1,350,860,800 + final_norm
// 10,240) is used ONLY as a test oracle; the planner never contains it.
//
// Gate: NINFER_QWEN3_8_27B_WEIGHTS (artifact path); returns 77 (skip) when unset.
#include "artifact/binder.h"
#include "artifact/reader.h"
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using ninfer::targets::PlacementSpec;
using ninfer::targets::qwen3_6_27b::Package;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

struct Weights {
    std::uint64_t slot0 = 0;
    std::uint64_t slot1 = 0;
};

Weights plan(const ninfer::artifact::Reader& reader, Package::WeightsProfile profile,
             const ninfer::EngineOptions& options, PlacementSpec spec) {
    ninfer::artifact::Binder binder(reader, 2);
    const Package::LoadPlan load_plan = Package::plan_load(binder, options, profile, spec);
    return Weights{load_plan.materialization().device_capacity_bytes[0],
                   load_plan.materialization().device_capacity_bytes[1]};
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    ninfer::artifact::Reader reader(artifact);
    const Package::WeightsProfile profile = Package::resolve_weights(reader.identity());

    ninfer::EngineOptions options;
    options.split_mode = ninfer::SplitMode::Layer;
    options.devices    = {0, 1};
    options.use_cuda_graph = false;

    constexpr std::uint64_t kAcceptedEndpointDelta = 1350871040ULL; // M4a oracle, not a formula

    // F. Same boundary, endpoint 0 vs 1: exactly the endpoint group moves, nothing duplicates.
    for (int boundary : {57, 58, 12, 45, 37}) {
        options.layer_split_boundary = boundary;
        const ninfer::EngineOptions resolved = Package::resolve_options(options);
        const Weights e0 = plan(reader, profile, resolved, PlacementSpec{boundary, 0});
        const Weights e1 = plan(reader, profile, resolved, PlacementSpec{boundary, 1});
        const std::string tag = "boundary " + std::to_string(boundary);
        check(e0.slot0 > e1.slot0 && e1.slot1 > e0.slot1, tag + ": endpoint bytes moved 0 -> 1");
        check(e0.slot0 - e1.slot0 == e1.slot1 - e0.slot1, tag + ": moved bytes are conserved");
        check(e0.slot0 + e0.slot1 == e1.slot0 + e1.slot1, tag + ": total residency unchanged");
        check(e0.slot0 - e1.slot0 == kAcceptedEndpointDelta,
              tag + ": delta " + std::to_string(e0.slot0 - e1.slot0) + " != accepted M4a delta");
        std::cout << "PLACEMENT_WEIGHTS boundary=" << boundary << " E0=" << e0.slot0 << "/" << e0.slot1
                  << " E1=" << e1.slot0 << "/" << e1.slot1 << "\n";
    }

    // Accepted M4a admission figures at 8K (weights are context-independent): 57/7 E0 and 58/6 E1.
    {
        options.layer_split_boundary = 57;
        const Weights w57 = plan(reader, profile, Package::resolve_options(options), PlacementSpec{57, 0});
        check(w57.slot0 == 15521870080ULL && w57.slot1 == 1571620608ULL, "57/7 E0 planned weights match accepted admission");
        options.layer_split_boundary = 58;
        const Weights w58 = plan(reader, profile, Package::resolve_options(options), PlacementSpec{58, 1});
        check(w58.slot0 == 14398348288ULL && w58.slot1 == 2695142400ULL, "58/6 E1 planned weights match accepted admission");
    }

    // G. Manual path == explicit path for the manual spec (whatever NINFER_ENDPOINT_DEVICE says).
    {
        options.layer_split_boundary = 57;
        const ninfer::EngineOptions resolved = Package::resolve_options(options);
        const PlacementSpec manual = Package::manual_placement(resolved);
        ninfer::artifact::Binder manual_binder(reader, 2);
        const Package::LoadPlan manual_plan = Package::plan_load(manual_binder, resolved, profile);
        const Weights explicit_plan = plan(reader, profile, resolved, manual);
        check(manual_plan.materialization().device_capacity_bytes[0] == explicit_plan.slot0 &&
                  manual_plan.materialization().device_capacity_bytes[1] == explicit_plan.slot1,
              "manual plan_load equals explicit plan_load(manual_placement)");
        check(manual.boundary == 57, "manual boundary");
    }

    // A spec that disagrees with the resolved boundary is rejected at the trust boundary.
    {
        options.layer_split_boundary = 57;
        bool rejected = false;
        try {
            (void)plan(reader, profile, Package::resolve_options(options), PlacementSpec{56, 0});
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "boundary/spec disagreement rejected");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

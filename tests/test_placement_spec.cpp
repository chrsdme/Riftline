// M5.0: host-only test of the ONE placement authority (config.h owner_for_layer/owner_for_object)
// and the manual-placement mapping. No artifact, no device.
#include "targets/placement.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

} // namespace

int main() {
    using ninfer::targets::PlacementSpec;
    using ninfer::targets::qwen3_6_27b::Package;
    using namespace ninfer::targets::qwen3_6_27b::detail;

    // A. Layers follow the boundary, for every boundary in the legal range.
    for (int boundary = 1; boundary < TextConfig::layers; ++boundary) {
        for (int endpoint = 0; endpoint <= 1; ++endpoint) {
            const PlacementSpec spec{.boundary = boundary, .endpoint_owner_slot = endpoint};
            for (int layer = 0; layer < TextConfig::layers; ++layer) {
                const int expected = layer < boundary ? 0 : 1;
                check(owner_for_layer(spec, static_cast<std::size_t>(layer)) == expected,
                      "owner_for_layer boundary=" + std::to_string(boundary));
                const std::string object = "text/layers/" + std::to_string(layer) + "/mlp/down";
                check(owner_for_object(spec, object) == expected,
                      "owner_for_object " + object + " boundary=" + std::to_string(boundary));
                // Must agree with the pre-M5 primitive the sequence planner sizes state from.
                check(owner_for_layer(spec, static_cast<std::size_t>(layer)) ==
                          layer_owner_for_layer(static_cast<std::size_t>(layer),
                                                static_cast<std::size_t>(boundary)),
                      "owner_for_layer drifted from layer_owner_for_layer");
            }
            // B. The output endpoint group follows the spec's endpoint slot, nothing else does.
            check(owner_for_object(spec, "text/final_norm") == endpoint, "final_norm owner");
            check(owner_for_object(spec, "text/output_head") == endpoint, "output_head owner");
            check(owner_for_object(spec, "text/token_embedding") == 0, "embedding stays on slot 0");
            check(owner_for_object(spec, "tokenizer/vocab") == 0, "frontend resource on slot 0");
            check(owner_for_object(spec, "mtp/final_norm") == 0, "mtp object on slot 0");
            check(owner_for_object(spec, "text/layers/") == 0, "layer prefix without index");
            check(owner_for_object(spec, "text/layers/x/norm") == 0, "non-numeric layer index");
            check(owner_for_object(spec, "text/layers/64/norm") == 0, "out-of-range layer index");
            check(owner_for_object(spec, "text/layers/7") == 0, "layer index without slash");
        }
    }

    // C. Endpoint 0 vs endpoint 1 at the same boundary differ ONLY on the endpoint group.
    {
        const PlacementSpec e0{.boundary = 58, .endpoint_owner_slot = 0};
        const PlacementSpec e1{.boundary = 58, .endpoint_owner_slot = 1};
        for (int layer = 0; layer < TextConfig::layers; ++layer) {
            const std::string object = "text/layers/" + std::to_string(layer) + "/input_norm";
            check(owner_for_object(e0, object) == owner_for_object(e1, object),
                  "endpoint choice moved a layer: " + object);
        }
        check(owner_for_object(e0, "text/output_head") == 0 &&
                  owner_for_object(e1, "text/output_head") == 1,
              "endpoint choice did not move the output head");
    }

    // D. Manual mapping: the resolved --layer-split boundary plus the env selection (unset here ->
    //    slot 0). Candidate evaluation never goes through this function.
    {
        ninfer::EngineOptions options;
        options.split_mode           = ninfer::SplitMode::Layer;
        options.layer_split_boundary = 57;
        const PlacementSpec manual   = Package::manual_placement(options);
        check(manual.boundary == 57, "manual boundary");
        const char* env = std::getenv("NINFER_ENDPOINT_DEVICE");
        const int expected_endpoint =
            (env != nullptr && env[0] == '1' && env[1] == '\0') ? 1 : 0;
        check(manual.endpoint_owner_slot == expected_endpoint, "manual endpoint follows env once");
        check(ninfer::targets::manual_endpoint_owner_slot() == expected_endpoint,
              "shared env parser disagrees with the manual placement");
        ninfer::EngineOptions tp_options;
        const PlacementSpec tp = Package::manual_placement(tp_options);
        check(tp.endpoint_owner_slot == 0, "non-layer-split placement keeps endpoint on slot 0");
        check(tp.boundary == static_cast<int>(kDefaultLayerSplitBoundary),
              "non-layer-split placement carries the default boundary");
    }

    // E. Boundary range rule is the single --layer-split rule.
    {
        check(resolve_layer_split_boundary(0) == static_cast<int>(kDefaultLayerSplitBoundary),
              "0 resolves to the default boundary");
        bool low = false, high = false;
        try { (void)resolve_layer_split_boundary(-1); } catch (const std::invalid_argument&) { low = true; }
        try { (void)resolve_layer_split_boundary(64); } catch (const std::invalid_argument&) { high = true; }
        check(low && high, "out-of-range boundaries are rejected");
        check(resolve_layer_split_boundary(1) == 1 && resolve_layer_split_boundary(63) == 63,
              "legal extremes accepted");
    }

    // F. M5.3: layer split + max_concurrency > 1 is rejected at the EngineOptions trust boundary.
    //    Under an asymmetric split the two owners' per-page-group KV strides differ, so the shared
    //    affine capacity curve cannot model Automatic growth; rejecting loudly beats silently
    //    handing back concurrency-1 capacity. No shipping surface can produce this combination (the
    //    CLI has no --max-concurrency, serve cannot select layer split), but EngineOptions is a
    //    public library surface, which is what that validation block exists to guard.
    //    The artifact path is deliberately nonexistent: validation must reject BEFORE any file is
    //    opened, so this stays host-only and needs no artifact or device.
    {
        ninfer::EngineOptions options;
        options.artifact_path  = "/nonexistent-m53-guard.ninfer";
        options.split_mode     = ninfer::SplitMode::Layer;
        options.devices        = {0, 1};
        options.use_cuda_graph = false;
        options.max_concurrency = 4;

        bool rejected_for_concurrency = false;
        try {
            ninfer::Engine engine(std::move(options));
        } catch (const std::invalid_argument& error) {
            rejected_for_concurrency =
                std::string(error.what()).find("max_concurrency") != std::string::npos;
        } catch (const std::exception&) {
            // Any other failure means validation did not reach the guard.
        }
        check(rejected_for_concurrency,
              "layer split with max_concurrency > 1 was not rejected at EngineOptions validation");

        // Control: the same options at concurrency 1 must get PAST that guard (it then fails later
        // on the missing artifact, which proves the guard itself did not fire).
        ninfer::EngineOptions allowed;
        allowed.artifact_path   = "/nonexistent-m53-guard.ninfer";
        allowed.split_mode      = ninfer::SplitMode::Layer;
        allowed.devices         = {0, 1};
        allowed.use_cuda_graph  = false;
        allowed.max_concurrency = 1;
        bool rejected_for_concurrency_at_one = false;
        try {
            ninfer::Engine engine(std::move(allowed));
        } catch (const std::exception& error) {
            rejected_for_concurrency_at_one =
                std::string(error.what()).find("max_concurrency") != std::string::npos;
        }
        check(!rejected_for_concurrency_at_one,
              "max_concurrency == 1 must not trip the layer-split concurrency guard");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

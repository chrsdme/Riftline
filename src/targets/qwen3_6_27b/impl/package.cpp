#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/config.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_27b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan, int tensor_parallel,
         bool layer_split_in, PlacementSpec placement_in)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)), tp(tensor_parallel),
          layer_split(layer_split_in), placement(placement_in) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
    int tp = 1;
    bool layer_split = false;
    PlacementSpec placement{.boundary = static_cast<int>(kDefaultLayerSplitBoundary)};
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6_27b {
namespace {

// General-task presets published with each exact model. Keep the registrations separate even
// while their values agree so an upstream model-specific change has one obvious owner.
constexpr ModelSamplingDefaults kQwen3_6Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

constexpr ModelSamplingDefaults kQwen3_8Defaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6Defaults; }
    if (model == qwen3_8_model_id) { return kQwen3_8Defaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

static_assert(Package::layer_count == detail::TextConfig::layers,
             "Package::layer_count (public diagnostics) has drifted from TextConfig::layers");

EngineOptions Package::resolve_options(const EngineOptions& options) {
    EngineOptions resolved = options;
    if (options.split_mode == SplitMode::Layer) {
        resolved.layer_split_boundary =
            detail::resolve_layer_split_boundary(options.layer_split_boundary);
    }
    return resolved;
}

Package::LayerSplitOwnership Package::layer_split_ownership(int boundary) {
    const auto boundary_sz = static_cast<std::size_t>(boundary);
    return LayerSplitOwnership{
        .owner0_full_attention = detail::TextConfig::owned_full_attention_layers(0, boundary_sz),
        .owner0_gdn            = detail::TextConfig::owned_gdn_layers(0, boundary_sz),
        .owner1_full_attention = detail::TextConfig::owned_full_attention_layers(1, boundary_sz),
        .owner1_gdn            = detail::TextConfig::owned_gdn_layers(1, boundary_sz),
    };
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen36GroupwiseInt;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::Qwen38GroupwiseInt;
    }
    if (identity.model_id == model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen36Nvfp4;
    }
    if (identity.model_id == qwen3_8_model_id && identity.weights_id == "nvfp4") {
        return WeightsProfile::Qwen38Nvfp4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

PlacementSpec Package::manual_placement(const EngineOptions& options) {
    const bool layer_split = options.split_mode == SplitMode::Layer;
    return PlacementSpec{
        .boundary = layer_split ? options.layer_split_boundary
                                : static_cast<int>(detail::kDefaultLayerSplitBoundary),
        .endpoint_owner_slot = layer_split ? detail::endpoint_owner_device() : 0,
    };
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return plan_load(binder, options, weights_profile, manual_placement(options));
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile, PlacementSpec placement) {
    const bool layer_split = options.split_mode == SplitMode::Layer;
    const int tensor_parallel = layer_split ? 1 : options.tp;
    const int expected_devices = layer_split ? 2 : options.tp;
    if (binder.device_count() != expected_devices) {
        throw std::invalid_argument("artifact binder device count does not match split mode");
    }
    // `options.layer_split_boundary` is ALREADY resolved (0 -> default, range-validated) by
    // Package::resolve_options, called once by the registry before plan_load and
    // make_sequence_planner both run -- this reads that canonical value, it does not re-resolve.
    // A direct-library caller that bypasses resolve_options (or resolves it wrong) and reaches
    // here with layer_split && boundary == 0 must fail loudly: 0 would make layer_owner_for_layer
    // place every layer on owner 1 (loader succeeds silently), while persistent_layout's mirror
    // guard throws on the SAME condition -- so a bypassed loader would materialize a model the
    // planner then refuses to size. Fail here too, at the same trust boundary.
    if (layer_split && options.layer_split_boundary == 0) {
        throw std::invalid_argument(
            "qwen3_6_27b: layer_split_boundary must be resolved (nonzero) before plan_load -- "
            "call Package::resolve_options first");
    }
    // The spec is the ONE ownership input (M5.0). It must agree with the boundary the sequence
    // planner sizes state from, or the loader would place layers the runtime then mis-sizes.
    if (layer_split && placement.boundary != options.layer_split_boundary) {
        throw std::invalid_argument(
            "qwen3_6_27b: placement boundary disagrees with options.layer_split_boundary");
    }
    if (placement.endpoint_owner_slot != 0 && placement.endpoint_owner_slot != 1) {
        throw std::invalid_argument("qwen3_6_27b: placement endpoint_owner_slot must be 0 or 1");
    }
    if (!layer_split) {
        placement = PlacementSpec{.boundary = static_cast<int>(detail::kDefaultLayerSplitBoundary)};
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(options),
                              tensor_parallel, layer_split, placement),
        tensor_parallel, layer_split, placement));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(plan.impl_->weights_profile,
                                                   std::move(plan.impl_->plan.bindings),
                                                   std::move(materialized), plan.impl_->tp,
                                                   plan.impl_->layer_split,
                                                   plan.impl_->placement);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan,
                        ExecutionContext& execution) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    const detail::LoadedModelData& data = model.impl_->data;
    if (!data.layer_split && data.tp != execution.tp) {
        throw std::invalid_argument(
            "loaded model shard width does not match the execution context");
    }
    const detail::RuntimeModelView* peer =
        data.runtime_peer.has_value() ? &*data.runtime_peer : nullptr;
    return qwen3_6::create_program<detail::Variant>(data.runtime, peer,
                                                   model.impl_->weights_profile, std::move(plan),
                                                   execution);
}

} // namespace ninfer::targets::qwen3_6_27b

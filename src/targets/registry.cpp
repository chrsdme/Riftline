#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/device_admission.h"
#include "targets/placement.h"
#include "targets/placement_planner.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

// Save/restore the current device around a per-device query.
class ScopedDevice {
public:
    explicit ScopedDevice(int device) {
        CUDA_CHECK(cudaGetDevice(&previous_));
        CUDA_CHECK(cudaSetDevice(device));
    }

    ~ScopedDevice() { (void)cudaSetDevice(previous_); }

    ScopedDevice(const ScopedDevice&)            = delete;
    ScopedDevice& operator=(const ScopedDevice&) = delete;

private:
    int previous_ = 0;
};

// `split_context` names the split this admission check is running under (e.g. "layer split
// (54/10)"), or is empty for TP/single-device. Runs BEFORE artifact::materialize(), so a shortfall
// is a clean pre-materialization rejection, not a crash mid-load; per-device weight bytes already
// reflect whatever boundary the shard resolver used (see qwen3_6_27b::detail::layer_placement), so
// this check is split-boundary-aware with no new admission mechanism.
std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes, int device,
                                                std::string_view split_context = {}) {
    const ScopedDevice scope(device);
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        std::string message = "model weights require an ESTIMATED " + std::to_string(weight_bytes) +
                              " bytes of device memory on device " + std::to_string(device) +
                              ", but only " + std::to_string(free_bytes) +
                              " bytes are free before loading weights (pre-materialization check)";
        if (!split_context.empty()) {
            message += "; requested split: " + std::string(split_context);
        }
        throw std::invalid_argument(message);
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

// M5.2 diagnostic: NINFER_PLACEMENT_REPORT=1 under --split-mode layer prints the analytical
// feasibility of EVERY legal boundary x endpoint-owner candidate against a capacity snapshot taken
// right here (pre-materialization), then returns so the manual load proceeds exactly as requested.
// Report-only: nothing downstream reads it, no candidate is applied, no ranking exists. Fails
// closed as a diagnostic -- any exception is printed and swallowed, never propagated into the
// load. Only targets exposing the explicit-placement plan_load (qwen3_6_27b) participate.
template <class Target>
void maybe_report_placements(const EngineOptions& options, ExecutionContext& execution,
                             const artifact::Reader& reader,
                             typename Target::WeightsProfile weights_profile) {
    if constexpr (requires(artifact::Binder& binder) {
                      Target::plan_load(binder, options, weights_profile, PlacementSpec{});
                  }) {
        const char* env = std::getenv("NINFER_PLACEMENT_REPORT");
        if (env == nullptr || std::string_view(env) != "1") { return; }
        if (options.split_mode != SplitMode::Layer || execution.tp != 2) { return; }
        try {
            PlacementReport report;
            report.max_context     = options.max_context;
            report.kv_policy       = options.kv_capacity;
            report.max_concurrency = options.max_concurrency;
            report.kv_dtype = options.kv_cache == KvCacheStorage::BFloat16 ? "bf16" : "int8";
            report.layer_count            = Target::layer_count;
            report.allocator_reserve_bytes = effective_allocator_reserve_bytes();
            for (int slot = 0; slot < kPlacementSlots; ++slot) {
                const int device = execution.dev[static_cast<std::size_t>(slot)]->device;
                report.devices[static_cast<std::size_t>(slot)]                = device;
                report.capacity_snapshot_bytes[static_cast<std::size_t>(slot)] =
                    current_free_device_bytes(device);
            }
            PlacementEnumerationSource source;
            source.layer_count = Target::layer_count;
            // The SAME legality rule manual --layer-split uses: resolve_options throws out of range.
            source.boundary_is_legal = [&options](int boundary) {
                EngineOptions candidate          = options;
                candidate.layer_split_boundary   = boundary;
                (void)Target::resolve_options(candidate);
                return true;
            };
            source.layer_counts = [](int boundary) {
                const typename Target::LayerSplitOwnership ownership =
                    Target::layer_split_ownership(boundary);
                return PlacementLayerCounts{.owner0_full_attention = ownership.owner0_full_attention,
                                            .owner0_gdn            = ownership.owner0_gdn,
                                            .owner1_full_attention = ownership.owner1_full_attention,
                                            .owner1_gdn            = ownership.owner1_gdn};
            };
            source.make_inputs = [&](PlacementSpec spec) {
                EngineOptions candidate        = options;
                candidate.layer_split_boundary = spec.boundary;
                candidate                      = Target::resolve_options(candidate);
                // Weights: the real materialization planner, host-only, for THIS spec.
                artifact::Binder binder(reader, 2);
                const auto load_plan = Target::plan_load(binder, candidate, weights_profile, spec);
                // Layouts: the real sequence planner for this boundary (pure planning; the device
                // is used only for capability validation). Shared by the preview callable.
                auto planner = std::make_shared<typename Target::SequencePlanner>(
                    Target::make_sequence_planner(execution.primary(), candidate, weights_profile));
                PlacementFeasibilityInputs inputs;
                inputs.spec = spec;
                for (int slot = 0; slot < kPlacementSlots; ++slot) {
                    const auto index = static_cast<std::size_t>(slot);
                    inputs.devices[index] = PlacementDeviceInputs{
                        .device                  = report.devices[index],
                        .planned_weight_bytes    = load_plan.materialization().device_capacity_bytes[index],
                        .capacity_snapshot_bytes = report.capacity_snapshot_bytes[index],
                    };
                }
                inputs.allocator_reserve_bytes = report.allocator_reserve_bytes;
                inputs.kv_policy               = options.kv_capacity;
                inputs.curve                   = planner->capacity_curve();
                inputs.preview = [planner](int slot, std::optional<std::uint32_t> groups) {
                    return planner->layout_preview(slot, groups);
                };
                return inputs;
            };
            enumerate_placements(source, report);
            std::cerr << render_placement_report(report);
        } catch (const std::exception& error) {
            std::cerr << "[placement] report failed (manual load continues): " << error.what()
                      << "\n";
        }
    }
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options_in, ExecutionContext& execution,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    // Resolved ONCE here (e.g. layer_split_boundary: 0 -> default, otherwise range-validated
    // against this target's layer count) so plan_load and make_sequence_planner below -- and the
    // Loaded* / diagnostics that read options afterward -- all see the identical canonical value.
    const EngineOptions options = Target::resolve_options(options_in);
    // M5.0: the manual placement, resolved ONCE (boundary from options, endpoint from the cached
    // NINFER_ENDPOINT_DEVICE read). Only reported here; the 27B loader resolves the identical spec
    // inside Target::plan_load.
    const PlacementSpec manual_placement{
        .boundary            = options.layer_split_boundary,
        .endpoint_owner_slot = manual_endpoint_owner_slot(),
    };
    const auto& identity                          = reader.identity();
    if (execution.phase1_heterogeneous() &&
        (identity.model_id != Qwen3_6_27B::qwen3_8_model_id ||
         identity.weights_id != "groupwise-int")) {
        throw std::invalid_argument(
            "Phase-1 heterogeneous TP requires the qwen3.8-27b/groupwise-int profile");
    }
    const auto weights_profile                    = Target::resolve_weights(identity);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);
    maybe_report_placements<Target>(options, execution, reader, weights_profile);
    const int tp                                  = execution.tp;
    DeviceContext& device                         = execution.primary();

    artifact::Binder binder(reader, tp);
    auto load_plan = Target::plan_load(binder, options, weights_profile);
    // The capacity curve describes ONE device. At tp 2 it is already the per-device curve (halved
    // KV heads, halved GDN state) because the sequence plan is built with `tp`, so the same curve
    // serves both devices and the resolver only has to pick the bottleneck device's budget.
    // Under `--split-mode layer` the devices are NOT symmetric -- each owns a disjoint layer range,
    // so owner 1's persistent footprint differs from owner 0's (12/52 owner 1 owns 52 of 64
    // layers). The curve stays single and shared, but its base is now the max over owning slots
    // (M2.2, layouts_impl.h device_reservation_bytes), so "the bottleneck device's budget" is
    // charged the worst owner's requirement rather than owner 0's. That single shared curve is only
    // legal while both owners share a per-page-group stride, which M5.3 showed is NOT automatic
    // (the stride scales with each owner's full-attention layer count). make_sequence_planner below
    // enforces it at construction and throws otherwise, so the curve read here always satisfies it.
    auto sequence_planner = Target::make_sequence_planner(device, options, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::string split_context =
        options.split_mode == SplitMode::Layer
            ? "layer split (" + std::to_string(options.layer_split_boundary) + "/" +
                  std::to_string(Target::layer_count - options.layer_split_boundary) + ")"
            : std::string();
    std::vector<std::size_t> preflight_runtime_bytes;
    preflight_runtime_bytes.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        preflight_runtime_bytes.push_back(runtime_bytes_after_planned_weights(
            load_plan.materialization().device_capacity_bytes[static_cast<std::size_t>(rank)],
            execution.dev[static_cast<std::size_t>(rank)]->device, split_context));
    }
    // M2.2b: each slot's OWN minimum-page-group reservation (persistent + workspace + request
    // transient + graph allowance -- the identical composition layouts_impl.h sums into
    // `device_reservation_bytes` before taking the max; `layout_preview(slot)` reads the very
    // same minimum-point candidate, so this is not a second accounting path). With these the
    // Automatic resolver charges the exact per-device residual `min_i(F_i - B_i)` instead of
    // the conservative `min(F) - max(B)`; Explicit mode is unchanged. If any slot's preview is
    // unavailable the span is left EMPTY, which the resolver treats as "fall back to the
    // conservative form" -- never as a zero base.
    std::vector<std::size_t> minimum_reservation_bases;
    minimum_reservation_bases.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        const targets::qwen3_6::DeviceLayoutPreview preview = sequence_planner.layout_preview(rank);
        if (!preview.available) {
            minimum_reservation_bases.clear();
            break;
        }
        minimum_reservation_bases.push_back(
            preview.persistent_bytes + preview.workspace_capacity_bytes +
            preview.request_transient_capacity_bytes + preview.graph_allowance_bytes);
    }
    (void)runtime::resolve_kv_capacity_symmetric(options.kv_capacity, curve,
                                                 preflight_runtime_bytes, minimum_reservation_bases);

    // Exact context-aware per-device admission (M2). Runs BEFORE artifact::materialize() below, so
    // a shortfall on either device is a clean pre-materialization rejection, never a mid-load
    // cudaMalloc crash. Guarded on layer split: under layer split owner-1's exact persistent
    // layout can differ from owner-0's (see layouts_impl.h persistent/persistent_peer), so the
    // single-scalar preflight above under-states one device pre-M2. TP2 and single-device stay
    // symmetric by construction and are unaffected -- this block only REPORTS there (never
    // rejects), so admitted/rejected outcomes for those modes are byte-identical to before M2.
    // Shared by both admission phases: renders a DeviceAdmissionPlan rejection into the same
    // exception message shape (reason_code / reason_detail / required-vs-capacity / split context).
    const auto admission_reject_message = [&split_context](int dev_ordinal, int rank,
                                                            const DeviceAdmissionPlan& plan) {
        std::string message = "device " + std::to_string(dev_ordinal) + " (slot " +
                              std::to_string(rank) + ") fails exact per-device admission: " +
                              plan.reason_code + " -- " + plan.reason_detail +
                              " (required " + std::to_string(plan.total_required_bytes) +
                              " bytes, effective capacity " +
                              std::to_string(plan.effective_capacity) + " bytes)";
        if (!split_context.empty()) { message += "; requested split: " + split_context; }
        return message;
    };

    for (int rank = 0; rank < tp; ++rank) {
        const int dev_ordinal = execution.dev[static_cast<std::size_t>(rank)]->device;
        const targets::qwen3_6::DeviceLayoutPreview preview = sequence_planner.layout_preview(rank);
        if (!preview.available) { continue; }
        DeviceAdmissionPlan plan = build_device_admission_plan(
            dev_ordinal, rank,
            load_plan.materialization().device_capacity_bytes[static_cast<std::size_t>(rank)],
            preview);
        plan.endpoint_owner_slot = manual_placement.endpoint_owner_slot;
        print_device_admission_summary(plan, split_context, "A");
        if (options.split_mode == SplitMode::Layer && plan.decision == DeviceAdmissionDecision::Reject) {
            throw std::invalid_argument(admission_reject_message(dev_ordinal, rank, plan));
        }
    }

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), execution,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    std::vector<std::size_t> free_before_runtime;
    free_before_runtime.reserve(static_cast<std::size_t>(tp));
    for (int rank = 0; rank < tp; ++rank) {
        free_before_runtime.push_back(
            current_free_device_bytes(execution.dev[static_cast<std::size_t>(rank)]->device));
    }
    runtime::KvCapacityResolution capacity_resolution = runtime::resolve_kv_capacity_symmetric(
        options.kv_capacity, curve, free_before_runtime, minimum_reservation_bases);

    // Exact post-materialize per-device admission (M2.1), re-run at the RESOLVED page-group count
    // -- the point `finalize()` below actually allocates at, rather than the minimum-page-group
    // point phase A checked. A no-op when resolved == curve.minimum_main_page_groups (the CLI
    // default: at --max-concurrency 1 the curve's minimum and maximum coincide, so the Automatic
    // growth arm in kv_capacity.cpp is unreachable). It bites when resolved exceeds the minimum:
    // `auto` growing under --max-concurrency > 1, and Explicit capacities above the minimum (which
    // are validated >=, not ==) -- for those, this is a new pre-finalize rejection gate.
    // Capacity basis is `free_before_runtime[rank]`, the same measured post-weight vector the
    // resolver was sized against, so approved and allocated are the identical number;
    // planned_weight_bytes is 0 because those weights are already resident inside that figure.
    // Guarded identically to phase A: only SplitMode::Layer rejects, TP2/single-device report only.
    for (int rank = 0; rank < tp; ++rank) {
        const int dev_ordinal = execution.dev[static_cast<std::size_t>(rank)]->device;
        const targets::qwen3_6::DeviceLayoutPreview resolved_preview =
            sequence_planner.layout_preview(rank, capacity_resolution.main_page_groups);
        if (!resolved_preview.available) { continue; }
        DeviceAdmissionPlan plan = build_device_admission_plan(
            dev_ordinal, rank, /*planned_weight_bytes=*/0, resolved_preview,
            /*capacity_override=*/free_before_runtime[static_cast<std::size_t>(rank)]);
        plan.kv_mode = options.kv_capacity.mode == KvCapacityMode::Automatic ? "auto" : "explicit";
        plan.resolved_tokens      = capacity_resolution.resolved_tokens;
        plan.resolved_page_groups = capacity_resolution.main_page_groups;
        plan.endpoint_owner_slot  = manual_placement.endpoint_owner_slot;
        print_device_admission_summary(plan, split_context, "B");
        if (options.split_mode == SplitMode::Layer && plan.decision == DeviceAdmissionDecision::Reject) {
            throw std::invalid_argument(admission_reject_message(dev_ordinal, rank, plan));
        }
    }

    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    auto loaded   = std::make_unique<Loaded>(std::move(model), options);
    auto instance = std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                               std::move(sequence_plan), execution);
    for (int rank = 0; rank < tp; ++rank) {
        execution.dev[static_cast<std::size_t>(rank)]->synchronize();
    }
    instance->kv_capacity_resolution.available_after_startup_bytes =
        current_free_device_bytes(device.device);

    LoadSummary summary;
    // Rotary regime, as the target runtime resolved it. `effective_max_context` and `yarn_mscale`
    // come from the constructed program (only it knows the variant's native capacity and the YaRN
    // table it built); the mode/factor/origin echo what the caller asked for.
    summary.rope_mode  = options.rope_mode;
    summary.yarn_factor  = options.rope_mode == RopeMode::Yarn ? options.yarn_factor : 0.0;
    summary.yarn_origin  = options.rope_mode == RopeMode::Yarn ? options.yarn_origin : 0U;
    // Per-device memory table. `weights_bytes` and the free/total figures are MEASURED per device;
    // the sequence, KV, GDN and workspace figures come from the finalized plan, which is symmetric
    // across devices by construction (identical page counts, identical halved head geometry).
    const MemorySummary memory = instance->program->memory_summary();
    summary.effective_max_context = memory.effective_max_context;
    summary.yarn_mscale           = memory.yarn_mscale;
    summary.tp                 = tp;
    if (options.split_mode == SplitMode::Layer) {
        summary.layer_split_boundary = options.layer_split_boundary;
        summary.layer_count          = Target::layer_count;
        const typename Target::LayerSplitOwnership ownership =
            Target::layer_split_ownership(options.layer_split_boundary);
        summary.owner0_full_attention_layers = ownership.owner0_full_attention;
        summary.owner0_gdn_layers            = ownership.owner0_gdn;
        summary.owner1_full_attention_layers = ownership.owner1_full_attention;
        summary.owner1_gdn_layers            = ownership.owner1_gdn;
    }
    for (int rank = 0; rank < tp; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        DeviceMemoryReport& row = summary.devices[slot];
        row.device              = execution.dev[slot]->device;
        row.owner_slot          = rank;
        row.weights_bytes       = stats.per_device_capacity_bytes[slot];
        // Under layer split, rank 1's decoder state is PeerRuntime's own (owner-local) allocation,
        // not rank 0's -- see MemorySummary::*_peer. TP2/single-device keeps reading rank 0's
        // figures for both rows, matching today's symmetric-by-construction behavior.
        if (options.split_mode == SplitMode::Layer && rank == 1) {
            row.kv_pool_bytes   = memory.kv_payload_bytes_peer;
            row.gdn_state_bytes = memory.gdn_state_bytes_peer;
            row.sequence_bytes  = memory.sequence_peer.capacity_bytes;
            row.full_attention_layers = summary.owner1_full_attention_layers;
            row.gdn_layers             = summary.owner1_gdn_layers;
        } else {
            row.kv_pool_bytes   = memory.kv_payload_bytes;
            row.gdn_state_bytes = memory.gdn_state_bytes;
            row.sequence_bytes  = memory.sequence.capacity_bytes;
            row.full_attention_layers =
                options.split_mode == SplitMode::Layer ? summary.owner0_full_attention_layers : 0;
            row.gdn_layers =
                options.split_mode == SplitMode::Layer ? summary.owner0_gdn_layers : 0;
        }
        // Planned workspace is the same on both devices by construction (build_workspace_plan is a
        // per-layer-execution high-water mark: no layer-count, ownership or architecture input), so
        // both rows read rank 0's capacity. The MEASURED peak is per device (M3) -- under layer
        // split rank 1 reads its own PeerRuntime arena, which pre-M3 was never surfaced anywhere.
        row.workspace_bytes     = memory.workspace.capacity_bytes;
        row.workspace_peak_bytes = options.split_mode == SplitMode::Layer && rank == 1
                                       ? memory.workspace_peer.peak_used_bytes
                                       : memory.workspace.peak_used_bytes;
        // Measured per device: instantiating the decode graphs materializes driver state on each
        // device the graph has nodes on, which at tp 2 is both of them.
        row.cuda_graph_bytes = rank == 0 ? memory.cuda_graph_observed_bytes
                                         : memory.cuda_graph_peer_observed_bytes;
        row.reserved_bytes = row.weights_bytes + capacity_resolution.runtime_reservation_bytes;
        row.free_after_startup_bytes = current_free_device_bytes(row.device);
        row.total_bytes              = current_total_device_bytes(row.device);
    }
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = ActiveTarget(std::move(instance)),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model,
                                     const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model, options)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), execution)) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

LoadedQwen3_6_35BA3B::LoadedQwen3_6_35BA3B(
    std::unique_ptr<Qwen3_6_35BA3B::LoadedModel> stable_model, const EngineOptions& options)
    : model(std::move(stable_model)), frontend(Qwen3_6_35BA3B::make_frontend(*model, options)) {}

LoadedQwen3_6_35BA3B::~LoadedQwen3_6_35BA3B() = default;

Qwen3_6_35BA3BInstance::Qwen3_6_35BA3BInstance(std::unique_ptr<LoadedQwen3_6_35BA3B> stable_loaded,
                                               runtime::KvCapacityResolution resolution,
                                               Qwen3_6_35BA3B::SequencePlan sequence_plan,
                                               ExecutionContext& execution)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(execution.primary(), sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_35BA3B::create_program(*loaded->model, std::move(sequence_plan),
                                             execution)) {}

Qwen3_6_35BA3BInstance::~Qwen3_6_35BA3BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, ExecutionContext& execution) {
    validate_options(options);
    const int expected_devices = options.split_mode == SplitMode::Layer ? 2 : options.tp;
    if (execution.tp != expected_devices) {
        throw std::invalid_argument("execution context width does not match EngineOptions.tp");
    }
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, execution, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
    }
    if (identity.model_id == Qwen3_6_35BA3B::model_id) {
        return construct_registered<Qwen3_6_35BA3B, LoadedQwen3_6_35BA3B, Qwen3_6_35BA3BInstance>(
            options, execution, reader, load_start, Qwen3_6_35BA3B::target_key);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets

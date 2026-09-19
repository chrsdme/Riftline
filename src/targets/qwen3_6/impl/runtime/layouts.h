#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/cyclic_kv_cache.h"
#include "core/dtype.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/startup_features.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using TensorLayout = TensorRegion;

struct DFlashPersistentLayout {
    CyclicKVCacheLayout local;
    CyclicKVCacheLayout rewrite_checkpoint_local;
    qwen3_6::PagedKVCacheLayout full;
    TensorLayout prefill_features;
    TensorLayout prefill_positions;
    TensorLayout pending_features;

    [[nodiscard]] std::size_t kv_payload_bytes() const noexcept {
        return local.payload_bytes() + rewrite_checkpoint_local.payload_bytes() +
               full.payload_bytes();
    }
};

struct PersistentLayout {
    qwen3_6::DecoderStateLayout decoder;
    std::optional<GdnReplayRecordLayout> replay_records;
    std::optional<DFlashPersistentLayout> dflash;
    qwen3_6::RoundStateLayout round;
    TensorLayout prefill_hidden;
    TensorLayout token_counts;
    TensorLayout sampling_config;
    TensorLayout tail_hidden;
    TensorLayout rewrite_checkpoint_hidden;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
    // Device SLOT (0 or 1, matching LoadedModelData::device_slot / layer_owner[] -- NOT a CUDA
    // ordinal) this layout is sized for under `--split-mode layer`. -1 outside layer split, where
    // one shared layout still serves every device.
    int owner_slot = -1;
};

struct WorkspacePlan {
    std::size_t text_prefill   = 0;
    std::size_t ordinary_round = 0;
    std::size_t mtp_prefill    = 0;
    std::size_t mtp_round      = 0;
    std::size_t dflash_context = 0;
    std::size_t dflash_round   = 0;
    std::size_t vision_encode  = 0;
    std::size_t capacity       = 0;
};

struct SequencePlanningInputs {
    WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    // Rotary regime. Nothing in the persistent or workspace layout depends on these:
    // the YaRN table is a 32-float per-device buffer the Program owns outside the planned arenas,
    // so a native plan is byte-identical to the pre-YaRN one. They are carried here only so the
    // Program can build and upload that table without re-reading EngineOptions.
    RopeMode rope_mode                  = RopeMode::Native;
    double yarn_factor                  = 0.0;
    std::uint32_t yarn_origin           = 0;
    // The ceiling `capacity` was admitted against: the variant's native capacity under Native,
    // `yarn_origin * yarn_factor` under Yarn.
    std::uint32_t effective_max_context = 0;
    bool use_cuda_graph = true;
    int device          = 0;
    // Tensor-parallel width. Every per-device geometry below (KV heads, GDN value heads, GDN conv
    // channels) is the model's own extent divided by `tp`, because each device holds only its own
    // head shard. Page COUNTS are not divided: all devices carry the same pages.
    int tp = 1;
    // True under `--split-mode layer` (EngineOptions::SplitMode::Layer). Unlike `tp`, this is not
    // a head/channel split -- each device executes a disjoint contiguous prefix/suffix of the 64
    // text layers (see qwen3_6_27b::detail::kDefaultLayerSplitBoundary), so persistent_layout()
    // sizes each device's KV/GDN decoder state to THAT device's own owned layer counts (see
    // PersistentLayout::owner_slot / SequencePlanImpl::persistent_peer) instead of the full
    // model's, under this flag only. False (TP or single-device) keeps today's full-size state,
    // byte-identical.
    bool layer_split = false;
    // The resolved layer-split boundary (EngineOptions::layer_split_boundary, defaulted/validated
    // once by qwen3_6_27b::detail::resolve_layer_split_boundary). Meaningless when layer_split is
    // false. Must be the SAME canonical value the loader used to populate layer_owner[], so
    // persistent_layout()'s owner-local counts agree with the layers actually materialized on each
    // device.
    std::size_t layer_split_boundary = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct SequencePlanImpl<NINFER_QWEN36_VARIANT> {
    typename NINFER_QWEN36_VARIANT::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t kv_capacity              = 0;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t max_concurrency          = 1;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    // See SequencePlanningInputs: rope regime carried through the plan, layout-neutral.
    RopeMode rope_mode                  = RopeMode::Native;
    double yarn_factor                  = 0.0;
    std::uint32_t yarn_origin           = 0;
    std::uint32_t effective_max_context = 0;
    bool use_cuda_graph = true;
    int device          = 0;
    int tp              = 1;
    bool layer_split    = false;
    std::size_t layer_split_boundary = 0;
    NINFER_QWEN36_RUNTIME_NS::PersistentLayout persistent;
    // Owner 1's own persistent layout under `--split-mode layer` (owner_slot == 1). Empty outside
    // layer split, where `persistent` above already serves both devices.
    std::optional<NINFER_QWEN36_RUNTIME_NS::PersistentLayout> persistent_peer;

    // The persistent layout sized for device SLOT `slot` (0 or 1, matching device_slot/
    // layer_owner[]). Outside layer split -- or for slot 0, which is always the prefix owner --
    // this is the same shared `persistent` layout every device has always read.
    [[nodiscard]] const NINFER_QWEN36_RUNTIME_NS::PersistentLayout&
    persistent_for(int slot) const noexcept {
        return slot == 1 && persistent_peer ? *persistent_peer : persistent;
    }
    NINFER_QWEN36_RUNTIME_NS::WorkspacePlan workspace;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_allowance_bytes            = 0;
    std::size_t device_reservation_bytes         = 0;
};

template <>
struct SequencePlannerImpl<NINFER_QWEN36_VARIANT> {
    NINFER_QWEN36_RUNTIME_NS::SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl<NINFER_QWEN36_VARIANT>> minimum;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using SequencePlanImpl = qwen3_6::detail::SequencePlanImpl<Variant>;

[[nodiscard]] std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

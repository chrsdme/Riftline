#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace ninfer {
struct DeviceContext;
struct ExecutionContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

// Per-device-SLOT preview of the sequence plan. With no `main_page_groups` argument (or when it
// equals the curve's minimum), this is the plan's MINIMUM page-group point (the same point
// `SequencePlannerImpl::minimum` is built at -- see layouts_impl.h): a pre-materialization,
// conservative lower bound on what the finalized plan will reserve, since page count only grows
// the sequence/workspace footprint on top of this. Passing an explicit `main_page_groups` (e.g.
// the RESOLVED `--kv-capacity auto` capacity, post-materialize) instead builds the exact preview
// at that page count via the SAME `build_sequence_candidate` path `finalize()` uses -- no second
// byte-accounting path. That call is NOT noexcept-safe internally (`build_sequence_candidate` can
// throw on overflow/inconsistency); `available` is false and the throw is swallowed on failure so
// this method can stay noexcept end-to-end.
struct DeviceLayoutPreview {
    bool available                              = false;
    std::size_t persistent_bytes                = 0;
    std::size_t persistent_kv_payload_bytes      = 0;
    std::size_t workspace_capacity_bytes         = 0;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_allowance_bytes            = 0;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    // `slot` is 0 or 1, matching device_slot/layer_owner[] (NOT a CUDA ordinal). Slot 1 only
    // differs from slot 0 under `--split-mode layer`; see PersistentLayout::persistent_for.
    // `main_page_groups`: nullopt (default) previews at the curve's minimum page-group point,
    // unchanged from before. A value previews at that exact page count instead (see
    // DeviceLayoutPreview's comment) -- used for exact post-resolve admission under
    // `--kv-capacity auto`.
    [[nodiscard]] DeviceLayoutPreview
    layout_preview(int slot, std::optional<std::uint32_t> main_page_groups = std::nullopt) const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    // Debug-only, OFF by default: capture of the full-vocabulary logits behind the most recently
    // sampled token, as raw BF16 bits (see ProgramImplCore::logits_capture in the impl runtime).
    // Nothing is allocated, copied, or otherwise changed until enable_logits_capture(true) runs,
    // so an engine that never asks for it executes exactly as before. The span is valid only
    // immediately after the caller's own round completes on a single-lane engine, and is empty
    // while capture is disabled. Added for the greedy tp1-vs-tp2 parity harness
    // (tools/tp2/parity.cpp); not part of any wire-facing API.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept;
    void enable_logits_capture(bool enabled);
    [[nodiscard]] std::span<const DebugTensorSnapshot> last_parity_trace() const noexcept;
    void enable_parity_trace(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, compare rank 1's MTP
    // egress record with rank 0's. The ranks run the acceptance Op over bit-identical inputs, so
    // the records are argued to agree; enabling this measures it instead. Off, execution is
    // unchanged; on, it costs one small device-to-host copy and a host compare per round. Counters
    // are cumulative over the Program's lifetime. No-op at tp1 or without MTP.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      const typename V::ModelView*,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      ExecutionContext&);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

// `peer_model` is rank 1's own model view at tp == 2 and nullptr at tp == 1; `execution` supplies
// the device contexts (one or two). The two must agree: a non-null peer view with a tp1 execution
// context, or the reverse, is rejected.
template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               const typename Variant::ModelView* peer_model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               ExecutionContext& execution);

} // namespace ninfer::targets::qwen3_6

#pragma once

#include "runtime/contract/types.h"

#include <cstddef>
#include <span>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// Resolves one KV capacity plan shared by every device in a tensor-parallel group. KV addressing
// requires every device to carry the same number of Main KV pages (attention/KV addressing is
// already head-local; the page count itself is not), so all devices must agree on a single
// `main_page_groups` / `resolved_tokens` outcome rather than each picking its own.
//
// `curve` describes ONE device's own per-page byte cost. At tp2 that curve already reflects the
// device's own 2-of-4 KV heads (half the bytes of the tp1 4-head curve for the same page geometry
// and page count) -- constructing that per-device curve is target/model-layer work, out of scope
// here. Because the split is symmetric, every device shares the same curve, so this function
// takes one curve plus one budget per device: `available_runtime_bytes_per_device[i]` is device
// i's own free-memory budget (e.g. from its own `cudaMemGetInfo`), read independently per device
// in automatic mode. The plan is sized to the tightest ("bottleneck") device so no device is ever
// overcommitted; every device then reserves `runtime_reservation_bytes` from the single returned
// resolution (same byte count on every device, since the curve is symmetric).
//
// Under `--split-mode layer` the devices are NOT symmetric: each owns a disjoint contiguous layer
// range, so the two owners' persistent footprints differ. The curve remains single and shared --
// this function is unchanged -- but its caller builds the curve's base as the MAX over owning
// slots (M2.2), so the "bottleneck device" budget is tested against the worst owner's requirement
// instead of owner 0's. That is representable in one affine curve only while both owners share an
// identical per-page-group stride. M5.3 corrected an earlier claim here that they always do: the
// stride is proportional to each owner's full-attention layer count, so it is equal only at the
// boundaries where those counts coincide (32..35 for this 64-layer target). The condition is now
// ENFORCED rather than assumed -- make_sequence_planner_impl (layouts_impl.h) computes both
// owners' strides at curve construction and throws when they differ -- so any curve reaching this
// function satisfies it. This three-arg form is CONSERVATIVE under layer split
// (`min(F) - max(B)`); the four-arg overload below is the exact per-device form for Automatic mode.
//
// A single-element span reduces to `resolve_kv_capacity` exactly (tp1 stays byte-identical; tp1
// call sites are untouched and do not need to route through this function).
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity_symmetric(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_device);

// M2.2b: exact per-device AUTOMATIC capacity for the asymmetric (layer-split) case. The three-arg
// overload above charges `min_i(F_i) - max_i(B_i)` -- the curve's base is already the MAX over
// owning slots, and the budget is the MIN over devices -- which is safe but CONSERVATIVE: the
// exact common-stride capacity is `min_i(F_i - B_i)`, and the two differ whenever the device with
// the least free memory is not the device with the largest fixed reservation (F=[100,200],
// B=[0,50]: separated form 50, exact 100).
//
// `minimum_reservation_bytes_per_device[i]` is device i's OWN minimum-page-group reservation
// (persistent + workspace + request-transient + graph allowance, the same composition
// layouts_impl.h sums into `device_reservation_bytes` before taking the max), and its max must
// equal `curve.minimum_device_reservation_bytes` (logic_error otherwise -- the caller and the
// planner have drifted). The page count stays SINGLE and COMMON; only its derivation is exact:
// Automatic mode requires `F_i >= headroom + B_i` on every device (rejection names the device),
// then grows by `min_i(F_i - headroom - B_i) / stride`. The returned `runtime_reservation_bytes`
// remains the curve's (max-base) value so `finalize()`'s affine identity check is unchanged;
// `available_after_weights_bytes` is the bottleneck (argmin residual) device's budget and
// `planned_slack_bytes` is the worst device's own `F_i - B_i - delta`.
//
// Explicit mode is deliberately UNCHANGED (delegates to the three-arg overload): the accepted
// explicit gate matrix (e.g. 46/18 @128K rejecting on the resolver's final check) is pinned to the
// conservative form, and widening it is a separate gate with its own evidence. An empty
// `minimum_reservation_bytes_per_device` span also delegates unchanged (the caller had no exact
// per-device preview available). Symmetric bases (TP2, single device) are byte-identical to the
// three-arg result by construction.
[[nodiscard]] KvCapacityResolution
resolve_kv_capacity_symmetric(const KvCapacityPolicy& policy, const SequenceCapacityCurve& curve,
                              std::span<const std::size_t> available_runtime_bytes_per_device,
                              std::span<const std::size_t> minimum_reservation_bytes_per_device);

} // namespace ninfer::runtime

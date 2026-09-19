#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    // ---- M2.1: exact admission for `--kv-capacity auto` ----
    //
    // `automatic` above (max_concurrency-shaped curve: minimum=2, maximum=6 page groups) already
    // proves the resolver is a MAXIMIZER, not "minimum then growth": it picked pages=4, strictly
    // above the curve's minimum of 2. That is the regime that only exists when
    // curve.minimum_main_page_groups < curve.maximum_main_page_groups -- i.e. only reachable at
    // `--max-concurrency > 1` (layouts_impl.h: minimum_pages = max(logical_pages, max_concurrency);
    // maximum_pages = max_concurrency * logical_pages; the two coincide, and the auto-fit growth
    // branch at kv_capacity.cpp is dead, when max_concurrency == 1, the CLI default). The cases
    // below make that explicit and add the two things `automatic`/`capped` don't cover: one
    // concrete resolved value for a single `auto` call (not "which of two runs"), and the
    // multi-device bottleneck selection `resolve_kv_capacity_symmetric` performs.

    // 1. auto produces ONE concrete resolved capacity for a given (curve, budget) pair -- the
    //    resolver is deterministic, not a range.
    const auto one_shot =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures += check(one_shot.main_page_groups == automatic.main_page_groups &&
                          one_shot.resolved_tokens == automatic.resolved_tokens,
                      "auto did not produce a single deterministic resolved capacity");

    // 2. Growth regime (max_concurrency > 1 shaped curve, minimum < maximum): resolved page count
    //    is STRICTLY GREATER than curve.minimum_main_page_groups, and page-group rounding (64
    //    tokens = 1 page, matching kPagedKVPageSize) holds at the resolved point, not just at the
    //    minimum. This is the actual regression case for the M2.1 defect: admission built from
    //    curve.minimum_main_page_groups would under-charge relative to this resolved value.
    failures += check(automatic.main_page_groups > curve.minimum_main_page_groups,
                      "auto growth case did not resolve strictly above the curve minimum -- test "
                      "is not exercising the M2.1 defect");
    failures +=
        check(automatic.resolved_tokens == automatic.main_page_groups * curve.main_page_tokens,
              "resolved token count is not page-group-count * main_page_tokens (64-token rounding)");
    // Owner-local exact BYTES at the resolved point must be strictly greater than at the minimum
    // point -- this is the actual M2.1 claim, not just page/token counts. `layout_preview`'s
    // resolved-page-group overload (registry.cpp phase B) computes bytes via
    // build_sequence_candidate, which finalize_sequence_plan_impl (layouts_impl.h ~958) asserts is
    // byte-exact to `curve.reservation_bytes(main_page_groups)` -- so this curve-level assertion is
    // the byte-exact proxy for that layout-level fact, at the same host-only level
    // test_kv_capacity_tp2.cpp already operates at. (A direct layout_preview/build_sequence_candidate
    // call needs a real DeviceContext -- validate_target_options() checks device.sm() -- so it is
    // out of reach for this host-only binary; see the M2.1 report for why that gap is proxied here
    // rather than silently dropped.)
    failures += check(curve.reservation_bytes(automatic.main_page_groups) >
                          curve.reservation_bytes(curve.minimum_main_page_groups),
                      "resolved-capacity reservation bytes are not strictly greater than the "
                      "minimum-point bytes -- M2.1 under-charge would not be caught");

    // 3. No-growth regime (max_concurrency == 1 shaped curve: minimum == maximum): resolved ==
    //    minimum, documenting that auto is a structural no-op there and phase-A/phase-B previews
    //    must coincide.
    const ninfer::runtime::SequenceCapacityCurve fixed_curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 4,
        .maximum_main_page_groups             = 4,
        .minimum_device_reservation_bytes     = 2000,
        .bytes_per_additional_main_page_group = 0,
    };
    const auto no_growth = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), fixed_curve, 10000);
    failures += check(no_growth.main_page_groups == fixed_curve.minimum_main_page_groups &&
                          no_growth.main_page_groups == fixed_curve.maximum_main_page_groups,
                      "max_concurrency==1-shaped curve (minimum==maximum) did not resolve to the "
                      "single fixed point");

    // 4. Bottleneck device selection: resolve_kv_capacity_symmetric must resolve against
    //    whichever device has the LEAST free memory, regardless of position in the span -- GPU0
    //    limiting and GPU1 limiting must both work.
    // 1100 clears the minimum (>= 1050 = 1000 reservation + 50 headroom) but stays below the
    // next page-group's threshold (1128 = 1000 + 1*128), so it resolves to exactly
    // curve.minimum_main_page_groups -- distinct from 1360's pages=4, so the bottleneck value is
    // provably the one selected, not incidentally equal to the generous device's result.
    const std::size_t gpu0_bytes[] = {1100, 1360}; // GPU0 (index 0) is the bottleneck.
    const auto gpu0_limits = ninfer::runtime::resolve_kv_capacity_symmetric(
        ninfer::KvCapacityPolicy::automatic(50), curve, gpu0_bytes);
    failures += check(gpu0_limits.main_page_groups == curve.minimum_main_page_groups,
                      "symmetric resolution did not bind to GPU0 when GPU0 is the bottleneck");

    const std::size_t gpu1_bytes[] = {1360, 1100}; // GPU1 (index 1) is the bottleneck.
    const auto gpu1_limits = ninfer::runtime::resolve_kv_capacity_symmetric(
        ninfer::KvCapacityPolicy::automatic(50), curve, gpu1_bytes);
    failures += check(gpu1_limits.main_page_groups == curve.minimum_main_page_groups,
                      "symmetric resolution did not bind to GPU1 when GPU1 is the bottleneck");

    const std::size_t both_generous[] = {1360, 10000}; // GPU0 still the (less generous) bottleneck.
    const auto gpu0_generous = ninfer::runtime::resolve_kv_capacity_symmetric(
        ninfer::KvCapacityPolicy::automatic(50), curve, both_generous);
    failures +=
        check(gpu0_generous.main_page_groups == automatic.main_page_groups,
              "symmetric resolution did not bind to the tighter of two generous device budgets");

    // ---- M2.2b: exact per-device Automatic capacity (four-arg overload) ----
    //
    // Curve base is the MAX over owning slots (layouts_impl.h device_reservation_bytes). With
    // per-slot bases B and per-device budgets F, the conservative three-arg form charges
    // min(F) - max(B); the exact form charges min_i(F_i - B_i). Stride 128, headroom 0 so the
    // page arithmetic is legible.
    const ninfer::runtime::SequenceCapacityCurve split_curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 40,
        .minimum_device_reservation_bytes     = 1000, // = max(B)
        .bytes_per_additional_main_page_group = 128,
    };
    const auto no_headroom = ninfer::KvCapacityPolicy::automatic(0);

    // A. Counterexample: device 0 has the LEAST free memory but the SMALLEST base. Conservative:
    //    (1400 - 1000) / 128 = 3 additional -> 5 pages. Exact: min(1400 - 200, 2000 - 1000) = 1000
    //    -> 7 additional -> 9 pages. The two formulas must differ, and the exact one must win.
    {
        const std::size_t F[] = {1400, 2000};
        const std::size_t B[] = {200, 1000};
        const auto conservative =
            ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F);
        const auto exact =
            ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F, B);
        failures += check(conservative.main_page_groups == 5,
                          "M2.2b A: conservative form did not resolve to 5 page groups");
        failures += check(exact.main_page_groups == 9,
                          "M2.2b A: exact per-device form did not resolve to 9 page groups");
        failures += check(conservative.main_page_groups < exact.main_page_groups,
                          "M2.2b A: constructed counterexample did not separate the two formulas");
        // The reservation stays on the shared max-base curve (finalize identity), the bottleneck
        // budget names the argmin-residual device, and slack is the worst device's own residual.
        failures += check(exact.runtime_reservation_bytes == 1000 + 7 * 128 &&
                              exact.minimum_runtime_reservation_bytes == 1000 &&
                              exact.available_after_weights_bytes == 2000 &&
                              exact.planned_slack_bytes == 104,
                          "M2.2b A: exact resolution reported the wrong reservation/budget/slack");
        // F. No over-admission: every device must still cover its own base plus the delta.
        const std::size_t delta = exact.runtime_reservation_bytes - 1000;
        failures += check(B[0] + delta <= F[0] && B[1] + delta <= F[1],
                          "M2.2b F: exact resolution over-admitted a device");
    }

    // A2. Minimum-point counterexample: the conservative form REJECTS (min F 900 < max B 1000)
    //     while every device individually covers its own base (900 - 0, 2000 - 1000). Exact must
    //     admit at exactly the minimum point (residuals 900 and 1000 -> 7 additional -> 9).
    {
        const std::size_t F[] = {900, 2000};
        const std::size_t B[] = {0, 1000};
        bool conservative_rejected = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F);
        } catch (const std::invalid_argument&) { conservative_rejected = true; }
        failures += check(conservative_rejected,
                          "M2.2b A2: conservative form unexpectedly admitted min F < max B");
        const auto exact =
            ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F, B);
        failures += check(exact.main_page_groups == 9,
                          "M2.2b A2: exact form did not admit the per-device-feasible plan");
    }

    // B. Agreement: the tightest device is also the heaviest owner, so both forms coincide.
    {
        const std::size_t F[] = {1400, 2000};
        const std::size_t B[] = {1000, 200};
        const auto conservative =
            ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F);
        const auto exact =
            ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F, B);
        failures += check(conservative.main_page_groups == exact.main_page_groups &&
                              conservative.main_page_groups == 5 &&
                              conservative.runtime_reservation_bytes ==
                                  exact.runtime_reservation_bytes &&
                              conservative.planned_slack_bytes == exact.planned_slack_bytes &&
                              conservative.available_after_weights_bytes ==
                                  exact.available_after_weights_bytes,
                          "M2.2b B: the two formulas disagreed where they must agree");
    }

    // C. Headroom is charged per device: with headroom 50, device 1's residual is 950, so the
    //    exact count drops to 2 + 950/128 = 9 (unchanged here) but a device at F = B + 49 rejects
    //    NAMING that device even though the other has plenty.
    {
        const std::size_t F[] = {5000, 1049};
        const std::size_t B[] = {200, 1000};
        bool rejected = false;
        std::string message;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::automatic(50), split_curve, F, B);
        } catch (const std::invalid_argument& error) {
            rejected = true;
            message  = error.what();
        }
        failures += check(rejected && message.find("device slot 1") != std::string::npos,
                          "M2.2b C: per-device headroom shortfall was not rejected naming slot 1");
    }

    // D. Explicit mode is UNCHANGED: with bases supplied it must produce the identical result the
    //    three-arg (conservative) form produces, including the conservative rejection.
    {
        const std::size_t F[] = {1400, 2000};
        const std::size_t B[] = {200, 1000};
        const auto explicit3 = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::explicit_capacity(64 * 5), split_curve, F);
        const auto explicit4 = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::explicit_capacity(64 * 5), split_curve, F, B);
        failures += check(explicit3.main_page_groups == 5 &&
                              explicit4.main_page_groups == explicit3.main_page_groups &&
                              explicit4.runtime_reservation_bytes ==
                                  explicit3.runtime_reservation_bytes &&
                              explicit4.planned_slack_bytes == explicit3.planned_slack_bytes &&
                              explicit4.available_after_weights_bytes ==
                                  explicit3.available_after_weights_bytes,
                          "M2.2b D: explicit mode changed when per-device bases were supplied");
        bool rejected3 = false;
        bool rejected4 = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::explicit_capacity(64 * 6), split_curve, F);
        } catch (const std::invalid_argument&) { rejected3 = true; }
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::explicit_capacity(64 * 6), split_curve, F, B);
        } catch (const std::invalid_argument&) { rejected4 = true; }
        failures += check(rejected3 && rejected4,
                          "M2.2b D: explicit mode stopped rejecting conservatively with bases");
    }

    // E. Empty base span and symmetric bases both reduce EXACTLY to the three-arg form (the
    //    single-device / TP2 byte-identity guarantee).
    {
        const std::size_t F[] = {1360, 1100};
        const std::size_t Bsym[] = {1000, 1000};
        const auto three = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::automatic(50), curve, F);
        const auto empty = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::automatic(50), curve, F, {});
        const auto symmetric = ninfer::runtime::resolve_kv_capacity_symmetric(
            ninfer::KvCapacityPolicy::automatic(50), curve, F, Bsym);
        const auto same = [](const ninfer::runtime::KvCapacityResolution& a,
                             const ninfer::runtime::KvCapacityResolution& b) {
            return a.main_page_groups == b.main_page_groups &&
                   a.runtime_reservation_bytes == b.runtime_reservation_bytes &&
                   a.available_after_weights_bytes == b.available_after_weights_bytes &&
                   a.planned_slack_bytes == b.planned_slack_bytes &&
                   a.resolved_tokens == b.resolved_tokens;
        };
        failures += check(same(three, empty),
                          "M2.2b E: empty base span did not reduce to the three-arg form");
        failures += check(same(three, symmetric),
                          "M2.2b E: symmetric bases did not reduce to the three-arg form");
    }

    // G. Drift guard: bases whose max disagrees with the curve base are a logic error, and a
    //    base span of the wrong length is rejected.
    {
        const std::size_t F[] = {1400, 2000};
        const std::size_t Bwrong[] = {200, 900};
        bool drift = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F,
                                                                 Bwrong);
        } catch (const std::logic_error&) { drift = true; }
        failures += check(drift, "M2.2b G: base/curve disagreement was not a logic_error");
        const std::size_t Bshort[] = {1000};
        bool length = false;
        try {
            (void)ninfer::runtime::resolve_kv_capacity_symmetric(no_headroom, split_curve, F,
                                                                 Bshort);
        } catch (const std::invalid_argument&) { length = true; }
        failures += check(length, "M2.2b G: mismatched base span length was accepted");
    }

    // M5.3: growth-arm capacity arithmetic on the shared affine curve. Asymmetric layer splits now
    // throw at curve CONSTRUCTION (layouts_impl.h make_sequence_planner_impl detects per-owner
    // stride divergence), so every curve that reaches the resolver is one whose two owners share a
    // stride -- which is exactly the configuration these cases pin. Synthetic curves only: no
    // SequencePlanner, no device, no artifact.
    {
        // Equal-stride ("agreement") curve: base 1000, stride 128, groups in [2, 10].
        const ninfer::runtime::SequenceCapacityCurve shared{
            .main_page_tokens                     = 64,
            .minimum_main_page_groups             = 2,
            .maximum_main_page_groups             = 10,
            .minimum_device_reservation_bytes     = 1000,
            .bytes_per_additional_main_page_group = 128,
        };
        const ninfer::KvCapacityPolicy automatic_no_headroom = ninfer::KvCapacityPolicy::automatic(0);

        // 1. Agreement case still grows exactly as before (guards accepted M2.2b behaviour). Two
        //    devices, equal budgets and equal bases: 1000 + 4*128 = 1512 <= 1512, so 4 extra groups.
        {
            const std::size_t budgets[] = {1512, 1512};
            const std::size_t bases[]   = {1000, 1000};
            const auto grown = ninfer::runtime::resolve_kv_capacity_symmetric(
                automatic_no_headroom, shared, budgets, bases);
            failures += check(grown.main_page_groups == 6,
                              "M5.3.1: equal-stride agreement case did not grow to 6 page groups");
            failures += check(grown.runtime_reservation_bytes == 1512,
                              "M5.3.1: agreement case reservation is not 1512");
        }

        // 2. Exact boundary fit: budget admits exactly 3 extra groups (1000 + 3*128 = 1384) with
        //    zero slack on the tighter device. Must resolve to 5, not 6.
        {
            const std::size_t budgets[] = {1384, 4000};
            const std::size_t bases[]   = {1000, 1000};
            const auto fit = ninfer::runtime::resolve_kv_capacity_symmetric(
                automatic_no_headroom, shared, budgets, bases);
            failures += check(fit.main_page_groups == 5,
                              "M5.3.2: exact boundary fit did not resolve to 5 page groups");
            failures += check(fit.runtime_reservation_bytes == 1384,
                              "M5.3.2: exact fit reservation is not 1384");
            failures += check(fit.planned_slack_bytes == 0,
                              "M5.3.2: exact fit should leave zero slack on the tight device");
        }

        // 3. One group over: one byte short of the next group must NOT take it (stays at 5).
        {
            const std::size_t budgets[] = {1384 + 127, 4000};
            const std::size_t bases[]   = {1000, 1000};
            const auto under = ninfer::runtime::resolve_kv_capacity_symmetric(
                automatic_no_headroom, shared, budgets, bases);
            failures += check(under.main_page_groups == 5,
                              "M5.3.3: one-byte-short budget incorrectly took another page group");
        }

        // 4. Explicit mode is unchanged by any of the above (delegates to the conservative form).
        {
            const std::size_t budgets[] = {1384, 4000};
            const std::size_t bases[]   = {1000, 1000};
            const auto explicit_with_bases = ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::explicit_capacity(320), shared, budgets, bases);
            const auto explicit_without_bases = ninfer::runtime::resolve_kv_capacity_symmetric(
                ninfer::KvCapacityPolicy::explicit_capacity(320), shared, budgets);
            failures += check(explicit_with_bases.main_page_groups == 5 &&
                                  explicit_with_bases.resolved_tokens == 320,
                              "M5.3.4: explicit capacity did not resolve to its requested point");
            failures += check(explicit_with_bases.main_page_groups ==
                                      explicit_without_bases.main_page_groups &&
                                  explicit_with_bases.runtime_reservation_bytes ==
                                      explicit_without_bases.runtime_reservation_bytes,
                              "M5.3.4: explicit mode changed when per-device bases were supplied");
        }
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

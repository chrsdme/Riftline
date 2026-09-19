# Architecture

Riftline separates hard feasibility from empirical performance ranking.

## Hard-constraint plane

```text
candidate PlacementSpec
  -> ownership
  -> exact device-specific memory requirements
  -> admission
  -> FEASIBLE / REJECTED
```

This plane decides whether a placement may load. It accounts for device-specific residency, KV growth, workspace allowance, request-transient pressure, allocator reserve, and endpoint ownership.

## Performance plane

```text
future calibration
  -> future performance model
  -> future candidate ranking
  -> future selection
  -> authoritative load revalidation
```

The performance plane is future work. Ranking will never be allowed to override hard feasibility.

## Runtime shape

The current implementation uses a contiguous transformer boundary and an explicit endpoint owner. For the qualified 64-layer target, GPU0 owns layers `0..N-1`, GPU1 owns layers `N..63`, and final norm / output head may be owned by either endpoint device when admitted.

The current endpoint path returns logits and required result data to the primary sampling/control path. That path is correct for the qualified scope but is not presented as fully optimized transport.

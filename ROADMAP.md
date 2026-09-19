# Roadmap

Riftline's roadmap deliberately separates hard memory feasibility from empirical performance modelling and automatic placement.

## Completed foundation

- heterogeneous execution groundwork
- contiguous runtime layer placement
- owner-local KV/GDN state
- per-device persistent layouts and context-aware admission
- automatic KV-capacity hardening for supported modes
- workspace observability
- endpoint placement and endpoint execution
- explicit placement specification
- hard-feasibility evaluation
- exhaustive boundary / endpoint-owner enumeration
- fail-closed handling of heterogeneous KV-growth slopes

## Next: calibration and provenance

Build a provenance-aware calibration layer for the actual GPUs, runtime, artifact, operation shape, transfer direction, and execution mode. Calibration records measurements; it does not rank or choose placements.

## Planned: calibrated performance model

Use qualified calibration data to estimate the cost of feasible placements while keeping prefill, decode, operation class, endpoint work, and transfer cost distinct.

## Planned: automatic placement

Rank only hard-feasible candidates, select a placement, and revalidate authoritative memory admission at load time. Performance ranking must never override hard feasibility.

## Future optimization tracks

- remove the redundant endpoint hidden-state round trip
- investigate endpoint-local sampling or reduced logits transfer
- evaluate low-latency host-staged/mailbox transport for synchronization-sensitive exchanges
- prototype heterogeneous pipelined prefill
- support concurrent heterogeneous execution with proper per-device capacity curves
- qualify additional models and GPU combinations
- investigate future KV/storage-policy dimensions without expanding the first placement search space prematurely

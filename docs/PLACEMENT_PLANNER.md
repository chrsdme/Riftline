# Placement Planner

The current placement planner is a hard-feasibility planner, not an automatic optimizer.

## PlacementSpec

The v1 candidate shape includes:

- contiguous transformer boundary
- endpoint owner

The v1 candidate dimensions intentionally do not include:

- KV format
- storage tier
- graph strategy
- speculative policy
- arbitrary per-layer mapping

## Enumeration

The planner enumerates legal layer-boundary / endpoint-owner candidates and evaluates each candidate through structured admission. Accepted evidence covers 126 candidates for the current target shape.

## AUTO boundary

Automatic placement selection does not yet exist. `AUTO` should not be documented or treated as implemented until calibrated ranking and authoritative load-time revalidation exist.

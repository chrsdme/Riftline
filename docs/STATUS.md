# Status

## Public source checkpoint

- Source checkpoint: `78d2b4d69fe35e61d151eb6aa189e5e366f3da7c`
- Project: Riftline
- Repository: <https://github.com/chrsdme/Riftline>

## Currently qualified target

- Reference GPUs: RTX 5060 Ti 16 GB + RTX 3060 12 GB
- Reference artifact: Qwen3.8-27B groupwise `.ninfer`
- Topology: no useful direct CUDA P2P path on the reference machine

## Implemented capabilities

- runtime contiguous layer split
- owner-local recurrent/KV state
- per-device memory admission
- endpoint placement and execution
- explicit `PlacementSpec`
- candidate feasibility evaluation
- exhaustive layer-boundary / endpoint-owner enumeration
- heterogeneous KV-growth hardening

## Not implemented yet

- provenance-aware performance calibration
- calibrated placement cost model
- performance ranking
- automatic placement selection
- optimized endpoint transport
- heterogeneous pipelined prefill

## Known limitations

- Endpoint transport is correct for the qualified scope but not fully optimized.
- Other models and GPU pairs require qualification before comparable claims are made.
- Concurrent heterogeneous growth is deliberately fail-closed where the current shared curve representation cannot model unequal per-device slopes exactly.

## Endpoint transient qualification

Endpoint transient accounting is currently **profile-qualified**.

The endpoint requirement is derived from the current target/configuration, while the workspace-dominance regression uses frozen qualified values rather than dynamically querying `build_workspace_plan`. A future target or workspace recipe must preserve or separately requalify that invariant.

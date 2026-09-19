# Upstream and Provenance

Riftline is a downstream experimental project derived from NInfer. It is not the upstream NInfer project and does not imply endorsement, review, or affiliation by upstream maintainers.

## Known lineage

- Upstream project: NInfer
- Upstream repository: <https://github.com/Neroued/ninfer>
- Upstream license: Apache License 2.0
- Upstream base recorded in the inherited `NOTICE`: `Neroued/ninfer` commit `feaf4dd`
- Intermediate fork recorded in the inherited `NOTICE`: `wamansou/ninfer-tp2-1m`

## Riftline snapshot provenance

- Riftline source checkpoint: `78d2b4d69fe35e61d151eb6aa189e5e366f3da7c`
- Public repository: <https://github.com/chrsdme/Riftline>

The initial public tree was exported from the tracked files at that source checkpoint and then given release-only documentation and hygiene cleanup. Internal agent state, development logs, raw evaluation outputs, local workspaces, caches, and model artifacts are intentionally excluded.

## Downstream focus

Riftline focuses on heterogeneous multi-GPU execution for unequal consumer NVIDIA GPUs. Its current reference topology is an RTX 5060 Ti 16 GB paired with an RTX 3060 12 GB without a useful direct CUDA P2P path.

The main architectural focus is hard feasibility for contiguous layer placement: per-device memory admission, owner-local model state, endpoint placement/execution, candidate enumeration, and fail-closed handling of heterogeneous KV growth. Performance calibration, ranking, and automatic placement are separate future layers.

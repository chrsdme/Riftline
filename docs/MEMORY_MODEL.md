# Memory Model

`16 GB + 12 GB` is not one 28 GB allocator.

Each physical GPU must satisfy its own admission ledger. A placement is valid only if every device can carry its own required residency and growth.

## Per-device ledger

Conceptually, each device is charged for:

- weights
- persistent state
- KV
- workspace
- request transient memory
- graph allowance
- allocator reserve

Persistent storage generally adds within a device. Sequential transient stages may be governed by peak/max lifetime rather than blind summation.

## Heterogeneous growth

Context growth is device-specific:

```text
required_0(g) = B0 + S0*g
required_1(g) = B1 + S1*g
```

`B0` and `B1` may differ. `S0` and `S1` may differ. The KV-growth slope depends on placement of KV-bearing/full-attention layers, so placement affects both initial residency and context-growth pressure.

For Qwen3.8-27B in the current target:

- 64 transformer layers
- 16 full-attention layers
- 48 GDN layers

The workspace evidence is qualified as `SUFFICIENT_WITH_EXTERNAL_MARGIN`; it is not a universal byte-exact proof for every future target, workspace recipe, or profile.

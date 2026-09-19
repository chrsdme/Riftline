# Benchmarks

Only benchmark claims supported by accepted project evidence are included here.

## Reference local result

Hardware:

- RTX 5060 Ti 16 GB
- RTX 3060 12 GB
- no useful direct CUDA P2P path on the reference machine

Artifact:

- Qwen3.8-27B groupwise `.ninfer`

Source checkpoint:

- `78d2b4d69fe35e61d151eb6aa189e5e366f3da7c`

Measured decode result:

| Runtime mode | Placement | Decode |
|---|---|---:|
| TP2 baseline | tensor-parallel baseline | 12.53 tok/s |
| Layer split | 56/8 contiguous split | 22.84 tok/s |

This is approximately 1.82x for the tested layer split against the tested TP2 baseline on the reference system. It is not a claim that layer splitting is universally faster or that 56/8 is automatically optimal.

Additional endpoint measurements showed 58/6 with endpoint owner 1 was runnable and measured at the accepted 8K configuration, but ranking and automatic placement selection remain unimplemented.

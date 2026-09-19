# Riftline

**Topology-aware inference across unequal GPUs.**

Riftline is an experimental heterogeneous multi-GPU inference runtime derived from [NInfer](https://github.com/Neroued/ninfer). It is focused on running selected models across GPUs with different architectures, memory capacities, and performance characteristics without treating those devices as interchangeable tensor-parallel ranks.

The current reference topology is:

- NVIDIA GeForce RTX 5060 Ti 16 GB
- NVIDIA GeForce RTX 3060 12 GB
- no useful direct CUDA P2P path on the reference machine

Riftline uses contiguous transformer-layer placement, owner-local state, explicit endpoint placement, and per-device admission accounting.

## Why heterogeneous placement?

Unequal GPUs are not one larger allocator. A 16 GB card plus a 12 GB card is not a single 28 GB GPU, and the two devices do not necessarily accumulate context-dependent memory at the same rate.

Riftline models each device independently:

```text
required_0(g) = B0 + S0*g
required_1(g) = B1 + S1*g
```

`B0` and `B1` represent device-local residency. `S0` and `S1` represent context/KV growth. Both can differ because placement changes which device owns KV-bearing full-attention layers, recurrent state, endpoint weights, workspace pressure, and request transients.

For the currently qualified Qwen3.8-27B target, the model has 64 transformer layers: 16 full-attention layers and 48 GDN layers.

## Current status

Implemented:

- heterogeneous layer execution
- runtime contiguous layer split
- owner-local recurrent/KV state
- per-device memory admission
- endpoint placement and execution
- explicit `PlacementSpec`
- placement feasibility evaluation
- exhaustive layer-boundary / endpoint-owner enumeration
- fail-closed heterogeneous KV-growth hardening

Planned, but not implemented yet:

- provenance-aware performance calibration
- calibrated placement cost model
- performance ranking
- automatic placement selection
- optimized endpoint transport
- heterogeneous pipelined prefill

## Architecture overview

```text
GPU0 / RTX 5060 Ti
  token embedding
  layers 0 .. N-1
         |
         | hidden state
         v
GPU1 / RTX 3060
  layers N .. 63
  optional final norm + lm_head
         |
         | logits / result path
         v
primary sampling/control path
```

Endpoint ownership is explicit. Moving final norm and `lm_head` can make an otherwise impossible placement feasible, but the current endpoint traffic path is not presented as fully optimized.

## Current qualification

The heterogeneous work is currently qualified around the Qwen3.8-27B groupwise NInfer artifact and the reference RTX 5060 Ti + RTX 3060 topology above. Other model targets remain in the source tree, but should not be assumed to have the same heterogeneous qualification without separate validation.

## Benchmark snapshot

Accepted project evidence for the reference system includes:

| Configuration | Hardware | Artifact | Placement | Decode |
|---|---|---|---|---:|
| TP2 baseline | RTX 5060 Ti 16 GB + RTX 3060 12 GB | Qwen3.8-27B groupwise `.ninfer` | tensor parallel | 12.53 tok/s |
| Layer split | RTX 5060 Ti 16 GB + RTX 3060 12 GB | Qwen3.8-27B groupwise `.ninfer` | 56/8 contiguous split | 22.84 tok/s |

This is approximately 1.82x the measured decode throughput for the tested layer split versus the tested TP2 baseline on this machine and configuration. It is not a general claim that every layer split is faster or that 56/8 is optimal.

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for the scope of published benchmark claims.

## Build and usage

The commands below are inherited from the current source tree and were not revalidated during public-release preparation.

Requirements include CMake 3.28+, a C++20 compiler, CUDA 12.8+, FFmpeg development libraries, libcurl, and supported CUDA architecture settings. A build targeting the reference Ampere + Blackwell pair is:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86;120a"
cmake --build build --parallel
```

Expected application paths after a successful build:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

See [docs/cli.md](docs/cli.md) and [docs/serving.md](docs/serving.md) for inherited CLI and serving documentation.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Memory model](docs/MEMORY_MODEL.md)
- [Placement planner](docs/PLACEMENT_PLANNER.md)
- [Status](docs/STATUS.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Roadmap](ROADMAP.md)
- [Upstream and provenance](UPSTREAM.md)
- [Acknowledgements](ACKNOWLEDGEMENTS.md)
- [Research references and design notes](docs/RESEARCH_NOTES.md)

## Project status

Riftline is research-grade systems software. Capability and performance claims should be interpreted only for the stated hardware, artifact, source checkpoint, context, and runtime mode.

Riftline is a downstream project derived from NInfer. It is not the upstream NInfer project and does not imply endorsement by upstream maintainers. Code lineage and legal attribution are recorded in [UPSTREAM.md](UPSTREAM.md) and [NOTICE](NOTICE); research influences and comparative references are recorded separately in [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) and [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).

## Model and artifact licensing

The repository contains model cards and conversion/evaluation support for Qwen-family artifacts inherited from the NInfer lineage. Model and artifact-specific provenance and licensing are documented under [`model-cards/`](model-cards/) and in the inherited [`NOTICE`](NOTICE). Model weights are not included in this source repository.

## License

Apache License 2.0. See [LICENSE](LICENSE). Third-party components retain their own license files.

# Acknowledgements

Riftline exists because of the work of the NInfer community and a wider set of inference-runtime, multi-GPU, quantization, and systems projects. This file records people and projects that materially influenced Riftline's implementation or design decisions. It is intentionally separate from [`NOTICE`](NOTICE): `NOTICE` and the license files record code lineage and legally relevant attribution, while this document also covers research references and comparative systems.

Listing a project here does **not** imply endorsement, affiliation, or that its source code is incorporated into Riftline. Where Riftline directly inherits code, the applicable lineage and licenses remain authoritative in [`UPSTREAM.md`](UPSTREAM.md), [`NOTICE`](NOTICE), [`LICENSE`](LICENSE), and `third_party/`.

## Upstream source lineage

Riftline is a downstream derivative of:

- **[Neroued/ninfer](https://github.com/Neroued/ninfer)** and its contributors — the upstream NInfer runtime, artifact format, kernels, serving stack, and model-target foundation from which Riftline is derived. The inherited `NOTICE` records upstream base commit `feaf4dd`.
- **[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m)** / **Wael Mansour** — intermediate NInfer TP2 and YaRN work preserved in Riftline's inherited source lineage and `NOTICE`.

## Architecture and planner references

The following projects materially informed planner, memory-accounting, residency, calibration, or evidence-management decisions. Riftline borrowed concepts and methodology, not wholesale implementations.

- **[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)** — component-aware placement, persistent-versus-transient accounting, output-device movement, and host-staged/shared-host transport patterns. Research snapshot: `02aef45cd681b960a00afcd0749a4ab99e6c1bfe`.
- **[MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install)** — empirical profile/calibration methodology and the separation between measured history and hard admission. Research snapshot: `622c7965ed5e02a13b188c9ef21bb9857fd2ba28`.
- **[MirkoCovizzi/ninfer-rtx5090-mobile](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile)** — resource scheduling, device/host residency, state images, paged/host KV, and long-context runtime ideas. Research snapshot: `8debca388fb84abe5a36386b5a45ee122ca80314`.
- **[Schestex/ninfer-lab](https://huggingface.co/datasets/Schestex/ninfer-lab)** — candidate isolation, source/artifact pinning, smoke qualification, promotion, and failure/provenance recording. The local Git research snapshot used by the project was `40eedb6c970f2dc969ad82b151c247426f11ba0a`.

## Heterogeneous multi-GPU and performance references

- **[parallelno/ninfer-dflash2-tp2](https://github.com/parallelno/ninfer-dflash2-tp2)** — no-P2P TP2 transport, pinned-host/mailbox exchange, rank-local residency, and per-rank memory reporting. Research snapshot: `479b1e890e214885f3623eeb27d847d468a28c39`.
- **[alphastorm/ninfer](https://github.com/alphastorm/ninfer)** — NInfer resource scheduling, state/KV residency, portability, and memory-reporting ideas. Research snapshot: `82a1a22725009f6cefb78e946f9e2994c1ebd41b`.
- **[aivrar/multi-turboquant](https://github.com/aivrar/multi-turboquant)** — explicit analytical-versus-empirical provenance, capability checks, and calibration-result lifecycle ideas. Research snapshot: `cda91a0d2a43d19be8cdca9ad921c37a64b55188`.
- **[sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill)** — intra-prompt/diagonal pipeline scheduling across layer-split devices, informing a possible future heterogeneous prefill path. Research snapshot: `eac68893b4719634db5296354f83a42ca3a7041f`.
- **[Astrangemaninhere/ninfer-fusion](https://github.com/Astrangemaninhere/ninfer-fusion)** — future-facing KV/storage-policy, speculative-control, and graph-policy ideas; these are research references, not current Riftline planner dimensions. Research snapshot: `0eaac046ee98b5e5bd629ea02bc42c7f2503198f`.
- **[xiersweet2648/sm120-llm-kernels](https://github.com/xiersweet2648/sm120-llm-kernels)** — shape-aware SM120 kernel measurement and roofline methodology, informing Riftline's planned calibration layer. Research snapshot: `3696eb26ee97993b7255a2b0bff1dd2562bde840`.

Additional NInfer variants were inspected during development, including **[giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)** and **[mr-september/ninfer-2080ti-22g](https://github.com/mr-september/ninfer-2080ti-22g)**, mainly as implementation and compatibility references for consumer multi-GPU/Ampere paths.

## Comparative runtimes

Riftline's design and benchmark work was also compared against other inference runtimes. These are comparative references rather than source lineage:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — heterogeneous/layer-split behavior and long-context comparison.
- **[vLLM](https://github.com/vllm-project/vllm)** — tensor-parallel capacity and throughput comparison.
- **[SGLang](https://github.com/sgl-project/sglang)** — tensor-parallel and runtime-memory behavior comparison.
- **[Lucebox](https://github.com/Luce-Org/lucebox)** — alternative long-context and speculative-inference comparison.

These comparisons helped identify which limitations were specific to Riftline/NInfer and which were consequences of hardware topology, context growth, transient buffers, or synchronization strategy.

## Models, algorithms, and papers

Model/artifact provenance remains under [`model-cards/`](model-cards/) and the inherited [`NOTICE`](NOTICE). Algorithm- and paper-level citations already present in inherited technical documentation — including attention, paging, grouped-query attention, recurrent/state-space and speculative-decoding work — remain with the documents where those mechanisms are discussed.

## Scope of this list

Many additional repositories were surveyed during discovery. This file intentionally lists projects that materially affected an architectural decision, implementation direction, validation methodology, or comparative result; it is not intended to enumerate every repository that was viewed.

For the technical mapping from each reference to Riftline's current or future design, see [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md).

# Research References and Design Notes

This document records external projects that materially influenced Riftline's architecture, validation methodology, or future design direction. It is not a claim that any referenced project endorses, participates in, or contributes code to Riftline.

Code lineage and legally relevant attribution are documented separately in [`../UPSTREAM.md`](../UPSTREAM.md), [`../NOTICE`](../NOTICE), [`../LICENSE`](../LICENSE), and the licenses under `third_party/`.

## Classification

Riftline uses the following terms throughout this document:

- **Source lineage** — code inherited through the NInfer/TP2 ancestry and covered by the repository's license/provenance files.
- **Borrowed concept** — an architectural or methodological idea that informed Riftline, without implying source-code reuse.
- **Comparative reference** — another runtime used to understand performance, capacity, or topology behavior.
- **Future experiment** — a researched idea that is not currently implemented in Riftline.

## Source lineage

The executable source tree is derived from **[Neroued/ninfer](https://github.com/Neroued/ninfer)** through the intermediate TP2/YaRN work recorded in `NOTICE`, including **[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m)**. Those projects belong in Riftline's provenance chain rather than merely in its inspiration list.

## Architecture and planner references

| Reference | Research identity | Classification | Riftline lesson |
|---|---|---|---|
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | `02aef45cd681b960a00afcd0749a4ab99e6c1bfe` | Borrowed concept | Separate persistent storage from transient overhead; make component placement/output ownership explicit; use host-staged/shared-host transport concepts without importing symmetric TP wholesale. |
| [MiaAI-Lab Qwen3.8 16 GB kit](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install) | `622c7965ed5e02a13b188c9ef21bb9857fd2ba28` | Borrowed methodology | Empirical calibration/history can guide ranking, but heuristic overheads and empirical ceilings must not replace fail-closed memory admission. |
| [MirkoCovizzi/ninfer-rtx5090-mobile](https://github.com/MirkoCovizzi/ninfer-rtx5090-mobile) | `8debca388fb84abe5a36386b5a45ee122ca80314` | Borrowed concept | Resource scheduling, logical device/host residency, state images, paged/host KV, prefix reuse, and long-context management are useful concepts; single-5090 assumptions are not. |
| [Schestex/ninfer-lab](https://huggingface.co/datasets/Schestex/ninfer-lab) | local Git snapshot `40eedb6c970f2dc969ad82b151c247426f11ba0a` | Borrowed methodology | Pin repository revision and artifact identity, isolate candidates, preserve failures, and promote only qualified candidates. |

### ExLlamaV3

The most important lesson was architectural, not algorithmic: placement should have a vocabulary richer than "GPU 0 or GPU 1," while persistent storage and transient peaks should not be conflated. Riftline adopted the separation of concerns, but its hard feasibility remains based on NInfer's real per-device ownership and admission paths rather than ExLlamaV3's allocator heuristics.

### MiaAI-Lab material

The useful distinction was **hard feasibility versus empirical operating history**. A measured profile can be useful for future ranking and sanity checks, but a flat runtime-overhead estimate or previously observed maximum context cannot prove that a new placement is safe.

### Mirko NInfer work

This work reinforced the value of treating GPU/host residency, state images, paged KV, and resource scheduling as explicit runtime concerns. Riftline did not import single-GPU Blackwell assumptions into the mixed `sm_120a` + `sm_86` path.

### Schestex NInfer Lab

The main influence was evidence discipline: a candidate should be identifiable by source revision, artifact, configuration and result; failed candidates should remain attributable rather than disappearing into an informal "works/doesn't work" history.

## Heterogeneous multi-GPU and transport references

| Reference | Pinned HEAD | Classification | Riftline lesson |
|---|---|---|---|
| [parallelno/ninfer-dflash2-tp2](https://github.com/parallelno/ninfer-dflash2-tp2) | `479b1e890e214885f3623eeb27d847d468a28c39` | Borrowed concept / future experiment | On no-P2P consumer GPUs, choreography and synchronization can dominate small exchanges. Pinned-host/mailbox transport is worth testing after unnecessary transfers are removed. |
| [alphastorm/ninfer](https://github.com/alphastorm/ninfer) | `82a1a22725009f6cefb78e946f9e2994c1ebd41b` | Borrowed concept | State/KV residency, resource scheduling and per-device reporting should remain explicit and architecture-aware. |
| [aivrar/multi-turboquant](https://github.com/aivrar/multi-turboquant) | `cda91a0d2a43d19be8cdca9ad921c37a64b55188` | Borrowed methodology | Calibration records should state assumptions, provenance and validity; approximate aggregate VRAM formulas are not acceptable as Riftline's hard admission model. |
| [sergey-automation/TurboPrefill](https://github.com/sergey-automation/TurboPrefill) | `eac68893b4719634db5296354f83a42ca3a7041f` | Future experiment | Diagonal/intra-prompt scheduling could overlap GPU0 work on a later prefill chunk with GPU1 work on an earlier chunk, reducing layer-pipeline bubbles. |
| [Astrangemaninhere/ninfer-fusion](https://github.com/Astrangemaninhere/ninfer-fusion) | `0eaac046ee98b5e5bd629ea02bc42c7f2503198f` | Future experiment | Per-layer KV/storage policies, adaptive speculative control and graph policy may become future dimensions, but they are intentionally outside Riftline's current placement search space. |
| [xiersweet2648/sm120-llm-kernels](https://github.com/xiersweet2648/sm120-llm-kernels) | `3696eb26ee97993b7255a2b0bff1dd2562bde840` | Borrowed methodology | Performance calibration should be operation- and shape-specific. One global "5060 Ti versus 3060" speed ratio is not an adequate model. |

### parallelno NInfer TP2

The relevant result is not "use TP2." Riftline's reference topology benefits from coarse layer ownership precisely because constant collectives are expensive. The mailbox work is interesting for small synchronization-sensitive exchanges, but the first optimization target is eliminating traffic that should not exist — for example, the current redundant hidden-state return around endpoint execution.

### TurboPrefill

TurboPrefill motivated a separate future **heterogeneous layer-pipeline prefill** direction. It should not be mixed into placement feasibility or initial automatic selection because it changes execution overlap, buffer lifetimes and the performance model itself.

### sm120-llm-kernels

Its benchmarking methodology reinforced that decode, prefill, attention, recurrent/GDN work, projections and `lm_head` can sit in different performance regimes. Planned Riftline calibration therefore needs device-, operation-, phase- and shape-specific measurements rather than a single scalar device ratio.

### multi-turboquant and ninfer-fusion

Both were useful mainly as design-boundary references. `multi-turboquant` reinforced provenance and capability gating while also illustrating why generic BPP/overhead estimates are unsuitable for byte-exact admission. `ninfer-fusion` contains interesting future policy dimensions, but Riftline's first placement candidate remains intentionally small: contiguous layer boundary plus endpoint owner.

## Additional NInfer variants inspected

Two additional public NInfer variants were useful implementation/compatibility references during development:

- **[giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)** — consumer Ampere multi-GPU and no-P2P/TP behavior.
- **[mr-september/ninfer-2080ti-22g](https://github.com/mr-september/ninfer-2080ti-22g)** — another consumer-GPU NInfer adaptation used as a portability/reference build.

They were supporting references rather than the primary source of Riftline's placement-planner design.

## Comparative runtimes

The following runtimes were used as comparative evidence. They are **not** Riftline source dependencies and are not represented as code lineage.

| Runtime | Reference | What the comparison informed |
|---|---|---|
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Layer-split behavior, manual split trade-offs, long-context capacity and the cost of heterogeneous execution over a weak interconnect. |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) | TP2 throughput and KV-capacity limits on the same broad class of workload. |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) | TP2 behavior and transient-memory failure modes. |
| Lucebox | [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) | Alternative long-context/speculative behavior and the effect of large full-context activation buffers. |

The comparative-runtime work helped separate topology effects from implementation effects. A result from another runtime is evidence about a design trade-off on the tested system; it is not automatically a Riftline performance prediction.

## Design conclusions carried into Riftline

The combined research led to several durable rules:

1. **Unequal GPUs are not a pooled VRAM total.** Every physical device must satisfy its own memory ledger.
2. **Placement changes both intercept and context-growth slope.** KV growth can differ by owner because full-attention ownership differs.
3. **Hard feasibility and performance ranking are separate planes.** Empirical calibration may rank feasible candidates but cannot make an infeasible candidate legal.
4. **Persistent storage and transient peaks need different accounting semantics.** Blind summation is as dangerous as blind pooling.
5. **Prefer coarse ownership on weak interconnects.** Avoid frequent synchronization when a layer boundary can keep work local.
6. **Remove unnecessary transfers before optimizing transport.** Mailbox-style transport is a later experiment, not a substitute for a cleaner dataflow.
7. **Calibration must be shape-aware.** Prefill, decode, GDN, attention, projections, endpoint work and transfer costs should not collapse to one device speed ratio.
8. **Future pipeline prefill is a distinct execution mode.** It should receive its own buffers, synchronization model, calibration and validation rather than silently changing the placement predictor.

## Ideas deliberately not adopted as current behavior

Researching a project is not the same as importing its architecture. Riftline currently does **not** claim:

- arbitrary per-layer placement;
- symmetric tensor parallelism as the preferred mixed-GPU execution mode;
- mailbox transport in the endpoint path;
- heterogeneous diagonal/pipelined prefill;
- per-layer KV-format or cold-tier selection;
- empirical/fixed-overhead formulas as hard memory admission;
- a global 5060 Ti/3060 performance ratio;
- automatic placement selection.

These boundaries are intentional so that the current source remains auditable: exact feasibility first, calibration second, performance modelling third, automatic selection last.

## Citation and provenance policy

Public documentation distinguishes three different obligations:

1. **Inherited code and third-party components** keep their required copyright, license and notice material.
2. **Research influences** are named and linked here and in [`../ACKNOWLEDGEMENTS.md`](../ACKNOWLEDGEMENTS.md), with the pinned research identity recorded where the project preserved one.
3. **Comparative runtimes** are cited as experimental references without implying code reuse or endorsement.

Model- and artifact-specific provenance remains under `model-cards/`. Paper-level citations retained by inherited technical documentation remain attached to the mechanisms they describe rather than being duplicated into one oversized bibliography.

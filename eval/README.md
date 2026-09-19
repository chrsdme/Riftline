# Evaluation tooling

The `eval/` tree contains reusable evaluation code and configuration inherited from the NInfer lineage. Riftline keeps the reusable framework and focused probes, but intentionally does **not** publish raw evaluation outputs or machine-specific orchestration scripts from the development workspace.

## Included

- `ninfer_eval/` — evaluation coordinator, configuration parsing, backend adapters, result handling, logging, progress reporting, and secret redaction
- `configs/` — representative evaluation configurations for supported NInfer/Qwen targets
- `mtp_reasoning_probe.py` — focused speculative/reasoning probe
- `tp2_concurrency_smoke.py` — TP2 concurrency smoke utility
- `tp2_needle_throughput_probe.py` — needle/throughput probe
- `tests/` — unit tests for the evaluation framework
- `requirements.txt` — Python dependencies for the evaluation tooling

## Intentionally not included

The public Riftline source snapshot omits:

- raw `eval/results/` output trees;
- development-machine runner scripts that encoded local process, power, corpus, or model paths;
- model weights, `.ninfer` artifacts, corpora, and caches.

Some inherited configurations therefore contain neutral `/path/to/...` placeholders. Replace those paths with local resources before use.

## Secrets

Evaluation targets may reference an **environment-variable name** for an API key. The value is resolved from the environment at runtime and is not stored in the configuration. `ninfer_eval.secrets` also provides redaction support for emitted text.

## Public benchmark claims

The curated Riftline benchmark claims are in [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md). Inherited NInfer performance notes may describe broader historical campaigns, but those are not automatically Riftline qualification claims.

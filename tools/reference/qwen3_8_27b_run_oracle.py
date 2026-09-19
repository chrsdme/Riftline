"""Minimal CPU reference driver for the qwen3.8-27b oracle comparison (B1.3/B1.4).

Bypasses tools.reference.qwen3_6_27b.cli's Frontend/AutoProcessor path (which
requires pillow+torchvision purely to construct a vision-capable image
processor, even for a text-only prompt -- not installed here per the task's
"install nothing large" rule). RefModel takes an explicit token-ID list and
does not touch Frontend at all, so this talks to it directly.

Ground-truth prompt token IDs (13 tokens) were read from the engine itself
via Engine::prepare(prompt_from_text("Hello", false)).debug_token_ids(),
printed by tests/targets/qwen3_6_27b/test_engine_layer_real.cpp's added
PROMPT_IDS line -- not re-derived from a reconstructed tokenizer/template.
"""

from __future__ import annotations

import argparse
import sys

import torch

from tools.reference.qwen3_8_27b_oracle import patch_for_qwen38

patch_for_qwen38()

from tools.reference.qwen3_6_27b.model import RefModel  # noqa: E402
from tools.reference.qwen3_6.common.tap import FileTap  # noqa: E402

# From b13_run_with_promptids.log PROMPT_IDS line (engine ground truth for "Hello", thinking off).
PROMPT_IDS = [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
# Free-run tp.tokens / layer.tokens agree post-race-fix: 9419 0 2500 628.
FORCED_ROUND_TOKENS = [9419, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump", required=True, help="activation dump directory")
    parser.add_argument("--dump-level", choices=("layer", "op"), default="layer")
    parser.add_argument("--prefill-chunk", type=int, default=1)
    args = parser.parse_args()

    model = RefModel(args.weights, device=args.device, prefill_chunk=args.prefill_chunk)

    # Round 0: prompt only (13 tokens). Round 1: + forced[0]. Round 2: + forced[0],forced[1].
    # Each round is a fresh prefill (fresh model.prepare each time), matching the engine test's
    # fork_probe idiom (fresh, from-scratch, one-output-token prefill over a hand-built context).
    contexts = [
        list(PROMPT_IDS),
        list(PROMPT_IDS) + FORCED_ROUND_TOKENS[:1],
        list(PROMPT_IDS) + FORCED_ROUND_TOKENS[:2],
    ]
    for round_idx, ids in enumerate(contexts):
        tap = FileTap(f"{args.dump}/round{round_idx}", level=args.dump_level)
        model.prepare(len(ids) + 1)
        target = model.prefill(ids, tap=tap)
        tap.close(round=round_idx, context_len=len(ids), model_id="qwen3.8-27b")
        print(f"ROUND {round_idx} context_len={len(ids)} argmax_token={target}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

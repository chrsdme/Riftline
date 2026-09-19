"""Run the Qwen3.6-27B reference CLI against a qwen3.8-27b artifact.

The 3.8 artifact is structurally identical to the 3.6 reference contract
(same 1124 object names, shapes, layouts; same hardcoded ModelConfig dims
verified against layer/mlp/gdn tensor shapes) except:

  - ArtifactIdentity.model_id is "qwen3.8-27b" instead of "qwen3.6-27b".
  - text/token_embedding and text/output_head are stored as W8G32_F16S
    instead of Q6G64_F16S (same shape/layout; byte count matches the
    generic row_split_geometry() formula for W8G32, and the dequant path
    in weights.py is format-parameterized, not hardcoded to Q6).

This module monkeypatches those two expectations in
tools.reference.qwen3_6_27b.bindings before delegating to the unmodified
reference CLI. It does not touch bindings.py itself.
"""

from __future__ import annotations

import dataclasses

from tools.reference.qwen3_6_27b import bindings as _bindings

_ALT_MODEL_ID = "qwen3.8-27b"
_ALT_FORMAT_TENSORS = {"text/token_embedding", "text/output_head"}
_ALT_FORMAT = "W8G32_F16S"


def patch_for_qwen38() -> None:
    _bindings.MODEL_ID = _ALT_MODEL_ID
    _bindings._OBJECT_CONTRACT = tuple(
        dataclasses.replace(obj, format=_ALT_FORMAT)
        if isinstance(obj, _bindings._ExpectedTensor) and obj.name in _ALT_FORMAT_TENSORS
        else obj
        for obj in _bindings._OBJECT_CONTRACT
    )


patch_for_qwen38()

from tools.reference.qwen3_6_27b.cli import main  # noqa: E402  (must patch before import-time use)

if __name__ == "__main__":
    main()

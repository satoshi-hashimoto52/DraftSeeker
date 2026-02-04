from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from .config import SAM_CHECKPOINT, SAM_MODEL_TYPE
from .sam_device import get_sam_device


if TYPE_CHECKING:
    from segment_anything import SamPredictor

_predictor: Optional["SamPredictor"] = None


def get_sam_predictor() -> "SamPredictor":
    global _predictor
    if _predictor is not None:
        return _predictor

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:
        raise RuntimeError("segment-anything is not installed") from exc

    checkpoint = os.getenv("SAM_CHECKPOINT", SAM_CHECKPOINT)
    model_type = os.getenv("SAM_MODEL_TYPE", SAM_MODEL_TYPE)
    if not checkpoint:
        raise RuntimeError("SAM_CHECKPOINT is not set")

    device = get_sam_device()
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    _predictor = SamPredictor(sam)
    return _predictor

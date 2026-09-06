"""SAM bbox-prompt mask generation for the experimental demo."""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.recognition_pipeline import BBox


logger = logging.getLogger(__name__)


class SegmentationModelNotConfigured(RuntimeError):
    pass


def _checkpoint_path() -> Path:
    configured = os.getenv("SEGMENTATION_CHECKPOINT_PATH", "").strip()
    if configured:
        path = Path(configured)
        if not path.exists():
            raise SegmentationModelNotConfigured(f"checkpoint not found: {path}")
        return path

    uri = os.getenv("SEGMENTATION_CHECKPOINT_URI", "").strip()
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, object_name = uri[5:].split("/", 1)
        directory = Path(tempfile.gettempdir()) / "yujian-segmentation"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / Path(object_name).name
        if not path.exists():
            storage.Client().bucket(bucket_name).blob(object_name).download_to_filename(str(path), timeout=600)
        return path

    raise SegmentationModelNotConfigured(
        "set SEGMENTATION_CHECKPOINT_PATH or SEGMENTATION_CHECKPOINT_URI for the SAM demo"
    )


@lru_cache(maxsize=1)
def _load_predictor():
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as exc:
        raise SegmentationModelNotConfigured("segment-anything is not installed") from exc

    model_type = os.getenv("SEGMENTATION_MODEL_TYPE", "vit_b").strip()
    if model_type not in sam_model_registry:
        raise SegmentationModelNotConfigured(f"unsupported SAM model type: {model_type}")
    model = sam_model_registry[model_type](checkpoint=str(_checkpoint_path()))
    model.eval()
    return SamPredictor(model)


def initialize_segmentation_model() -> bool:
    """Warm-load SAM when configured, without making startup dependent on it."""
    try:
        _load_predictor()
        model_name = os.getenv("SEGMENTATION_MODEL_TYPE", "vit_b").strip().upper()
        logger.info("Fish Segmentation Model Ready model=SAM_%s checkpoint=loaded", model_name)
        return True
    except SegmentationModelNotConfigured as exc:
        logger.warning("Segmentation unavailable: %s", exc)
        return False
    except Exception:
        logger.exception("Segmentation unavailable: SAM checkpoint load failed")
        return False

def generate_mask(image: Image.Image, bbox: BBox) -> np.ndarray:
    predictor = _load_predictor()
    rgb = np.asarray(image.convert("RGB"))
    predictor.set_image(rgb)
    left, top, right, bottom = _pixel_box(bbox, image.width, image.height)
    masks, _, _ = predictor.predict(
        box=np.asarray([left, top, right, bottom], dtype=np.float32),
        multimask_output=False,
    )
    mask = np.asarray(masks[0], dtype=bool)
    if mask.shape != (image.height, image.width):
        raise RuntimeError(
            f"SAM mask shape {mask.shape} does not match image {(image.height, image.width)}"
        )
    return mask


def _pixel_box(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    normalized = box.normalized()
    left = max(0, min(width - 1, int(np.floor(normalized.x1 * width))))
    top = max(0, min(height - 1, int(np.floor(normalized.y1 * height))))
    right = max(left + 1, min(width, int(np.ceil(normalized.x2 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(normalized.y2 * height))))
    return left, top, right, bottom

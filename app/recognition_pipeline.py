from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable


class PipelineStatus(str, Enum):
    READY = "ready"
    NO_FISH = "no_fish"
    UNCERTAIN = "uncertain"
    MULTIPLE_FISH = "multiple_fish"
    INCOMPLETE_FISH = "incomplete_fish"
    FISH_TOO_SMALL = "fish_too_small"


@dataclass(frozen=True)
class BBox:
    """Normalized [0, 1] axis-aligned box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "BBox":
        left = max(0.0, min(1.0, min(self.x1, self.x2)))
        top = max(0.0, min(1.0, min(self.y1, self.y2)))
        right = max(0.0, min(1.0, max(self.x1, self.x2)))
        bottom = max(0.0, min(1.0, max(self.y1, self.y2)))
        return BBox(left, top, right, bottom)

    @property
    def width(self) -> float:
        b = self.normalized()
        return max(0.0, b.x2 - b.x1)

    @property
    def height(self) -> float:
        b = self.normalized()
        return max(0.0, b.y2 - b.y1)

    @property
    def area_ratio(self) -> float:
        return self.width * self.height

    def touches_edge(self, margin: float) -> bool:
        b = self.normalized()
        return b.x1 <= margin or b.y1 <= margin or b.x2 >= 1.0 - margin or b.y2 >= 1.0 - margin

    def expand(self, ratio: float) -> "BBox":
        b = self.normalized()
        dx = b.width * ratio
        dy = b.height * ratio
        return BBox(
            max(0.0, b.x1 - dx),
            max(0.0, b.y1 - dy),
            min(1.0, b.x2 + dx),
            min(1.0, b.y2 + dy),
        )


@dataclass(frozen=True)
class Detection:
    confidence: float
    box: BBox
    class_name: str = "fish"

    @property
    def area_ratio(self) -> float:
        return self.box.area_ratio


@dataclass(frozen=True)
class PipelineAssessment:
    status: PipelineStatus
    primary: Detection | None
    crop_box: BBox | None
    strong_detections: tuple[Detection, ...]
    weak_detections: tuple[Detection, ...]
    reason: str


@lru_cache(maxsize=1)
def load_contract(path: str | Path | None = None) -> dict:
    contract_path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "recognition_pipeline_v1.json"
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _rank_score(detection: Detection) -> float:
    # Prefer confident, visually dominant fish without allowing area alone to overwhelm confidence.
    return max(0.0, detection.confidence) * math.sqrt(max(0.0, detection.area_ratio))


def select_primary(detections: Iterable[Detection]) -> Detection | None:
    items = list(detections)
    if not items:
        return None
    return max(items, key=_rank_score)


def assess_detections(detections: Iterable[Detection], contract: dict | None = None) -> PipelineAssessment:
    contract = contract or load_contract()
    detector_cfg = contract["detector"]
    gate_cfg = contract["quality_gate"]
    crop_cfg = contract["crop"]

    strong_threshold = float(detector_cfg["strong_confidence"])
    weak_threshold = float(detector_cfg["weak_confidence"])
    min_area = float(gate_cfg["min_primary_area_ratio"])
    edge_margin = float(gate_cfg["incomplete_edge_margin_ratio"])
    expand_ratio = float(crop_cfg["expand_ratio"])
    fish_class = str(detector_cfg.get("class_name") or "fish").lower()

    fish = [
        Detection(float(d.confidence), d.box.normalized(), d.class_name)
        for d in detections
        if str(d.class_name).lower() == fish_class and d.box.area_ratio > 0.0
    ]
    strong = tuple(sorted((d for d in fish if d.confidence >= strong_threshold), key=_rank_score, reverse=True))
    weak = tuple(sorted((d for d in fish if weak_threshold <= d.confidence < strong_threshold), key=_rank_score, reverse=True))

    if not strong:
        if weak:
            return PipelineAssessment(
                status=PipelineStatus.UNCERTAIN,
                primary=weak[0],
                crop_box=None,
                strong_detections=strong,
                weak_detections=weak,
                reason="weak_fish_detection_only",
            )
        return PipelineAssessment(
            status=PipelineStatus.NO_FISH,
            primary=None,
            crop_box=None,
            strong_detections=strong,
            weak_detections=weak,
            reason="no_fish_detection_above_weak_threshold",
        )

    primary = select_primary(strong)
    assert primary is not None

    if len(strong) >= 2:
        return PipelineAssessment(
            status=PipelineStatus.MULTIPLE_FISH,
            primary=primary,
            crop_box=None,
            strong_detections=strong,
            weak_detections=weak,
            reason="multiple_strong_fish_detections",
        )

    if primary.box.touches_edge(edge_margin):
        return PipelineAssessment(
            status=PipelineStatus.INCOMPLETE_FISH,
            primary=primary,
            crop_box=None,
            strong_detections=strong,
            weak_detections=weak,
            reason="primary_fish_bbox_touches_image_edge",
        )

    if primary.area_ratio < min_area:
        return PipelineAssessment(
            status=PipelineStatus.FISH_TOO_SMALL,
            primary=primary,
            crop_box=None,
            strong_detections=strong,
            weak_detections=weak,
            reason="primary_fish_area_below_minimum",
        )

    return PipelineAssessment(
        status=PipelineStatus.READY,
        primary=primary,
        crop_box=primary.box.expand(expand_ratio),
        strong_detections=strong,
        weak_detections=weak,
        reason="single_complete_fish_ready_for_classifier",
    )


def crop_box_pixels(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    b = box.normalized()
    # Explicit floor/ceil contract. Android must mirror this exactly.
    left = max(0, min(width - 1, math.floor(b.x1 * width)))
    top = max(0, min(height - 1, math.floor(b.y1 * height)))
    right = max(left + 1, min(width, math.ceil(b.x2 * width)))
    bottom = max(top + 1, min(height, math.ceil(b.y2 * height)))
    return left, top, right, bottom

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
    INVALID_BBOX = "invalid_bbox"
    EMPTY_CROP = "empty_crop"
    UNCERTAIN = "uncertain"
    MULTIPLE_FISH = "multiple_fish"
    # Kept for wire/backward compatibility with older reports.  Source-edge
    # proximity is no longer emitted as this hard-blocking status.
    INCOMPLETE_FISH = "incomplete_fish"
    FISH_TOO_SMALL = "fish_too_small"


SOURCE_EDGE_NEAR = "SOURCE_EDGE_NEAR"
CROP_EDGE_NEAR = "CROP_EDGE_NEAR"


@dataclass(frozen=True)
class BoundaryCheck:
    """Non-blocking source/crop boundary diagnostics for one primary fish."""

    source_edge_near: bool = False
    crop_edge_near: bool = False
    reason: str | None = None
    hard_block: bool = False


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
    boundary: BoundaryCheck = BoundaryCheck()


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


def _has_positive_geometry(box: BBox) -> bool:
    """Reject an actually inverted/empty detector box before normalization.

    ``BBox.normalized`` intentionally sorts coordinates for display and crop
    compatibility.  A detector result with x2 <= x1 or y2 <= y1 is different:
    it has no valid image region and must remain a hard gate failure.
    """

    return (
        all(math.isfinite(float(value)) for value in (box.x1, box.y1, box.x2, box.y2))
        and box.x2 > box.x1
        and box.y2 > box.y1
    )


def _boundary_check(box: BBox, edge_margin: float, expand_ratio: float) -> BoundaryCheck:
    source = box.normalized()
    crop = box.expand(expand_ratio)
    source_edge_near = source.touches_edge(edge_margin)
    crop_edge_near = crop.touches_edge(0.0)
    reason = SOURCE_EDGE_NEAR if source_edge_near else (CROP_EDGE_NEAR if crop_edge_near else None)
    return BoundaryCheck(
        source_edge_near=source_edge_near,
        crop_edge_near=crop_edge_near,
        reason=reason,
        hard_block=False,
    )


def boundary_debug_payload(
    assessment: PipelineAssessment,
    width: int,
    height: int,
    contract: dict | None = None,
) -> dict:
    """Serialize additive boundary diagnostics for Debug/API responses.

    Distances are measured from the detector bbox to the original source
    image edges.  ``crop_edge_near`` describes the expanded crop touching the
    source raster edge; neither signal is a classifier hard blocker.
    """

    if width <= 0 or height <= 0 or assessment.primary is None:
        return {
            "source_edge_near": False,
            "crop_edge_near": False,
            "crop_touch_source_edge": False,
            "hard_block": False,
            "reason": None,
            "distances_px": {"top": None, "bottom": None, "left": None, "right": None},
        }

    contract = contract or load_contract()
    detector_cfg = contract["quality_gate"]
    crop_cfg = contract["crop"]
    boundary = assessment.boundary
    source = assessment.primary.box.normalized()
    crop = assessment.crop_box or source.expand(float(crop_cfg["expand_ratio"]))
    distances = {
        "top": int(round(source.y1 * height)),
        "bottom": int(round((1.0 - source.y2) * height)),
        "left": int(round(source.x1 * width)),
        "right": int(round((1.0 - source.x2) * width)),
    }
    # Recompute from the serialized source geometry as a defensive check for
    # callers constructing an assessment manually in tests.
    source_edge_near = source.touches_edge(float(detector_cfg["incomplete_edge_margin_ratio"]))
    crop_edge_near = crop.touches_edge(0.0)
    reason = boundary.reason or (
        SOURCE_EDGE_NEAR if source_edge_near else (CROP_EDGE_NEAR if crop_edge_near else None)
    )
    return {
        "source_edge_near": source_edge_near,
        "crop_edge_near": crop_edge_near,
        "crop_touch_source_edge": crop_edge_near,
        "hard_block": False,
        "reason": reason,
        "distances_px": distances,
    }


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

    fish: list[Detection] = []
    invalid_bbox_found = False
    for detection in detections:
        if str(detection.class_name).lower() != fish_class:
            continue
        if not _has_positive_geometry(detection.box):
            invalid_bbox_found = True
            continue
        normalized = detection.box.normalized()
        if normalized.area_ratio <= 0.0:
            invalid_bbox_found = True
            continue
        fish.append(Detection(float(detection.confidence), normalized, detection.class_name))
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
        if invalid_bbox_found:
            return PipelineAssessment(
                status=PipelineStatus.INVALID_BBOX,
                primary=None,
                crop_box=None,
                strong_detections=strong,
                weak_detections=weak,
                reason="invalid_bbox_geometry",
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

    boundary = _boundary_check(primary.box, edge_margin, expand_ratio)

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
        reason=boundary.reason or "single_complete_fish_ready_for_classifier",
        boundary=boundary,
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

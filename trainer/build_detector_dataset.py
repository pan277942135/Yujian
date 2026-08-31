from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from google.cloud import storage
from PIL import Image, ImageOps
from sqlalchemy import select

from app.db import SessionLocal
from app.models import ImageAsset
from app.presence import FishPresenceResult, _is_fish_term, effective_status
from app.recognition_pipeline import BBox, load_contract


@dataclass
class Candidate:
    image: ImageAsset
    status: str
    boxes: list[tuple[float, BBox]]


def _parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def _saved_evidence(row: FishPresenceResult) -> dict:
    try:
        return json.loads(row.evidence_json or "{}")
    except json.JSONDecodeError:
        return {}


def _vertices_to_box(vertices: list[dict]) -> BBox | None:
    if not vertices:
        return None
    xs = [float(v.get("x", 0.0) or 0.0) for v in vertices]
    ys = [float(v.get("y", 0.0) or 0.0) for v in vertices]
    if not xs or not ys:
        return None
    box = BBox(min(xs), min(ys), max(xs), max(ys)).normalized()
    return box if box.area_ratio > 0.002 else None


def _split_key(image: ImageAsset) -> str:
    digest = hashlib.sha256(f"{image.batch_id}:{image.image_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:2], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _download(client: storage.Client, uri: str, target: Path) -> None:
    bucket_name, object_name = _parse_gs(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(object_name).download_to_filename(str(target))


def _collect() -> list[Candidate]:
    contract = load_contract()
    strong_threshold = float(contract["detector"]["strong_confidence"])
    out: list[Candidate] = []
    with SessionLocal() as db:
        rows = db.execute(
            select(ImageAsset, FishPresenceResult)
            .join(FishPresenceResult, FishPresenceResult.image_asset_id == ImageAsset.id)
            .order_by(ImageAsset.id)
        ).all()
        for image, presence in rows:
            status = effective_status(presence)
            evidence = _saved_evidence(presence)
            objects = evidence.get("objects") or []
            boxes: list[tuple[float, BBox]] = []
            for item in objects:
                if not _is_fish_term(item.get("name")):
                    continue
                score = float(item.get("score", 0.0) or 0.0)
                if score < strong_threshold:
                    continue
                box = _vertices_to_box(item.get("vertices") or [])
                if box is not None:
                    boxes.append((score, box))
            if status in {"single_fish", "multi_fish"} and boxes:
                out.append(Candidate(image=image, status=status, boxes=boxes))
            elif status == "no_fish":
                out.append(Candidate(image=image, status=status, boxes=[]))
    return out


def main() -> None:
    output_root = Path(os.environ.get("DETECTOR_DATASET_ROOT", "/tmp/yujian-detector-dataset"))
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = _collect()
    positives = [x for x in candidates if x.boxes]
    negatives = [x for x in candidates if not x.boxes]

    # Keep enough true negatives to teach the detector to return no boxes, without swamping positives.
    max_negatives = max(1, int(len(positives) * 0.35)) if positives else len(negatives)
    negatives = sorted(negatives, key=lambda x: hashlib.sha256(x.image.image_id.encode()).hexdigest())[:max_negatives]
    selected = positives + negatives

    client = storage.Client()
    contract = load_contract()
    edge_margin = float(contract["quality_gate"]["incomplete_edge_margin_ratio"])

    coco = {
        split: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "fish"}]}
        for split in ("train", "val", "test")
    }
    manifest: list[dict] = []
    annotation_id = 1
    image_ids = {"train": 1, "val": 1, "test": 1}
    incomplete_candidates = 0

    for candidate in selected:
        split = _split_key(candidate.image)
        suffix = Path(candidate.image.file_name or candidate.image.object_name).suffix.lower() or ".jpg"
        local_name = f"{candidate.image.id:08d}{suffix}"
        local_path = output_root / "images" / split / local_name
        try:
            _download(client, candidate.image.gcs_uri, local_path)
            with Image.open(local_path) as source:
                source = ImageOps.exif_transpose(source)
                width, height = source.size
        except Exception as exc:
            manifest.append({
                "image_asset_id": candidate.image.id,
                "status": candidate.status,
                "gcs_uri": candidate.image.gcs_uri,
                "skipped": True,
                "error": str(exc),
            })
            continue

        image_id = image_ids[split]
        image_ids[split] += 1
        coco[split]["images"].append({"id": image_id, "file_name": local_name, "width": width, "height": height})
        saved_boxes = []
        for score, box in candidate.boxes:
            b = box.normalized()
            x = b.x1 * width
            y = b.y1 * height
            w = b.width * width
            h = b.height * height
            incomplete = b.touches_edge(edge_margin)
            incomplete_candidates += int(incomplete)
            coco[split]["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                "area": round(w * h, 3),
                "iscrowd": 0,
            })
            annotation_id += 1
            saved_boxes.append({
                "confidence": round(score, 6),
                "bbox_normalized": [b.x1, b.y1, b.x2, b.y2],
                "touches_edge": incomplete,
            })

        manifest.append({
            "image_asset_id": candidate.image.id,
            "batch_id": candidate.image.batch_id,
            "image_id": candidate.image.image_id,
            "gcs_uri": candidate.image.gcs_uri,
            "split": split,
            "presence_status": candidate.status,
            "negative": not bool(candidate.boxes),
            "boxes": saved_boxes,
            "width": width,
            "height": height,
        })

    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps(coco[split], ensure_ascii=False), encoding="utf-8"
        )

    report = {
        "source": "FishPresenceResult/google-vision-presence-v0.4",
        "selected": len(selected),
        "positives": len(positives),
        "negatives": len(negatives),
        "incomplete_candidates": incomplete_candidates,
        "splits": {
            split: {
                "images": len(coco[split]["images"]),
                "annotations": len(coco[split]["annotations"]),
            }
            for split in ("train", "val", "test")
        },
        "contract_version": contract["contract_version"],
    }
    (output_root / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "bootstrap_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

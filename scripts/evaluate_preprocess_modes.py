from __future__ import annotations

import argparse
import csv
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from google.cloud import storage
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Padding uses the ImageNet mean so padded pixels become approximately zero after normalization.
IMAGENET_MEAN_RGB = tuple(round(x * 255) for x in IMAGENET_MEAN)


def parse_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://") or "/" not in uri[5:]:
        raise ValueError(f"invalid GCS URI: {uri}")
    return tuple(uri[5:].split("/", 1))  # type: ignore[return-value]


def download_bytes(client: storage.Client, uri: str) -> bytes:
    bucket, obj = parse_gs(uri)
    return client.bucket(bucket).blob(obj).download_as_bytes(timeout=120)


def download_text(client: storage.Client, uri: str) -> str:
    return download_bytes(client, uri).decode("utf-8")


class WholeImageLetterbox:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        contained = ImageOps.contain(source, (self.size, self.size), method=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), IMAGENET_MEAN_RGB)
        x = (self.size - contained.width) // 2
        y = (self.size - contained.height) // 2
        canvas.paste(contained, (x, y))
        return canvas


def build_transforms(size: int):
    normalize = transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
    center = transforms.Compose(
        [
            transforms.Resize(int(size * 1.15)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    letterbox = transforms.Compose(
        [
            WholeImageLetterbox(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return center, letterbox


@dataclass
class Running:
    count: int = 0
    top1: int = 0
    top3: int = 0
    confidence_sum: float = 0.0

    def add(self, logits: torch.Tensor, truth: int) -> None:
        probs = torch.softmax(logits, dim=1)[0]
        k = min(3, probs.shape[0])
        top = torch.topk(probs, k=k)
        indices = [int(x) for x in top.indices.tolist()]
        self.count += 1
        self.top1 += int(indices[0] == truth)
        self.top3 += int(truth in indices)
        self.confidence_sum += float(top.values[0].item())

    def report(self) -> dict:
        if not self.count:
            return {"count": 0, "top1_accuracy": 0.0, "top3_accuracy": 0.0, "mean_top1_confidence": 0.0}
        return {
            "count": self.count,
            "top1_accuracy": round(self.top1 / self.count, 6),
            "top3_accuracy": round(self.top3 / self.count, 6),
            "mean_top1_confidence": round(self.confidence_sum / self.count, 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    classes = sorted(list(class_map.get("classes") or []), key=lambda x: int(x.get("class_index", 0)))
    image_size = int((metrics.get("params") or {}).get("image_size") or 224)
    manifest_uri = str(metrics.get("dataset_manifest_uri") or "").strip()
    if not manifest_uri:
        raise RuntimeError("metrics.json does not contain dataset_manifest_uri")

    model = torch.jit.load(args.torchscript, map_location="cpu")
    model.eval()
    probe = torch.randn(1, 3, image_size, image_size)
    with torch.inference_mode():
        output = model(probe)
    if int(output.shape[-1]) != len(classes):
        raise RuntimeError(f"model output {output.shape[-1]} != class map {len(classes)}")

    client = storage.Client()
    rows = list(csv.DictReader(io.StringIO(download_text(client, manifest_uri))))
    candidates = [r for r in rows if (r.get("split") or "").strip() == "test"]
    split = "test"
    if not candidates:
        candidates = [r for r in rows if (r.get("split") or "").strip() == "val"]
        split = "val"
    if not candidates:
        raise RuntimeError("dataset manifest has no test or val rows")

    random.Random(20260830).shuffle(candidates)
    candidates = candidates[: max(1, args.limit)]
    center_transform, letterbox_transform = build_transforms(image_size)
    center_stats = Running()
    letterbox_stats = Running()
    disagreements = 0
    evaluated = 0
    failures: list[dict] = []

    for row in candidates:
        uri = (row.get("gcs_uri") or "").strip()
        try:
            truth = int(row.get("class_index") or "")
            data = download_bytes(client, uri)
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                center_tensor = center_transform(image).unsqueeze(0)
                letterbox_tensor = letterbox_transform(image).unsqueeze(0)
            with torch.inference_mode():
                center_logits = model(center_tensor)
                letterbox_logits = model(letterbox_tensor)
            center_pred = int(center_logits.argmax(dim=1).item())
            letterbox_pred = int(letterbox_logits.argmax(dim=1).item())
            disagreements += int(center_pred != letterbox_pred)
            center_stats.add(center_logits, truth)
            letterbox_stats.add(letterbox_logits, truth)
            evaluated += 1
        except (ValueError, OSError, UnidentifiedImageError, RuntimeError) as exc:
            failures.append({"image_id": row.get("image_id"), "gcs_uri": uri, "error": str(exc)})

    center_report = center_stats.report()
    letterbox_report = letterbox_stats.report()
    result = {
        "model_version": metrics.get("model_version"),
        "dataset_version": metrics.get("dataset_version"),
        "evaluated_split": split,
        "requested_limit": args.limit,
        "evaluated": evaluated,
        "failed": len(failures),
        "image_size": image_size,
        "num_classes": len(classes),
        "center_crop": center_report,
        "whole_image_letterbox": letterbox_report,
        "top1_accuracy_delta_letterbox_minus_center": round(
            letterbox_report["top1_accuracy"] - center_report["top1_accuracy"], 6
        ),
        "prediction_disagreement_rate": round(disagreements / evaluated, 6) if evaluated else 0.0,
        "letterbox_contract": {
            "resize": "preserve aspect ratio, fit entire image inside square",
            "padding_rgb": list(IMAGENET_MEAN_RGB),
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "cropping": "none",
        },
        "failures": failures[:20],
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

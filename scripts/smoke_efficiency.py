#!/usr/bin/env python3
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from PIL import Image, ImageDraw  # noqa: E402

from app.dedupe import ImageFingerprint, duplicate_distance, fingerprint_bytes  # noqa: E402
from app.presence import classify_presence  # noqa: E402


def make_image(crop=False, quality=90):
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 75, 255, 165), fill=(90, 130, 150), outline=(20, 30, 40), width=4)
    draw.polygon([(245, 120), (300, 80), (300, 160)], fill=(90, 130, 150))
    draw.ellipse((90, 102, 100, 112), fill="black")
    if crop:
        img = img.crop((35, 35, 305, 205)).resize((320, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def as_row(fp, image_id):
    import json
    return ImageFingerprint(
        image_asset_id=image_id,
        batch_id="TEST",
        sha256=fp["sha256"],
        phash_json=json.dumps(fp["phashes"]),
        dhash=fp["dhash"],
        crop_hash=fp["crop_hash"],
        histogram_json=json.dumps(fp["histogram"]),
        width=fp["width"],
        height=fp["height"],
    )


def main():
    single = classify_presence(
        [{"name": "Fish", "score": 0.91, "vertices": [{"x": 0.1, "y": 0.2}, {"x": 0.8, "y": 0.7}]}],
        [{"name": "Fish", "score": 0.92}],
    )
    assert single["status"] == "single_fish", single

    multi = classify_presence(
        [
            {"name": "Fish", "score": 0.91, "vertices": [{"x": 0.1, "y": 0.2}, {"x": 0.4, "y": 0.5}]},
            {"name": "Fish", "score": 0.89, "vertices": [{"x": 0.5, "y": 0.3}, {"x": 0.9, "y": 0.7}]},
        ],
        [{"name": "Fish", "score": 0.94}],
    )
    assert multi["status"] == "multi_fish", multi

    no_fish = classify_presence([], [
        {"name": "Person", "score": 0.98},
        {"name": "Outdoor", "score": 0.93},
        {"name": "Grass", "score": 0.91},
        {"name": "Sky", "score": 0.88},
    ])
    assert no_fish["status"] == "no_fish", no_fish

    original = fingerprint_bytes(make_image(crop=False, quality=92))
    same_visual = fingerprint_bytes(make_image(crop=False, quality=75))
    cropped = fingerprint_bytes(make_image(crop=True, quality=88))
    kind1, _ = duplicate_distance(as_row(original, 1), as_row(same_visual, 2))
    kind2, _ = duplicate_distance(as_row(original, 1), as_row(cropped, 3))
    assert kind1 in {"near", "exact"}, kind1
    assert kind2 in {"near", "exact"}, kind2

    print("P0 efficiency smoke test: OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("REGISTRY_DB_URL", "sqlite:///:memory:")

from PIL import Image, ImageDraw  # noqa: E402

from app.dedupe import (  # noqa: E402
    ImageFingerprint,
    _crop_match,
    _hamming_hex,
    _hist_similarity,
    _min_phash_distance,
    duplicate_distance,
    fingerprint_bytes,
)
from app.presence import classify_presence  # noqa: E402


def encode(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def make_image(crop=False, quality=90):
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((60, 75, 255, 165), fill=(90, 130, 150), outline=(20, 30, 40), width=4)
    draw.polygon([(245, 120), (300, 80), (300, 160)], fill=(90, 130, 150))
    draw.ellipse((90, 102, 100, 112), fill="black")
    if crop:
        img = img.crop((35, 35, 305, 205)).resize((320, 200))
    return encode(img, quality)


def make_other_image(quality=90):
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((115, 35, 205, 210), radius=20, fill=(90, 130, 150), outline=(20, 30, 40), width=4)
    draw.rectangle((130, 60, 190, 85), fill=(20, 30, 40))
    draw.rectangle((130, 160, 190, 185), fill=(20, 30, 40))
    return encode(img, quality)


def as_row(fp, image_id):
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


def metrics(a, b):
    return {
        "p_dist": _min_phash_distance(a["phashes"], b["phashes"]),
        "d_dist": _hamming_hex(a["dhash"], b["dhash"]),
        "hist": round(_hist_similarity(a["histogram"], b["histogram"]), 6),
        "crop": _crop_match(a["crop_hash"], b["crop_hash"]),
    }


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

    rod_scene = classify_presence([], [
        {"name": "Fishing", "score": 0.97},
        {"name": "Fishing rod", "score": 0.94},
        {"name": "Water", "score": 0.91},
    ])
    assert rod_scene["status"] == "no_fish", rod_scene
    assert rod_scene["routing_reason"] == "fishing_scene_without_fish", rod_scene
    assert rod_scene["fish_score"] == 0.0, rod_scene

    # A common user-reported failure: person + scenery with no visible fish.
    person_scenery = classify_presence([], [
        {"name": "Person", "score": 0.94},
        {"name": "Outdoor", "score": 0.72},
        {"name": "Water", "score": 0.68},
    ])
    assert person_scenery["status"] == "no_fish", person_scenery

    # Two solid landscape signals are enough when there is absolutely no fish evidence.
    sparse_landscape = classify_presence([], [
        {"name": "Natural landscape", "score": 0.78},
        {"name": "Body of water", "score": 0.73},
    ])
    assert sparse_landscape["status"] == "no_fish", sparse_landscape

    fishery_landscape = classify_presence([], [
        {"name": "Fishery", "score": 0.93},
        {"name": "Landscape", "score": 0.91},
        {"name": "Water", "score": 0.90},
        {"name": "Sky", "score": 0.89},
    ])
    assert fishery_landscape["status"] == "no_fish", fishery_landscape
    assert fishery_landscape["fish_score"] == 0.0, fishery_landscape

    fish_with_rod = classify_presence(
        [{"name": "Fish", "score": 0.82, "vertices": [{"x": 0.2, "y": 0.2}, {"x": 0.7, "y": 0.7}]}],
        [
            {"name": "Fishing", "score": 0.97},
            {"name": "Fishing rod", "score": 0.95},
            {"name": "Water", "score": 0.90},
        ],
    )
    assert fish_with_rod["status"] == "single_fish", fish_with_rod

    # Fish label-only evidence remains uncertain rather than being auto-rejected.
    fish_label_only = classify_presence([], [
        {"name": "Fish", "score": 0.81},
        {"name": "Water", "score": 0.84},
        {"name": "Outdoor", "score": 0.79},
    ])
    assert fish_label_only["status"] == "uncertain", fish_label_only

    original = fingerprint_bytes(make_image(crop=False, quality=92))
    same_visual = fingerprint_bytes(make_image(crop=False, quality=75))
    cropped = fingerprint_bytes(make_image(crop=True, quality=88))
    different = fingerprint_bytes(make_other_image(quality=90))

    print("reencode metrics", metrics(original, same_visual))
    print("crop metrics", metrics(original, cropped))
    print("negative metrics", metrics(original, different))

    kind1, _ = duplicate_distance(as_row(original, 1), as_row(same_visual, 2))
    kind2, _ = duplicate_distance(as_row(original, 1), as_row(cropped, 3))
    kind3, _ = duplicate_distance(as_row(original, 1), as_row(different, 4))
    assert kind1 in {"near", "exact"}, kind1
    assert kind2 in {"near", "exact"}, kind2
    assert kind3 is None, kind3

    print("P0 efficiency smoke test: OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chromium layout smoke for DET_FISH preview images; inference is simulated."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import socket
import sys
import threading
import time

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.detector_parity_api import _normalized_xywh, _overlay, _pixel_box
from app.recognition_pipeline import BBox, Detection


ARTIFACTS = ROOT / "test-results" / "det-fish-display"

app = FastAPI(docs_url=None, openapi_url=None)
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))


@app.get("/platform/model/det-fish-test", response_class=HTMLResponse)
def det_fish_test_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="platform/det_fish_test.html",
        context={"page_title": "DET_FISH 模型测试"},
    )


@app.get("/api/det-fish/models")
def simulated_model_list():
    return {
        "models": [
            {
                "model_version": "UI_SIMULATION_ONLY",
                "dataset_version": "display-fixture",
                "available": True,
                "is_configured_production_version": False,
                "onnx_sha256": None,
            }
        ]
    }


@app.post("/api/det-fish/test")
async def simulated_detection(image: UploadFile = File(...), model_version: str = Form(...)):
    if model_version != "UI_SIMULATION_ONLY":
        return {"detail": "This Chromium harness only provides a simulated response."}
    contents = await image.read()
    with Image.open(BytesIO(contents)) as uploaded:
        source = ImageOps.exif_transpose(uploaded).convert("RGB")
    detections = (
        Detection(0.91, BBox(0.12, 0.18, 0.76, 0.82)),
        Detection(0.74, BBox(0.54, 0.08, 0.91, 0.42)),
    )
    boxes = [
        {
            "rank": index + 1,
            "confidence": detection.confidence,
            "bbox_pixel": _pixel_box(detection.box, source.width, source.height),
            "bbox_pixel_input": _pixel_box(detection.box, source.width, source.height),
            "bbox_normalized_xywh": _normalized_xywh(detection.box),
            "bbox_area_ratio": detection.area_ratio,
        }
        for index, detection in enumerate(detections)
    ]
    overlay_data_url = _overlay(source, detections, detections[0])
    width, height = source.size
    source.close()
    return {
        "model_version": "UI_SIMULATION_ONLY (not a model inference)",
        "onnx_sha256": None,
        "input": {
            "oriented_width": width,
            "oriented_height": height,
            "width": width,
            "height": height,
        },
        "model_input": {
            "width": 416,
            "height": 416,
            "shape": None,
            "dtype": "simulated",
            "layout": "simulated",
            "resized_content_width": 416,
            "resized_content_height": 416,
            "letterbox_fill": 114,
        },
        "detector": {
            "detections": boxes,
            "assessment": "SIMULATED",
            "confidence": detections[0].confidence,
            "latency_ms": 0.0,
            "attempts": [
                {
                    "orientation_attempt": "SIMULATED",
                    "quality_status": "SIMULATED",
                    "quality_level": "SIMULATED",
                    "detection_count": len(detections),
                    "top_confidence": detections[0].confidence,
                }
            ],
            "selected_orientation": "SIMULATED",
            "selection_reason": "display smoke harness",
            "reason": "No production model was invoked",
        },
        "overlay_data_url": overlay_data_url,
    }


def start_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}/platform/model/det-fish-test"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if server.started:
            return server, thread, url
        if not thread.is_alive():
            raise RuntimeError("local Chromium fixture server stopped before startup")
        time.sleep(0.1)
    server.should_exit = True
    raise TimeoutError("local Chromium fixture server did not start")


def verify_image(page, image_id: str, preview_selector: str, expected_size: tuple[int, int]) -> None:
    image = page.locator(f"#{image_id}")
    preview = page.locator(preview_selector)
    image.evaluate("img => img.decode()")
    dimensions = image.evaluate("img => [img.naturalWidth, img.naturalHeight]")
    assert tuple(dimensions) == expected_size, (image_id, dimensions, expected_size)
    image_box = image.bounding_box()
    preview_box = preview.bounding_box()
    assert image_box and preview_box, (image_id, image_box, preview_box)
    displayed_ratio = image_box["width"] / image_box["height"]
    source_ratio = expected_size[0] / expected_size[1]
    assert abs(displayed_ratio - source_ratio) < 0.01, (image_id, displayed_ratio, source_ratio)
    assert image_box["x"] >= preview_box["x"] - 1
    assert image_box["y"] >= preview_box["y"] - 1
    assert image_box["x"] + image_box["width"] <= preview_box["x"] + preview_box["width"] + 1
    assert image_box["y"] + image_box["height"] <= preview_box["y"] + preview_box["height"] + 1


def prepare_fixtures() -> tuple[tuple[str, Path, tuple[int, int]], ...]:
    landscape_source = Path(os.environ["DET_FISH_LANDSCAPE_PHOTO"])
    portrait_source = Path(os.environ["DET_FISH_PORTRAIT_PHOTO"])
    with Image.open(landscape_source) as image:
        landscape = ImageOps.exif_transpose(image).convert("RGB").resize((800, 600), Image.Resampling.LANCZOS)
    with Image.open(portrait_source) as image:
        portrait = ImageOps.exif_transpose(image).convert("RGB").resize((600, 817), Image.Resampling.LANCZOS)

    inputs = ARTIFACTS / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    landscape_path = inputs / "fish-catch-landscape.jpg"
    portrait_path = inputs / "fish-catch-portrait.jpg"
    square_path = inputs / "fish-catch-near-square.jpg"
    landscape.save(landscape_path, quality=82, optimize=True)
    portrait.save(portrait_path, quality=82, optimize=True)
    square = Image.new("RGB", (600, 600), (244, 247, 246))
    contained = ImageOps.contain(landscape, (540, 540), Image.Resampling.LANCZOS)
    square.paste(contained, ((600 - contained.width) // 2, (600 - contained.height) // 2))
    square.save(square_path, quality=82, optimize=True)
    landscape.close()
    portrait.close()
    square.close()
    return (
        ("landscape", landscape_path, (800, 600)),
        ("portrait", portrait_path, (600, 817)),
        ("near-square", square_path, (600, 600)),
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for existing in ARTIFACTS.glob("*.png"):
        existing.unlink()

    fixtures = prepare_fixtures()
    viewports = (("desktop-1440", 1440, 1000, 2), ("narrow-390", 390, 844, 1))
    server, thread, url = start_server()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                for viewport_name, width, height, expected_columns in viewports:
                    for photo_name, photo_path, dimensions in fixtures:
                        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
                        try:
                            page.goto(url, wait_until="networkidle")
                            page.locator("#detTestModel option[value='UI_SIMULATION_ONLY']").wait_for(state="attached")
                            page.locator("#detTestFile").set_input_files(str(photo_path))
                            page.locator("#detTestOriginal").wait_for(state="visible")
                            verify_image(page, "detTestOriginal", ".det-fish-test-grid .panel:first-child .det-fish-image-preview", dimensions)
                            page.locator("#detTestRun").click()
                            page.locator("#detTestOverlay").wait_for(state="visible")
                            verify_image(page, "detTestOverlay", ".det-fish-test-grid .panel:nth-child(2) .det-fish-image-preview", dimensions)

                            columns = page.locator(".det-fish-test-grid").evaluate(
                                "node => getComputedStyle(node).gridTemplateColumns"
                            ).split()
                            assert len(columns) == expected_columns, (viewport_name, columns)
                            overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                            assert overflow <= 1, (viewport_name, photo_name, overflow)

                            screenshot_path = ARTIFACTS / f"{viewport_name}-{photo_name}.png"
                            page.screenshot(path=str(screenshot_path), full_page=True)
                            print(
                                f"PASS viewport={viewport_name} photo={photo_name} source={dimensions[0]}x{dimensions[1]} "
                                "original=contained overlay=contained grid=responsive inference=SIMULATED "
                                f"screenshot={screenshot_path.relative_to(ROOT)}"
                            )
                        except Exception:
                            page.screenshot(
                                path=str(ARTIFACTS / f"failure-{viewport_name}-{photo_name}.png"),
                                full_page=True,
                            )
                            raise
                        finally:
                            page.close()
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()

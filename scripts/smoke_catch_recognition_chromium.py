#!/usr/bin/env python3
"""Responsive Chromium UI smoke for catch recognition; API data is simulated."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.platform.routes.pages import PLATFORM_PAGES, templates


ARTIFACTS = ROOT / "test-results" / "catch-recognition-chromium"
PAGE = next(page for page in PLATFORM_PAGES if page.path == "/platform/model/catch-recognition")
app = FastAPI(docs_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


@app.get(PAGE.path)
def catch_recognition_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name=PAGE.template,
        context={
            "page": PAGE,
            "page_title": PAGE.title,
            "platform_pages": PLATFORM_PAGES,
            "catch_api_url": "/api/ui-simulation/catch-recognition/infer",
            "ui_simulation": True,
        },
    )


@app.get("/api/species")
def simulated_species_catalog():
    return [
        {"species_key": "grass_carp", "common_name_zh": "草鱼"},
        {"species_key": "common_carp", "common_name_zh": "鲤鱼"},
        {"species_key": "crucian_carp", "common_name_zh": "鲫鱼"},
    ]


@app.post("/api/ui-simulation/catch-recognition/infer")
async def simulated_catch_inference(file: UploadFile = File(...)):
    if "simulate-error" in (file.filename or ""):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "UI_SIMULATION_ONLY_ERROR",
                "message": "UI_SIMULATION_ONLY：模拟服务暂不可用。",
                "request_id": "SIMULATED-ERROR",
            },
        )
    data = await file.read(15 * 1024 * 1024 + 1)
    if not data or len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail={"code": "UI_SIMULATION_ONLY_INVALID", "message": "测试图片无效"})
    try:
        with Image.open(BytesIO(data)) as uploaded:
            image = ImageOps.exif_transpose(uploaded)
            width, height = image.size
            image_format = uploaded.format or "JPEG"
    except OSError as exc:
        raise HTTPException(status_code=400, detail={"code": "UI_SIMULATION_ONLY_INVALID", "message": "测试图片无法解码"}) from exc

    bbox = {"x1": 0.14, "y1": 0.18, "x2": 0.86, "y2": 0.84}
    return JSONResponse(
        {
            "request_id": "SIMULATED-UI-ONLY",
            "pipeline_version": "UI_SIMULATION_ONLY",
            "result_state": "RESULT_HIGH",
            "input": {"width": width, "height": height, "format": image_format},
            "status": "UI_SIMULATION_ONLY",
            "status_wire": "success",
            "message": "UI_SIMULATION_ONLY：模拟草鱼结果",
            "guidance": None,
            "detector": {
                "model_version": "UI_SIMULATION_ONLY",
                "onnx_sha256": None,
                "latency_ms": 0,
                "detections": [{"bbox": bbox, "confidence": 0.91}],
                "quality_gate": {"status": "SIMULATED"},
                "coordinate_space": "EXIF_TRANSPOSED_SOURCE_IMAGE_NORMALIZED",
            },
            "crop": {"bbox": bbox, "applied": True},
            "classifier": {
                "model_version": "UI_SIMULATION_ONLY",
                "model_status": "SIMULATED",
                "pipeline_type": "SIMULATED",
                "input": "crop",
                "top1": {"species": "草鱼", "species_key": "grass_carp", "confidence": 0.92},
                "top3": [
                    {"species": "草鱼", "species_key": "grass_carp", "confidence": 0.92},
                    {"species": "鲤鱼", "species_key": "common_carp", "confidence": 0.06},
                    {"species": "鲫鱼", "species_key": "crucian_carp", "confidence": 0.02},
                ],
                "latency_ms": 0,
            },
            "timing": {
                "total_ms": 0,
                "detector_ms": 0,
                "classifier_ms": 0,
                "stages": [
                    {"name": "input_validation", "status": "SUCCESS", "duration_ms": 0},
                    {"name": "detector", "status": "SUCCESS", "duration_ms": 0},
                    {"name": "crop", "status": "SUCCESS", "duration_ms": 0},
                    {"name": "classifier", "status": "SUCCESS", "duration_ms": 0},
                ],
            },
            "segmentation": {
                "status": "UNAVAILABLE",
                "reason_code": "SAM_NOT_QUALIFIED",
                "model_version": None,
                "quality": None,
                "mask": None,
            },
        }
    )


def start_server() -> tuple[uvicorn.Server, threading.Thread, str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}{PAGE.path}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if server.started:
            return server, thread, url
        if not thread.is_alive():
            raise RuntimeError("local UI simulation server stopped before startup")
        time.sleep(0.1)
    server.should_exit = True
    raise TimeoutError("local UI simulation server did not start")


def prepare_photos() -> tuple[tuple[str, Path, tuple[int, int]], ...]:
    landscape_source = Path(os.environ["DET_FISH_LANDSCAPE_PHOTO"])
    portrait_source = Path(os.environ["DET_FISH_PORTRAIT_PHOTO"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for existing in ARTIFACTS.glob("*.png"):
        existing.unlink()

    with Image.open(landscape_source) as uploaded:
        landscape = ImageOps.exif_transpose(uploaded).convert("RGB").resize((800, 600), Image.Resampling.LANCZOS)
    with Image.open(portrait_source) as uploaded:
        portrait = ImageOps.exif_transpose(uploaded).convert("RGB").resize((600, 817), Image.Resampling.LANCZOS)
    paths = []
    for name, image in (("landscape", landscape), ("portrait", portrait)):
        path = ARTIFACTS / f"{name}.jpg"
        image.save(path, quality=82, optimize=True)
        paths.append((name, path, image.size))
        image.close()
    return tuple(paths)


def assert_no_horizontal_overflow(page, viewport_name: str) -> None:
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    assert overflow <= 1, (viewport_name, overflow)


def main() -> None:
    photos = prepare_photos()
    server, thread, url = start_server()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                desktop = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                try:
                    desktop.goto(url, wait_until="networkidle")
                    assert desktop.locator("#catchRecognitionApp[data-ui-simulation='true']").count() == 1
                    assert desktop.get_by_text("UI_SIMULATION_ONLY", exact=False).count() >= 1
                    assert desktop.locator('.nav-link[href="/platform/model/catch-recognition"]').count() == 1
                    capabilities = desktop.evaluate(
                        """() => {
                          const twoD = document.createElement('canvas').getContext('2d');
                          const webgl2 = document.createElement('canvas').getContext('webgl2');
                          return {
                            userAgent: navigator.userAgent,
                            devicePixelRatio: window.devicePixelRatio,
                            canvas2d: Boolean(twoD),
                            webgl2: Boolean(webgl2),
                            cssTransitions: CSS.supports('transition', 'left 1s linear'),
                            reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
                          };
                        }"""
                    )
                    assert capabilities["canvas2d"] and capabilities["cssTransitions"], capabilities
                    (ARTIFACTS / "render-capabilities.json").write_text(json.dumps(capabilities, ensure_ascii=False, indent=2), encoding="utf-8")
                    timeline = desktop.evaluate("window.YUJIAN_CATCH_RECOGNITION_TIMELINE")
                    assert timeline["nominalProcessingMs"] == 2800
                    assert timeline["rr09Ms"] == 850
                    assert timeline["totalMs"] == 3650
                    assert [row["id"] for row in timeline["processingFrames"]] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
                    assert [row["at"] for row in timeline["rr09Frames"]] == [0, 100, 200, 380, 550, 680, 850]
                    assert desktop.locator(".catch-grid").evaluate("node => getComputedStyle(node).gridTemplateColumns.split(' ').length") == 3
                    assert_no_horizontal_overflow(desktop, "desktop-1440")
                    photo_name, photo_path, dimensions = photos[0]
                    desktop.locator("#catchFileInput").set_input_files(str(photo_path))
                    desktop.locator("#catchUploadPhoto").evaluate("image => image.decode()")
                    assert desktop.locator("#catchUploadPhoto").evaluate("image => [image.naturalWidth, image.naturalHeight]") == list(dimensions)
                    desktop.screenshot(path=str(ARTIFACTS / "desktop-photo-selected-ui-simulation.png"), full_page=True)
                    desktop.evaluate("window._catchUiSmokeStartedAt = performance.now()")
                    desktop.locator("#catchStartButton").click()
                    desktop.wait_for_function("document.querySelector('.catch-transition-photo')?.dataset.progress !== undefined", timeout=8000)
                    desktop.wait_for_function("Math.abs(Number(document.querySelector('.catch-transition-photo')?.dataset.progress || 0) - 0.5) < 0.08", timeout=1500)
                    geometry = desktop.evaluate(
                        """() => {
                          const source = document.querySelector('#catchProcessingMedia').getBoundingClientRect();
                          const target = document.querySelector('#catchResultMedia').getBoundingClientRect();
                          const moving = document.querySelector('.catch-transition-photo').getBoundingClientRect();
                          const progress = Number(document.querySelector('.catch-transition-photo').dataset.progress);
                          const atProgress = (from, to) => from + (to - from) * progress;
                          return {
                            progress,
                            elapsedMs: progress * 200,
                            leftError: Math.abs(moving.left - atProgress(source.left, target.left)),
                            topError: Math.abs(moving.top - atProgress(source.top, target.top)),
                            widthError: Math.abs(moving.width - atProgress(source.width, target.width)),
                            heightError: Math.abs(moving.height - atProgress(source.height, target.height)),
                          };
                        }"""
                    )
                    assert 0.42 <= geometry["progress"] <= 0.58, geometry
                    assert max(geometry[key] for key in ("leftError", "topError", "widthError", "heightError")) <= 4, geometry
                    desktop.wait_for_function("document.querySelector('#catchResultPanel')?.dataset.revealState === 'complete'", timeout=7000)
                    assert desktop.locator("#catchSpeciesName").inner_text() == "草鱼"
                    assert desktop.locator("#catchProcessingPhoto").get_attribute("src") == desktop.locator("#catchResultPhoto").get_attribute("src")
                    assert desktop.locator("#catchResultPanel").get_attribute("data-business-state") == "ready"
                    elapsed = desktop.evaluate("performance.now() - window._catchUiSmokeStartedAt")
                    assert 3500 <= elapsed <= 3900, elapsed
                    assert_no_horizontal_overflow(desktop, "desktop-1440-result")
                    desktop.screenshot(path=str(ARTIFACTS / f"desktop-result-{photo_name}-ui-simulation.png"), full_page=True)
                    print(f"PASS desktop 1440x1000; renderer capabilities={capabilities}; geometry midpoint={geometry}; nominal visual duration={elapsed:.1f}ms; result/data marked UI_SIMULATION_ONLY")
                finally:
                    desktop.close()

                mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
                try:
                    mobile.goto(url, wait_until="networkidle")
                    assert mobile.locator(".catch-grid").evaluate("node => getComputedStyle(node).gridTemplateColumns.split(' ').length") == 1
                    assert_no_horizontal_overflow(mobile, "mobile-390")
                    photo_name, photo_path, dimensions = photos[1]
                    mobile.locator("#catchFileInput").set_input_files(str(photo_path))
                    mobile.locator("#catchUploadPhoto").evaluate("image => image.decode()")
                    mobile.locator("#catchStartButton").click()
                    mobile.wait_for_function("document.querySelector('#catchResultPanel')?.dataset.revealState === 'complete'", timeout=9000)
                    assert mobile.locator("#catchResultPhoto").evaluate("image => [image.naturalWidth, image.naturalHeight]") == list(dimensions)
                    assert mobile.locator("#catchSpeciesName").inner_text() == "草鱼"
                    assert_no_horizontal_overflow(mobile, "mobile-390-result")
                    mobile.screenshot(path=str(ARTIFACTS / f"mobile-result-{photo_name}-ui-simulation.png"), full_page=True)
                    print("PASS mobile 390x844; portrait photo and result form fit without horizontal overflow; UI_SIMULATION_ONLY")
                finally:
                    mobile.close()

                narrow = browser.new_page(viewport={"width": 320, "height": 480}, device_scale_factor=1)
                try:
                    narrow.goto(url, wait_until="networkidle")
                    assert_no_horizontal_overflow(narrow, "narrow-320")
                    narrow.screenshot(path=str(ARTIFACTS / "narrow-320-initial-ui-simulation.png"), full_page=True)
                    print("PASS narrow 320x480; no horizontal overflow; UI_SIMULATION_ONLY")
                finally:
                    narrow.close()

                failed = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
                try:
                    failed.goto(url, wait_until="networkidle")
                    _photo_name, photo_path, _dimensions = photos[0]
                    failed.locator("#catchFileInput").set_input_files({"name": "simulate-error.jpg", "mimeType": "image/jpeg", "buffer": photo_path.read_bytes()})
                    failed.locator("#catchStartButton").click()
                    failed.wait_for_function("document.querySelector('#catchResultPanel')?.dataset.businessState === 'recovery'", timeout=5000)
                    assert failed.locator("#catchSpeciesName").inner_text().startswith("UI_SIMULATION_ONLY")
                    assert failed.locator("#catchResultPanel").get_attribute("data-reveal-state") != "complete"
                    failed.screenshot(path=str(ARTIFACTS / "mobile-error-no-success-reveal-ui-simulation.png"), full_page=True)
                    print("PASS simulated service error; no success reveal; UI_SIMULATION_ONLY")
                finally:
                    failed.close()
            finally:
                browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()

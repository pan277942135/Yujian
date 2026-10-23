#!/usr/bin/env python3
"""Responsive Chromium UI smoke for catch recognition; API data is simulated."""

from __future__ import annotations

from io import BytesIO
import asyncio
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
from PIL import Image, ImageChops, ImageOps, ImageStat
from playwright.sync_api import sync_playwright
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.offline_test_guard import install
install()

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

    if "simulate-slow" in (file.filename or ""):
        await asyncio.sleep(4)
    result_state = next((value for name,value in {
        "simulate-medium":"RESULT_MEDIUM", "simulate-low":"RESULT_LOW",
        "simulate-nofish":"ERROR_NO_FISH", "simulate-quality":"ERROR_IMAGE_QUALITY",
    }.items() if name in (file.filename or "")), "RESULT_HIGH")
    bbox = {"x1": 0.14, "y1": 0.18, "x2": 0.86, "y2": 0.84}
    return JSONResponse(
        {
            "request_id": "SIMULATED-UI-ONLY",
            "pipeline_version": "UI_SIMULATION_ONLY",
            "result_state": result_state,
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
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Generated color grids only: no user/real fish photos and no production inference.
    landscape = Image.new("RGB", (800, 600), (75, 125, 115))
    portrait = Image.new("RGB", (600, 817), (145, 130, 100))
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
                # Layout-only matrix: no photo selection required.
                for width,height in [(1440,1000),(1280,800),(1024,768),(768,1024),(480,900),(430,932),(390,844),(360,640),(320,480)]:
                    check=browser.new_page(viewport={"width":width,"height":height})
                    errors=[]
                    check.on("pageerror",lambda error: errors.append(str(error)))
                    check.goto(url,wait_until="networkidle")
                    assert_no_horizontal_overflow(check,f"{width}x{height}")
                    assert check.locator('[data-component="ProcessJourney"] li').count()==5
                    assert not check.locator('[data-component="AdvancedDiagnostics"]').evaluate("n=>n.open")
                    assert check.locator('[data-stage-step]').count()==3
                    if width < 768:
                        assert check.locator('#catchUploadPanel').is_visible()
                        assert check.locator('#catchProcessingPanel').is_hidden()
                        assert check.locator('#catchResultPanel').is_hidden()
                        assert check.locator('.catch-journey ol').evaluate("n=>n.scrollWidth>n.clientWidth")
                    else:
                        assert check.locator('#catchResultEmpty').is_visible()
                    check.screenshot(path=str(ARTIFACTS/f"shell-{width}x{height}-UI_SIMULATION_ONLY.png"),full_page=True)
                    assert not errors,errors
                    check.close()
                    print(f"PASS B1 shell {width}x{height}; UI_SIMULATION_ONLY")
                desktop = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                try:
                    desktop.goto(url, wait_until="networkidle")
                    assert desktop.locator("#catchRecognitionApp[data-ui-simulation='true']").count() == 1
                    assert desktop.get_by_text("UI_SIMULATION_ONLY", exact=False).count() >= 1
                    assert desktop.locator(".platform-sidebar").count() == 0
                    assert desktop.locator('[data-component="ProductShell"]').count() == 1
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
                    assert desktop.locator('#catchPhotoStage [data-optical-layer]').count()==7
                    assert desktop.locator(".catch-grid").evaluate("node => getComputedStyle(node).gridTemplateColumns.split(' ').length") == 3
                    assert_no_horizontal_overflow(desktop, "desktop-1440")
                    photo_name, photo_path, dimensions = photos[0]
                    desktop.locator("#catchFileInput").set_input_files(str(photo_path))
                    desktop.locator("#catchUploadPhoto").evaluate("image => image.decode()")
                    assert desktop.locator("#catchUploadPhoto").evaluate("image => [image.naturalWidth, image.naturalHeight]") == list(dimensions)
                    desktop.locator("#catchProcessingPhoto").evaluate("image => image.decode()")
                    assert desktop.locator("#catchProcessingPhoto").evaluate("image => [image.naturalWidth, image.naturalHeight]") == list(dimensions)
                    assert not desktop.locator("#catchTabProcessing").is_disabled()
                    assert desktop.locator("#catchTabResult").is_disabled()
                    bbox_probe = desktop.evaluate(
                        """() => {
                          const media=document.querySelector('#catchProcessingMedia');
                          const stage=document.querySelector('#catchPhotoStage');
                          const host=document.querySelector('#catchPhotoEffects');
                          const focus=document.querySelector('#catchRegionGlow');
                          const sourcePhoto=document.querySelector('#catchProcessingPhoto');
                          const sourceDimensions=[sourcePhoto?.naturalWidth||0,sourcePhoto?.naturalHeight||0];
                          if(!sourcePhoto?.complete||sourceDimensions.some(value=>value<=0))throw new Error('processing photo is not fully decoded');
                          const stageWidth=640;
                          const stageHeight=Math.max(1,Math.round(stageWidth*sourceDimensions[1]/sourceDimensions[0]));
                          Object.assign(stage.style,{position:'relative',inset:'auto',width:`${stageWidth}px`,height:`${stageHeight}px`});
                          Object.assign(media.style,{width:`${stageWidth}px`,height:`${stageHeight}px`,aspectRatio:'auto'});
                          Object.assign(host.style,{position:'absolute',inset:'0',width:'100%',height:'100%'});
                          const renderer=window.YujianCatch.createSwosRenderer({
                            media,host,focus,photoStage:stage,
                            maskCanvas:document.querySelector('#catchMaskCanvas'),
                            waterGLCanvas:document.querySelector('#catchWaterGLCanvas'),
                            water2DCanvas:document.querySelector('#catchWater2DCanvas'),
                            getReducedMotion:()=>false,
                          });
                          renderer.start(performance.now(),{
                            processingFrames:window.YUJIAN_CATCH_RECOGNITION_TIMELINE.processingFrames,
                            waterStrength:[0,10,45,75,60,35,10,0],
                            goldStrength:[0,0,15,45,100,100,100,100],
                          });
                          renderer.stop(false);
                          const frame=renderer.renderAt(1550);
                          const result={
                            fixture_only:true,
                            frame:frame.frame,
                            display:getComputedStyle(focus).display,
                            opacity:Number(getComputedStyle(focus).opacity),
                            geometryTrust:focus.dataset.geometryTrust,
                            maskTrust:media.dataset.maskTrust,
                            bboxOutlineDisplay:getComputedStyle(document.querySelector('#catchBBoxOverlay')).display,
                            source_photo_loaded:sourcePhoto.complete&&sourcePhoto.naturalWidth>0,
                            sourceDimensions,
                            stageRect:stage.getBoundingClientRect().toJSON(),
                            mediaRect:media.getBoundingClientRect().toJSON(),
                            focusRect:focus.getBoundingClientRect().toJSON(),
                          };
                          window._swosBboxFocusFixtureRenderer=renderer;
                          return result;
                        }"""
                    )
                    assert bbox_probe["fixture_only"] is True
                    assert bbox_probe["frame"] == "P4"
                    assert bbox_probe["geometryTrust"] == "NONE_NO_SPATIAL_FOCUS"
                    assert bbox_probe["maskTrust"] == "NONE"
                    assert bbox_probe["source_photo_loaded"] is True
                    assert bbox_probe["sourceDimensions"] == list(dimensions)
                    source_ratio=dimensions[0]/dimensions[1]
                    assert abs(bbox_probe["stageRect"]["width"]/bbox_probe["stageRect"]["height"]-source_ratio) < 0.005, bbox_probe
                    assert abs(bbox_probe["mediaRect"]["width"]/bbox_probe["mediaRect"]["height"]-source_ratio) < 0.005, bbox_probe
                    desktop.locator("#catchProcessingMedia").screenshot(path=str(ARTIFACTS / "product-renderer-no-geometry-baseline-TEST_FIXTURE.png"))
                    bbox_focus = desktop.evaluate(
                        """() => {
                          const renderer=window._swosBboxFocusFixtureRenderer;
                          renderer.setGeometry({x1:.3,y1:.25,x2:.7,y2:.72});
                          const frame=renderer.renderAt(1550);
                          const focus=document.querySelector('#catchRegionGlow');
                          const media=document.querySelector('#catchProcessingMedia');
                          return {
                            fixture_only:true,
                            frame:frame.frame,
                            display:getComputedStyle(focus).display,
                            opacity:Number(getComputedStyle(focus).opacity),
                            geometryTrust:focus.dataset.geometryTrust,
                            maskTrust:media.dataset.maskTrust,
                            bboxOutlineDisplay:getComputedStyle(document.querySelector('#catchBBoxOverlay')).display,
                            mediaRect:media.getBoundingClientRect().toJSON(),
                            focusRect:focus.getBoundingClientRect().toJSON(),
                          };
                        }"""
                    )
                    assert bbox_focus["fixture_only"] is True
                    assert bbox_focus["frame"] == "P4"
                    assert bbox_focus["display"] != "none"
                    assert bbox_focus["opacity"] > 0
                    assert bbox_focus["geometryTrust"] == "BBOX_ONLY_SOFT_REGION"
                    assert bbox_focus["maskTrust"] == "BBOX_ONLY"
                    assert bbox_focus["bboxOutlineDisplay"] == "none", bbox_focus
                    desktop.locator("#catchProcessingMedia").screenshot(path=str(ARTIFACTS / "product-renderer-bbox-soft-focus-TEST_FIXTURE.png"))
                    with Image.open(ARTIFACTS / "product-renderer-no-geometry-baseline-TEST_FIXTURE.png") as baseline, Image.open(ARTIFACTS / "product-renderer-bbox-soft-focus-TEST_FIXTURE.png") as focused:
                        delta=ImageChops.difference(baseline.convert("RGB"),focused.convert("RGB"))
                        fx=bbox_focus["focusRect"];mx=bbox_focus["mediaRect"]
                        left=max(0,int(fx["left"]-mx["left"]+fx["width"]*.2));top=max(0,int(fx["top"]-mx["top"]+fx["height"]*.2))
                        right=min(delta.width,int(fx["right"]-mx["left"]-fx["width"]*.2));bottom=min(delta.height,int(fx["bottom"]-mx["top"]-fx["height"]*.2))
                        roi_delta=delta.crop((left,top,right,bottom))
                        bbox_focus["soft_focus_roi_mean_pixel_delta"]=sum(ImageStat.Stat(roi_delta).mean)/3
                        bbox_focus["soft_focus_roi_max_pixel_delta"]=max(channel[1] for channel in roi_delta.getextrema())
                        assert bbox_focus["soft_focus_roi_mean_pixel_delta"] >= 2.5, bbox_focus
                        assert bbox_focus["soft_focus_roi_max_pixel_delta"] >= 12, bbox_focus
                    (ARTIFACTS / "product-renderer-bbox-focus-TEST_FIXTURE.json").write_text(json.dumps(bbox_focus, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(
                        "PASS BBOX soft-focus pixel delta; TEST_FIXTURE only; "
                        f"trust={bbox_focus['maskTrust']}; "
                        f"ROI mean={bbox_focus['soft_focus_roi_mean_pixel_delta']:.2f}; "
                        f"ROI max={bbox_focus['soft_focus_roi_max_pixel_delta']:.0f}; "
                        "precise outline=none"
                    )
                    desktop.evaluate(
                        """() => {
                          window._swosBboxFocusFixtureRenderer.stop();
                          delete window._swosBboxFocusFixtureRenderer;
                          const stage=document.querySelector('#catchPhotoStage');
                          for(const [node,props] of [
                            [stage,['position','inset','width','height']],
                            [document.querySelector('#catchProcessingMedia'),['width','height','aspect-ratio']],
                            [document.querySelector('#catchPhotoEffects'),['position','inset','width','height']],
                          ])for(const prop of props)node.style.removeProperty(prop);
                        }"""
                    )
                    # The fixture deliberately mutates the product renderer's photo stage.
                    # Reload and reselect the same generated photo so later transition checks
                    # start from the real page state produced by chooseFile().
                    desktop.reload(wait_until="networkidle")
                    desktop.locator("#catchFileInput").set_input_files(str(photo_path))
                    desktop.locator("#catchUploadPhoto").evaluate("image => image.decode()")
                    desktop.locator("#catchProcessingPhoto").evaluate("image => image.decode()")
                    assert desktop.locator("#catchProcessingPhoto").evaluate("image => image.naturalWidth") == dimensions[0]
                    desktop.screenshot(path=str(ARTIFACTS / "desktop-photo-selected-ui-simulation.png"), full_page=True)
                    desktop.evaluate("window._catchUiSmokeStartedAt = performance.now()")
                    desktop.locator("#catchStartButton").click()
                    desktop.wait_for_function("document.querySelector('.catch-transition-photo')?.dataset.progress !== undefined", timeout=8000)
                    desktop.wait_for_function("Math.abs(Number(document.querySelector('.catch-transition-photo')?.dataset.progress || 0) - 0.5) < 0.08", timeout=1500)
                    geometry = desktop.evaluate(
                        """() => {
                          const g=window.YujianCatch.PhotoViewport;
                          const img=document.querySelector('#catchProcessingPhoto');
                          const source = g.viewport(img.naturalWidth,img.naturalHeight,document.querySelector('#catchProcessingMedia').getBoundingClientRect()).rect;
                          const target = g.viewport(img.naturalWidth,img.naturalHeight,document.querySelector('#catchResultMedia').getBoundingClientRect()).rect;
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
                    assert desktop.locator('#catchProcessingMedia').get_attribute('data-swos-frame')=='R5'
                    assert desktop.locator('#catchProcessingMedia').get_attribute('data-relative-water-strength')=='0'
                    assert desktop.locator('#catchProcessingMedia').get_attribute('data-relative-fish-gold-strength')=='0'
                    assert desktop.locator('#catchResultPanel').get_attribute('data-rr09-frame')=='R5'
                    assert not desktop.locator('#catchTabResult').is_disabled()
                    assert desktop.locator("#catchSpeciesName").inner_text() == "草鱼"
                    assert desktop.locator("#catchResultEmpty").is_hidden(), "successful result must hide the empty-state placeholder"
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
                    assert mobile.locator("#catchResultEmpty").is_hidden(), "mobile successful result must hide the empty-state placeholder"
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

                for filename,expected in [("medium","RESULT_MEDIUM"),("low","RESULT_LOW"),("nofish","ERROR_NO_FISH"),("quality","ERROR_IMAGE_QUALITY")]:
                    recovery=browser.new_page(viewport={"width":390,"height":844})
                    try:
                        recovery.goto(url,wait_until="networkidle")
                        recovery.locator("#catchFileInput").set_input_files({"name":f"simulate-{filename}.jpg","mimeType":"image/jpeg","buffer":photos[0][1].read_bytes()})
                        recovery.locator("#catchStartButton").click()
                        recovery.wait_for_function("document.querySelector('#catchResultContent').classList.contains('visible')")
                        assert recovery.locator("#catchResultPanel").get_attribute("data-reveal-state")!="complete"
                        assert recovery.locator("#catchSaveButton").is_disabled()
                        if filename in ("medium","low"):
                            recovery.locator("#catchSpeciesSelect").select_option("grass_carp")
                            assert recovery.locator("#catchSaveButton").is_enabled()
                        print(f"PASS {expected}; no HIGH reveal; explicit confirmation required; UI_SIMULATION_ONLY")
                    finally:
                        recovery.close()
                slow=browser.new_page(viewport={"width":390,"height":844},reduced_motion="reduce")
                try:
                    slow.goto(url,wait_until="networkidle")
                    slow.locator("#catchFileInput").set_input_files({"name":"simulate-slow.jpg","mimeType":"image/jpeg","buffer":photos[0][1].read_bytes()})
                    slow.locator("#catchStartButton").click()
                    slow.wait_for_timeout(3000)
                    assert slow.locator("#catchSpeciesName").inner_text()==""
                    assert slow.locator("#catchRecognitionApp").get_attribute("data-active-stage")=="processing"
                    assert slow.locator("#catchProcessingPanel").is_visible()
                    assert slow.locator("#catchResultPanel").is_hidden()
                    slow.wait_for_function("document.querySelector('#catchResultPanel').dataset.revealState==='complete'",timeout=4000)
                    assert slow.locator(".catch-transition-photo").count()==0
                    print("PASS slow response and Reduce Motion; no premature name; UI_SIMULATION_ONLY")
                finally:
                    slow.close()

                cancelled = browser.new_page(viewport={"width":390,"height":844})
                try:
                    cancelled.goto(url,wait_until="networkidle")
                    cancelled.locator("#catchFileInput").set_input_files(str(photos[0][1]))
                    cancelled.locator("#catchStartButton").click()
                    cancelled.locator("#catchCancelButton").click()
                    cancelled.wait_for_timeout(3800)
                    assert cancelled.locator("#catchSpeciesName").inner_text()==""
                    assert cancelled.locator(".catch-transition-photo").count()==0
                    assert cancelled.locator("#catchRecognitionApp").get_attribute("data-stage")=="CANCELLED"
                    # Cancel during the transition itself, then choose another photo.
                    cancelled.locator("#catchStartButton").click()
                    cancelled.wait_for_function("!!document.querySelector('.catch-transition-photo')",timeout=7000)
                    cancelled.locator("#catchCancelButton").click()
                    assert cancelled.locator("#catchRecognitionApp").get_attribute("data-stage")=="CANCELLED"
                    cancelled.locator("#catchFileInput").set_input_files(str(photos[1][1]))
                    cancelled.wait_for_timeout(1200)
                    assert cancelled.locator("#catchSpeciesName").inner_text()==""
                    assert cancelled.locator(".catch-transition-photo").count()==0
                    assert cancelled.locator("#catchProcessingMedia #catchProcessingPhoto").count()==1
                    assert cancelled.locator("#catchUploadPhoto").evaluate("n=>n.naturalWidth")==photos[1][2][0]
                    assert cancelled.locator("#catchStartButton").is_enabled()
                    print("PASS cancel/clear during reveal and photo replacement; no stale result; UI_SIMULATION_ONLY")
                finally:
                    cancelled.close()

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

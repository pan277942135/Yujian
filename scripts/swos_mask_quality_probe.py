#!/usr/bin/env python3
"""Collect source-pinned SAM mask candidates and latency evidence from a no-traffic revision.

This is a quality probe, not a production-mask certification tool. It uses only public,
license-attributed photo fixtures. No private user photo is downloaded or uploaded.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

USER_AGENT = "YuJian-SWOS-Resource-Quality-Probe/1.1 (+https://github.com/pan277942135/Yujian)"
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_EDGE = 1600


def digest_bytes(raw: bytes, algorithm: str) -> str:
    if algorithm not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported digest algorithm: {algorithm}")
    return hashlib.new(algorithm, raw).hexdigest()


def verify_expected_digest(raw: bytes, expected: dict[str, str]) -> dict[str, str]:
    algorithm = str(expected.get("algorithm", "")).lower()
    value = str(expected.get("value", "")).lower()
    actual = digest_bytes(raw, algorithm)
    if not value or actual != value:
        raise ValueError(
            f"fixture digest mismatch: {algorithm} expected={value or '<missing>'} actual={actual}"
        )
    return {"algorithm": algorithm, "expected": value, "actual": actual, "verified": "true"}


def decode_data_url(value: str) -> bytes:
    if not isinstance(value, str) or "," not in value:
        raise ValueError("expected a data URL")
    header, payload = value.split(",", 1)
    if not header.startswith("data:") or ";base64" not in header:
        raise ValueError("expected a base64 data URL")
    return base64.b64decode(payload, validate=True)


def build_multipart(image_bytes: bytes, filename: str, field_name: str = "image") -> tuple[bytes, str]:
    boundary = "----YuJianSWOS" + hashlib.sha256(
        (filename + str(len(image_bytes)) + str(time.time_ns())).encode("utf-8")
    ).hexdigest()[:24]
    safe_name = filename.replace('"', "").replace("\r", "").replace("\n", "")
    safe_field = field_name.replace('"', "").replace("\r", "").replace("\n", "")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{safe_field}"; filename="{safe_name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def console_auth_headers() -> dict[str, str]:
    """Build the same console-cookie value as app/secure.py without logging the key."""
    access_key = os.getenv("SWOS_PROBE_CONSOLE_ACCESS_KEY", "").strip()
    if not access_key:
        raise ValueError("AUTH_NOT_CONFIGURED: SWOS_PROBE_CONSOLE_ACCESS_KEY is missing")
    cookie_value = hashlib.sha256(("yujian-console:" + access_key).encode("utf-8")).hexdigest()
    return {"Cookie": f"yujian_console={cookie_value}"}


def retry_delay_seconds(headers: Any, attempt: int) -> float:
    """Respect Retry-After where available, otherwise use bounded exponential backoff."""
    value = str(headers.get("Retry-After", "")).strip() if headers else ""
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = float(2 ** attempt)
    else:
        delay = float(2 ** attempt)
    return max(0.5, min(8.0, delay))


def request_json(url: str, data: bytes | None = None, headers: dict[str, str] | None = None,
                 timeout: int = 180) -> tuple[int | None, dict[str, Any], float]:
    request_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    request_headers.update(headers or {})
    method = "POST" if data is not None else "GET"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            elapsed_ms = (time.perf_counter() - started) * 1000
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                payload = {"_non_json_body_prefix": raw[:600].decode("utf-8", "replace")}
            return int(response.status), payload if isinstance(payload, dict) else {"_payload": payload}, round(elapsed_ms, 1)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = {"error": raw[:1200].decode("utf-8", "replace")}
        return int(exc.code), payload if isinstance(payload, dict) else {"_payload": payload}, round(elapsed_ms, 1)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return None, {"error": f"{type(exc).__name__}: {exc}"}, round(elapsed_ms, 1)


def download_source(spec: dict[str, Any], directory: Path) -> tuple[bytes, Path, dict[str, str]]:
    request = urllib.request.Request(
        spec["download_url"],
        headers={"User-Agent": USER_AGENT, "Accept": "image/jpeg,*/*;q=0.8"},
    )
    raw: bytes | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read(MAX_DOWNLOAD_BYTES + 1)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= 3:
                raise
            time.sleep(retry_delay_seconds(exc.headers, attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt >= 3:
                raise
            time.sleep(min(8.0, float(2 ** attempt)))
    if raw is None:
        raise RuntimeError("fixture download exhausted retries without a response")
    if not raw:
        raise ValueError("empty fixture download")
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"source exceeds {MAX_DOWNLOAD_BYTES} bytes")
    verified = verify_expected_digest(raw, spec["expected_digest"])
    path = directory / spec["filename"]
    path.write_bytes(raw)
    verified["sha256"] = digest_bytes(raw, "sha256")
    verified["bytes"] = str(len(raw))
    return raw, path, verified


def encode_jpeg(image: Image.Image, path: Path, max_edge: int = MAX_UPLOAD_EDGE) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    image.save(path, format="JPEG", quality=88, optimize=True, progressive=False)
    return path.read_bytes()


def make_variants(spec: dict[str, Any], raw: bytes, work_dir: Path) -> list[dict[str, Any]]:
    with Image.open(io.BytesIO(raw)) as opened:
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
    source_size = [oriented.width, oriented.height]
    resized_path = work_dir / f"{spec['id']}-resized.jpg"
    resized_bytes = encode_jpeg(oriented.copy(), resized_path)
    variants = [{
        "id": spec["id"],
        "source_id": spec["id"],
        "variant": "source_resized_max_edge_1600",
        "intent": spec["intent"],
        "bytes": resized_bytes,
        "source_dimensions": source_size,
        "upload_filename": resized_path.name,
    }]

    if spec["id"] == "gambia_rod_landscape":
        dark = ImageEnhance.Brightness(oriented).enhance(0.38)
        dark_path = work_dir / "gambia-dark.jpg"
        dark_bytes = encode_jpeg(dark, dark_path)
        variants.append({
            "id": "gambia_dark", "source_id": spec["id"], "variant": "brightness_0_38",
            "intent": "low-light robustness; derived fixture, not an independent scene",
            "bytes": dark_bytes, "source_dimensions": source_size, "upload_filename": dark_path.name,
        })
        square = Image.new("RGB", (1200, 1200), (244, 247, 246))
        contained = ImageOps.contain(oriented, (1080, 1080), Image.Resampling.LANCZOS)
        square.paste(contained, ((1200 - contained.width) // 2, (1200 - contained.height) // 2))
        square_path = work_dir / "gambia-near-square.jpg"
        square_bytes = encode_jpeg(square, square_path, max_edge=1200)
        variants.append({
            "id": "gambia_near_square", "source_id": spec["id"], "variant": "contain_1200x1200",
            "intent": "contain/letterbox geometry; derived fixture, not an independent scene",
            "bytes": square_bytes, "source_dimensions": [1200, 1200], "upload_filename": square_path.name,
        })
        dark.close()
        square.close()
    elif spec["id"] == "auckland_boat_portrait":
        low = ImageOps.contain(oriented, (320, 480), Image.Resampling.LANCZOS)
        low_path = work_dir / "auckland-low-resolution.jpg"
        low_bytes = encode_jpeg(low, low_path, max_edge=480)
        variants.append({
            "id": "auckland_low_resolution", "source_id": spec["id"], "variant": "max_edge_480",
            "intent": "low-resolution robustness; derived fixture, not an independent scene",
            "bytes": low_bytes, "source_dimensions": [low.width, low.height], "upload_filename": low_path.name,
        })
        low.close()
    elif spec["id"] == "catching_fish_sunset":
        dark = ImageEnhance.Brightness(oriented).enhance(0.55)
        dark_path = work_dir / "sunset-dark.jpg"
        dark_bytes = encode_jpeg(dark, dark_path)
        variants.append({
            "id": "sunset_dark", "source_id": spec["id"], "variant": "brightness_0_55",
            "intent": "additional low-light robustness; derived fixture, not an independent scene",
            "bytes": dark_bytes, "source_dimensions": source_size, "upload_filename": dark_path.name,
        })
        dark.close()
    oriented.close()
    return variants


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize the same-request product inference adapter response."""
    detector = payload.get("detector") or {}
    gate = detector.get("quality_gate") or {}
    detections = detector.get("detections") or []
    primary = gate.get("primary_bbox") or ((detections[0].get("bbox") or {}) if detections else {})
    segmentation = payload.get("segmentation") or {}
    quality = segmentation.get("quality") or {}
    mask = segmentation.get("mask") or {}
    source_input = payload.get("input") or {}
    checkpoint_sha = segmentation.get("checkpoint_sha256")
    mask_present = bool(mask.get("data_base64"))
    mask_sha = mask.get("sha256")
    mask_executed = bool(
        mask_present
        and segmentation.get("geometry_trust_state") == "REVIEW_PROXY"
        and checkpoint_sha
        and len(str(checkpoint_sha)) == 64
        and mask_sha
        and len(str(mask_sha)) == 64
    )
    primary_detection = next((item for item in detections if item.get("bbox") == primary), {})
    return {
        "result_state": payload.get("result_state"),
        "input": {
            "source_sha256": source_input.get("source_sha256"),
            "width": source_input.get("width"),
            "height": source_input.get("height"),
            "format": source_input.get("format"),
            "coordinate_space": detector.get("coordinate_space"),
        },
        "detector": {
            "model": detector.get("model_version"),
            "onnx_sha256": detector.get("onnx_sha256"),
            "detections_count": len(detections),
            "primary_detection_found": bool(primary),
            "primary_bbox": primary or None,
            "quality_gate_status": gate.get("status"),
            "primary_confidence": primary_detection.get("confidence"),
            "processing_ms": detector.get("latency_ms"),
        },
        "classifier": {
            "model": (payload.get("classifier") or {}).get("model_version"),
            "top1": (payload.get("classifier") or {}).get("top1"),
            "result_state": payload.get("result_state"),
        },
        "segmentation": {
            "executed": mask_executed,
            "status": segmentation.get("status"),
            "trust_state": segmentation.get("geometry_trust_state"),
            "model": segmentation.get("model_version"),
            "checkpoint_sha256": checkpoint_sha,
            "quality": quality.get("status"),
            "quality_reason": quality.get("reason"),
            "mask_area_ratio": quality.get("mask_area_ratio"),
            "edge_ratio": quality.get("edge_ratio"),
            "connected_components": quality.get("connected_components"),
            "mask_width": mask.get("width"),
            "mask_height": mask.get("height"),
            "mask_sha256": mask.get("sha256"),
            "source_image_sha256": segmentation.get("source_image_sha256"),
            "processing_ms": quality.get("processing_ms"),
        },
        "timing": payload.get("timing") or {},
        "mask_data_base64": mask.get("data_base64"),
    }


def verify_same_request_binding(payload: dict[str, Any], expected_upload_sha: str) -> None:
    source_sha = ((payload.get("input") or {}).get("source_sha256"))
    if source_sha != expected_upload_sha:
        raise ValueError(f"inference source SHA mismatch: expected {expected_upload_sha}, got {source_sha}")
    segmentation = payload.get("segmentation") or {}
    mask = segmentation.get("mask") or {}
    for label, actual in (
        ("segmentation", segmentation.get("source_image_sha256")),
        ("mask", mask.get("source_image_sha256")),
    ):
        if actual is not None and actual != source_sha:
            raise ValueError(f"{label} source SHA differs from inference input")
    primary = ((payload.get("detector") or {}).get("quality_gate") or {}).get("primary_bbox")
    sam_bbox = segmentation.get("bbox")
    if mask.get("data_base64"):
        if not primary or not sam_bbox:
            raise ValueError("Mask response is missing its same-request detector BBox")
        if any(abs(float(primary[key]) - float(sam_bbox[key])) > 0.00001 for key in ("x1", "y1", "x2", "y2")):
            raise ValueError("SAM BBox differs from the primary detector BBox")
        source_input = payload.get("input") or {}
        expected_dimensions = (source_input.get("width"), source_input.get("height"))
        mask_dimensions = (mask.get("width"), mask.get("height"))
        if all(value is not None for value in expected_dimensions) and mask_dimensions != expected_dimensions:
            raise ValueError(f"Mask dimensions {mask_dimensions} differ from inference input {expected_dimensions}")


def write_mask_artifacts(source_bytes: bytes, mask_png: bytes, output_dir: Path, name: str) -> dict[str, str]:
    """Write review-only overlay, binary mask and cutout from the same source pixels."""
    with Image.open(io.BytesIO(source_bytes)) as opened:
        source = ImageOps.exif_transpose(opened).convert("RGBA")
    with Image.open(io.BytesIO(mask_png)) as opened_mask:
        mask = opened_mask.convert("L")
    if source.size != mask.size:
        raise ValueError(f"Mask/source dimensions differ: {mask.size} vs {source.size}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{name}_sam_mask.png"
    overlay_path = output_dir / f"{name}_mask_overlay.png"
    cutout_path = output_dir / f"{name}_transparent_fish.png"
    mask.save(mask_path, format="PNG", optimize=True)
    gold = Image.new("RGBA", source.size, (255, 205, 103, 0))
    gold.putalpha(mask.point(lambda value: int(value * 0.42)))
    overlay = Image.alpha_composite(source, gold)
    overlay.save(overlay_path, format="PNG", optimize=True)
    cutout = source.copy()
    cutout.putalpha(mask)
    cutout.save(cutout_path, format="PNG", optimize=True)
    source.close()
    mask.close()
    gold.close()
    overlay.close()
    cutout.close()
    return {"mask": mask_path.name, "overlay": overlay_path.name, "cutout": cutout_path.name}


def write_report(output_dir: Path, report: dict[str, Any], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = report.get("cases", [])
    masks = [case for case in cases if case.get("segmentation", {}).get("executed")]
    latency = [case.get("timing", {}).get("round_trip_ms") for case in cases
               if isinstance(case.get("timing", {}).get("round_trip_ms"), (int, float))]
    lines = [
        "# SWOS resource quality probe",
        "",
        f"- Overall probe status: **{report.get('status')}**",
        f"- Production Mask qualification: **{report.get('production_mask_qualification')}**",
        f"- Authentication status: **{report.get('auth_status', 'NOT_REPORTED')}**",
        f"- Fixture HTTP 429 errors: **{len(report.get('fixture_download_errors', []))}**",
        f"- Source SHA: {report.get('expected_source_sha')}",
        f"- Preview revision: {report.get('expected_revision')}",
        f"- Independent licensed source scenes: {len(manifest.get('sources', []))}",
        f"- Mask outputs collected: {len(masks)} / {len(cases)} cases",
        f"- Round-trip median/max: {round(statistics.median(latency), 1) if latency else 'n/a'} / {max(latency) if latency else 'n/a'} ms",
        "",
        "| Case | Variant | Detector assessment | SAM quality | Detector ms | SAM ms | Round-trip ms |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for case in cases:
        det = case.get("detector") or {}
        seg = case.get("segmentation") or {}
        timing = case.get("timing") or {}
        lines.append(
            f"| {case.get('id')} | {case.get('variant')} | {det.get('assessment')} | {seg.get('quality')} | "
            f"{det.get('processing_ms')} | {seg.get('processing_ms')} | {timing.get('round_trip_ms')} |"
        )
    lines.extend([
        "",
        "## Qualification boundary",
        "",
        "SAM outputs and heuristic quality labels are candidate evidence only. They do not prove boundary accuracy. Production qualification remains blocked until reviewed ground-truth masks, boundary metrics, hard-case review, and explicit visual sign-off exist.",
        "This run probes the no-traffic candidate revision only. It does not change default Cloud Run traffic.",
        "",
        "## Source attribution",
        "",
    ])
    for source in manifest.get("sources", []):
        lines.append(
            f"- {source['id']}: {source['attribution']}; {source['license']}; "
            f"[source]({source['page_url']}); [license]({source['license_url']})."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "auth_status": "NOT_CHECKED",
        "fixture_download_errors": [],
        "qualification_scope": "CANDIDATE_MASK_AND_API_LATENCY_ONLY",
        "production_mask_qualification": "NOT_QUALIFIED_NO_GROUND_TRUTH_OR_VISUAL_SIGNOFF",
        "expected_source_sha": args.expected_sha,
        "expected_revision": args.expected_revision,
        "base_url": args.base_url.rstrip("/"),
        "cases": [],
        "runtime_binding": {},
        "notes": [
            "No private user photos are included.",
            "Derived variants test low-light/responsive geometry; they are not independent photo scenes.",
            "A heuristic GOOD mask is not a production-quality certification.",
            "Browser optical frame-time benchmarking remains separate from this API latency probe."
        ],
    }

    base_url = args.base_url.rstrip("/")
    status, health, health_ms = request_json(base_url + "/health/deploy", timeout=30)
    report["runtime_binding"] = {
        "http_status": status, "round_trip_ms": health_ms, "payload": {key: health.get(key) for key in ("status", "git_commit", "revision", "service")},
        "matches_expected": status == 200 and health.get("git_commit") == args.expected_sha
        and health.get("revision") == args.expected_revision,
    }
    if not report["runtime_binding"]["matches_expected"]:
        report["status"] = "BLOCKED_RUNTIME_SHA_OR_REVISION_MISMATCH"
        report["error"] = "Health endpoint did not match the exact candidate source SHA and revision."
        write_report(output_dir, report, manifest)
        print(json.dumps({"status": report["status"], "report": str(output_dir / "report.json")}, ensure_ascii=False))
        return 3

    try:
        auth_headers = console_auth_headers()
        report["auth_status"] = "CONFIGURED_NOT_YET_VERIFIED"
    except ValueError:
        report["status"] = "BLOCKED_AUTH_NOT_CONFIGURED"
        report["auth_status"] = "AUTH_NOT_CONFIGURED"
        report["error"] = "SWOS_PROBE_CONSOLE_ACCESS_KEY is not configured; no protected inference request was made."
        write_report(output_dir, report, manifest)
        print(json.dumps({"status": report["status"], "auth_status": report["auth_status"], "report": str(output_dir / "report.json")}, ensure_ascii=False))
        return 4

    transport_errors = 0
    with tempfile.TemporaryDirectory(prefix="yujian-swos-photo-fixtures-") as temp:
        work_dir = Path(temp)
        for spec in manifest.get("sources", []):
            try:
                raw, _, source_digest = download_source(spec, work_dir)
                report.setdefault("source_verification", []).append({
                    "id": spec["id"], "page_url": spec["page_url"], "attribution": spec["attribution"],
                    "license": spec["license"], "source_digest": source_digest,
                })
                variants = make_variants(spec, raw, work_dir)
                for variant in variants:
                    body, content_type = build_multipart(variant["bytes"], variant["upload_filename"], field_name="file")
                    http_status, payload, elapsed_ms = request_json(
                        base_url + "/api/product/catch-recognition/infer",
                        data=body,
                        headers={"Content-Type": content_type, **auth_headers},
                        timeout=180,
                    )
                    case: dict[str, Any] = {
                        "id": variant["id"],
                        "source_id": variant["source_id"],
                        "variant": variant["variant"],
                        "intent": variant["intent"],
                        "source_dimensions": variant["source_dimensions"],
                        "upload_dimensions": None,
                        "source_sha256": source_digest["sha256"],
                        "upload_sha256": digest_bytes(variant["bytes"], "sha256"),
                        "upload_bytes": len(variant["bytes"]),
                        "http_status": http_status,
                        "timing": {"round_trip_ms": elapsed_ms},
                        "status": "HTTP_ERROR",
                    }
                    if http_status in {401, 403}:
                        case["status"] = "AUTH_REJECTED"
                        case["error_category"] = "AUTH_REJECTED"
                        case["error"] = {"http_status": http_status, "detail": payload.get("detail", "protected inference API rejected the configured credential")}
                        report["cases"].append(case)
                        report["status"] = "BLOCKED_AUTH_REJECTED"
                        report["auth_status"] = "AUTH_REJECTED"
                        report["error"] = "The protected inference API rejected the configured probe credential; no valid mask conclusion can be drawn."
                        write_report(output_dir, report, manifest)
                        print(json.dumps({"status": report["status"], "auth_status": report["auth_status"], "report": str(output_dir / "report.json")}, ensure_ascii=False))
                        return 4
                    if http_status != 200:
                        case["status"] = "INFERENCE_HTTP_ERROR"
                        case["error"] = payload
                        transport_errors += 1
                        report["cases"].append(case)
                        continue
                    report["auth_status"] = "AUTHENTICATED"
                    try:
                        verify_same_request_binding(payload, case["upload_sha256"])
                    except ValueError as exc:
                        case["source_binding_error"] = str(exc)
                        transport_errors += 1
                    summary = summarize_payload(payload)
                    summary["runtime"] = {
                        "service": health.get("service"),
                        "revision": args.expected_revision,
                        "commit": args.expected_sha,
                        "detector_model": summary["detector"]["model"],
                        "segmentation_model": summary["segmentation"]["model"],
                        "segmentation_checkpoint": summary["segmentation"]["checkpoint_sha256"],
                    }
                    summary["timing"] = {
                        **summary["timing"],
                        "round_trip_ms": elapsed_ms,
                    }
                    case.update(summary)
                    case["upload_dimensions"] = [
                        summary["input"].get("width"), summary["input"].get("height")
                    ]
                    case["status"] = (
                        "MASK_CANDIDATE"
                        if summary["segmentation"]["executed"]
                        else ("BBOX_ONLY_NO_MASK" if summary["segmentation"]["trust_state"] == "BBOX_ONLY"
                              else "NO_DETECTION_OR_MASK_NOT_EXECUTED")
                    )
                    mask_data = summary.pop("mask_data_base64", None)
                    case.pop("mask_data_base64", None)
                    if summary["segmentation"]["executed"] and mask_data:
                        mask_png = base64.b64decode(mask_data, validate=True)
                        actual_mask_sha = digest_bytes(mask_png, "sha256")
                        if actual_mask_sha != summary["segmentation"]["mask_sha256"]:
                            raise ValueError("SAM mask bytes do not match the returned mask SHA-256")
                        artifacts = write_mask_artifacts(variant["bytes"], mask_png, output_dir, variant["id"])
                        case["segmentation"]["mask_artifacts"] = artifacts
                    report["cases"].append(case)
            except Exception as exc:
                transport_errors += 1
                is_rate_limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
                report["cases"].append({
                    "id": spec.get("id"),
                    "source_id": spec.get("id"),
                    "status": "FIXTURE_RATE_LIMITED" if is_rate_limited else "FIXTURE_OR_PROBE_ERROR",
                    "error_category": "FIXTURE_HTTP_429" if is_rate_limited else type(exc).__name__,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if is_rate_limited:
                    report["fixture_download_errors"].append({"source_id": spec.get("id"), "http_status": 429})

    status_after, health_after, health_after_ms = request_json(base_url + "/health/deploy", timeout=30)
    report["runtime_binding"]["after"] = {
        "http_status": status_after, "round_trip_ms": health_after_ms, "payload": {key: health_after.get(key) for key in ("status", "git_commit", "revision", "service")},
        "matches_expected": status_after == 200 and health_after.get("git_commit") == args.expected_sha
        and health_after.get("revision") == args.expected_revision,
    }
    masks = [case for case in report["cases"] if case.get("segmentation", {}).get("executed")]
    report["mask_candidate_count"] = len(masks)
    report["case_count"] = len(report["cases"])
    report["transport_or_binding_errors"] = transport_errors
    if not report["runtime_binding"]["after"]["matches_expected"]:
        report["status"] = "BLOCKED_RUNTIME_CHANGED_DURING_PROBE"
    elif report.get("auth_status") != "AUTHENTICATED":
        report["status"] = "BLOCKED_NO_AUTHENTICATED_INFERENCE"
    elif transport_errors:
        report["status"] = "BLOCKED_ONE_OR_MORE_CASES_FAILED"
    elif len(masks) < 2:
        report["status"] = "BLOCKED_INSUFFICIENT_MASK_CANDIDATES"
    else:
        report["status"] = "CANDIDATE_OUTPUTS_COLLECTED_NOT_PRODUCTION_QUALIFIED"

    write_report(output_dir, report, manifest)
    print(json.dumps({
        "status": report["status"],
        "cases": report["case_count"],
        "mask_candidates": report["mask_candidate_count"],
        "production_mask_qualification": report["production_mask_qualification"],
        "report": str(output_dir / "report.json"),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))
    return 0 if report["status"] == "CANDIDATE_OUTPUTS_COLLECTED_NOT_PRODUCTION_QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


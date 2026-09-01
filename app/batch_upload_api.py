from __future__ import annotations

import csv
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import zipfile
from pathlib import PurePosixPath
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel, Field
from starlette.requests import Request

from app.factory import IMAGE_EXTS, get_bucket_name
from app.services.manifest_normalizer import (
    ManifestNormalizationError,
    normalize_manifest_text,
    validate_fish_manifest_text,
)

router = APIRouter(tags=["batch-upload"])
templates = Jinja2Templates(directory="app/templates")

MAX_SINGLE_FILE_BYTES = 25 * 1024 * 1024
BATCH_ID_RE = re.compile(r"^BATCH_[A-Za-z0-9_.-]{3,120}$")


class UploadStartRequest(BaseModel):
    batch_id: str | None = Field(default=None, max_length=128)
    source: str = Field(default="other", min_length=1, max_length=64)


class UploadFinalizeRequest(BaseModel):
    batch_id: str = Field(min_length=4, max_length=128)
    source: str = Field(default="other", min_length=1, max_length=64)


def _validate_batch_id(value: str | None) -> str:
    batch_id = (value or "").strip() or f"BATCH_{uuid4().hex[:12].upper()}"
    if not BATCH_ID_RE.fullmatch(batch_id):
        raise ValueError("batch_id 必须以 BATCH_ 开头，且只能包含字母、数字、点、下划线和连字符")
    return batch_id


def _safe_relative_path(value: str) -> str:
    raw = (value or "").strip().replace("\\", "/").lstrip("/")
    if not raw:
        raise ValueError("relative_path 不能为空")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path 非法")
    if path.parts and path.parts[0] == "__MACOSX":
        raise ValueError("忽略 macOS 元数据目录")
    return str(path)


def _build_fish_manifest(source_text: str) -> tuple[str, int]:
    reader = csv.DictReader(io.StringIO(source_text.lstrip("\ufeff")))
    if not reader.fieldnames:
        raise ValueError("metadata/manifest.csv 没有表头")
    fields = list(reader.fieldnames)
    for required in ("image_id", "file_name"):
        if required not in fields:
            raise ValueError(f"metadata/manifest.csv 缺少字段：{required}")

    if "claimed_species" not in fields:
        if "species_name" in fields:
            fields.insert(fields.index("file_name") + 1, "claimed_species")
        else:
            raise ValueError("metadata/manifest.csv 需要 species_name 或 claimed_species 字段")

    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        normalized = {name: (row.get(name) or "").strip() for name in reader.fieldnames}
        if not normalized.get("claimed_species"):
            normalized["claimed_species"] = normalized.get("species_name", "")
        if not normalized.get("image_id") or not normalized.get("file_name") or not normalized.get("claimed_species"):
            raise ValueError(f"metadata/manifest.csv 第 {row_number} 行缺少 image_id / file_name / species")
        rows.append(normalized)

    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), len(rows)


def _prefix(batch_id: str) -> str:
    return f"incoming/{batch_id}/"


def _list_blobs(client: storage.Client, bucket_name: str, batch_id: str) -> list[storage.Blob]:
    return [b for b in client.list_blobs(bucket_name, prefix=_prefix(batch_id)) if not b.name.endswith("/")]


def _payload_md5(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


def _payload_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob_matches_payload(blob: storage.Blob, data: bytes, client: storage.Client) -> bool:
    """Compare an existing object without ever replacing it."""

    try:
        blob.reload(client)
    except Exception:
        pass
    remote_md5 = getattr(blob, "md5_hash", None)
    if remote_md5:
        return remote_md5 == _payload_md5(data)
    if getattr(blob, "size", None) is not None and blob.size != len(data):
        return False
    try:
        remote = blob.download_as_bytes()
    except Exception:
        return False
    return _payload_sha256(remote) == _payload_sha256(data)


def _upload_resumable_blob(
    bucket: storage.Bucket,
    client: storage.Client,
    object_name: str,
    data: bytes,
    *,
    content_type: str,
) -> dict:
    """Upload one object idempotently and report UPLOADED/SKIP/CONFLICT."""

    sha256 = _payload_sha256(data)
    blob = bucket.blob(object_name)
    if blob.exists(client):
        if _blob_matches_payload(blob, data, client):
            return {
                "relative_path": object_name.rsplit("/", 1)[-1],
                "size_bytes": len(data),
                "sha256": sha256,
                "status": "SKIP",
                "skipped": True,
            }
        return {
            "relative_path": object_name.rsplit("/", 1)[-1],
            "size_bytes": len(data),
            "sha256": sha256,
            "status": "CONFLICT",
            "conflict": True,
        }
    try:
        blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
    except Exception:
        # Another worker may have won the race between exists() and the
        # conditional write. Re-check and report an explicit outcome.
        if blob.exists(client):
            if _blob_matches_payload(blob, data, client):
                return {
                    "relative_path": object_name.rsplit("/", 1)[-1],
                    "size_bytes": len(data),
                    "sha256": sha256,
                    "status": "SKIP",
                    "skipped": True,
                }
            return {
                "relative_path": object_name.rsplit("/", 1)[-1],
                "size_bytes": len(data),
                "sha256": sha256,
                "status": "CONFLICT",
                "conflict": True,
            }
        raise
    return {
        "relative_path": object_name.rsplit("/", 1)[-1],
        "size_bytes": len(data),
        "sha256": sha256,
        "status": "UPLOADED",
        "uploaded": True,
    }


def _source_manifest_blob(blobs: list[storage.Blob]) -> storage.Blob | None:
    candidates = [
        b
        for b in blobs
        if b.name.endswith("/metadata/manifest.csv")
        or b.name.endswith("/manifest.csv")
    ]
    candidates.sort(key=lambda b: (0 if b.name.endswith("/metadata/manifest.csv") else 1, len(b.name)))
    return candidates[0] if candidates else None


def _existing_fish_manifest(blobs: list[storage.Blob]) -> storage.Blob | None:
    canonical = [b for b in blobs if b.name.endswith("/metadata/fish_manifest.csv")]
    if canonical:
        return sorted(canonical, key=lambda b: (len(b.name), b.name))[0]
    candidates = [b for b in blobs if b.name.endswith("/fish_manifest.csv")]
    if len(candidates) > 1:
        raise ManifestNormalizationError(f"multiple fish_manifest.csv files found: {len(candidates)}")
    return candidates[0] if candidates else None


def _download_manifest_text(blob: storage.Blob) -> str:
    try:
        return blob.download_as_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestNormalizationError("manifest is not valid UTF-8", source_path=blob.name) from exc


def _ensure_manifest_ready(
    client: storage.Client,
    bucket: storage.Bucket,
    *,
    bucket_name: str,
    incoming_prefix: str,
    blobs: list[storage.Blob],
) -> dict:
    """Validate or materialize the canonical metadata/fish_manifest.csv in GCS."""

    prefix = incoming_prefix.strip("/") + "/"
    existing = _existing_fish_manifest(blobs)
    if existing is not None:
        manifest_text = _download_manifest_text(existing)
        rows = validate_fish_manifest_text(manifest_text, source_name=existing.name)
        return {
            "status": "MANIFEST_READY",
            "manifest_path": existing.name[len(prefix):] if existing.name.startswith(prefix) else existing.name,
            "manifest_rows": rows,
            "generated": False,
        }

    source = _source_manifest_blob(blobs)
    if source is None:
        raise ManifestNormalizationError("missing metadata/manifest.csv")
    source_text = _download_manifest_text(source)
    normalized, rows = normalize_manifest_text(source_text, source_name=source.name)
    output_name = prefix + "metadata/fish_manifest.csv"
    output = bucket.blob(output_name)
    try:
        output.upload_from_string(
            normalized,
            content_type="text/csv; charset=utf-8",
            if_generation_match=0,
        )
    except Exception:
        # A concurrent finalize may have created the canonical file. Never overwrite it;
        # validate and reuse it if it is now present.
        if not output.exists(client):
            raise
        rows = validate_fish_manifest_text(
            _download_manifest_text(output),
            source_name=output_name,
        )
        return {
            "status": "MANIFEST_READY",
            "manifest_path": "metadata/fish_manifest.csv",
            "manifest_rows": rows,
            "generated": False,
        }
    return {
        "status": "MANIFEST_READY",
        "manifest_path": "metadata/fish_manifest.csv",
        "manifest_rows": rows,
        "generated": True,
    }


def ensure_incoming_manifest(incoming_prefix: str, bucket_name: str | None = None) -> dict:
    """Public Batch/Audit fallback that keeps GCS handling out of business logic."""

    bucket_name = bucket_name or get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = incoming_prefix.strip("/") + "/"
    blobs = [b for b in client.list_blobs(bucket_name, prefix=prefix) if not b.name.endswith("/")]
    if not blobs:
        raise ManifestNormalizationError("no objects under incoming batch")
    return _ensure_manifest_ready(
        client,
        bucket,
        bucket_name=bucket_name,
        incoming_prefix=prefix,
        blobs=blobs,
    )


def _manifest_error_response(exc: ManifestNormalizationError) -> JSONResponse:
    return JSONResponse(status_code=400, content=exc.as_dict())


def _finalize_upload(batch_id: str, source: str) -> dict:
    batch_id = _validate_batch_id(batch_id)
    bucket_name = get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = _list_blobs(client, bucket_name, batch_id)
    if not blobs:
        raise ValueError("上传目录为空，请先上传采集数据")

    images = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError("没有发现 jpg/jpeg/png/webp 图片")

    manifest_info = _ensure_manifest_ready(
        client,
        bucket,
        bucket_name=bucket_name,
        incoming_prefix=_prefix(batch_id),
        blobs=blobs,
    )
    generated_manifest = bool(manifest_info["generated"])
    manifest_rows = int(manifest_info["manifest_rows"])
    status_history = ["UPLOADED", "MANIFEST_READY", "READY_FOR_AUDIT"]

    marker = {
        "batch_id": batch_id,
        "source": source,
        "image_count": len(images),
        "manifest_rows": manifest_rows,
        "generated_fish_manifest": generated_manifest,
        "manifest_path": manifest_info["manifest_path"],
        "status": "READY_FOR_AUDIT",
        "status_history": status_history,
    }
    bucket.blob(_prefix(batch_id) + "_upload.json").upload_from_string(
        json.dumps(marker, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

    return {
        "batch_id": batch_id,
        "incoming_prefix": _prefix(batch_id),
        "source": source,
        "uploaded_files": len(blobs),
        "image_count": len(images),
        "manifest_rows": manifest_rows,
        "generated_fish_manifest": generated_manifest,
        "manifest_path": manifest_info["manifest_path"],
        "manifest_status": manifest_info["status"],
        "status_history": status_history,
        "status": "READY_FOR_AUDIT",
    }


@router.get("/batches/upload", response_class=HTMLResponse)
def batch_upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="batch_upload.html", context={})


@router.post("/api/batches/upload-start")
def start_batch_upload(payload: UploadStartRequest):
    try:
        batch_id = _validate_batch_id(payload.batch_id)
        bucket_name = get_bucket_name()
        client = storage.Client()
        existing = next(iter(client.list_blobs(bucket_name, prefix=_prefix(batch_id), max_results=1)), None)
        return {
            "batch_id": batch_id,
            "source": payload.source,
            "status": "RESUME" if existing is not None else "READY_FOR_FILES",
            "resumed": existing is not None,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/batches/upload-file")
async def upload_batch_file(
    file: UploadFile = File(...),
    batch_id: str = Form(...),
    relative_path: str = Form(...),
):
    try:
        batch_id = _validate_batch_id(batch_id)
        relative_path = _safe_relative_path(relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await file.read(MAX_SINGLE_FILE_BYTES + 1)
    if len(data) > MAX_SINGLE_FILE_BYTES:
        raise HTTPException(status_code=413, detail="单文件超过 25 MiB，请先压缩该图片或拆分数据")

    content_type = file.content_type or mimetypes.guess_type(relative_path)[0] or "application/octet-stream"
    try:
        client = storage.Client()
        bucket = client.bucket(get_bucket_name())
        object_name = _prefix(batch_id) + relative_path
        result = _upload_resumable_blob(
            bucket,
            client,
            object_name,
            data,
            content_type=content_type,
        )
        result.update({"batch_id": batch_id, "relative_path": relative_path})
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/batches/upload-finalize")
def finalize_batch_upload(payload: UploadFinalizeRequest):
    try:
        return _finalize_upload(payload.batch_id, payload.source)
    except ManifestNormalizationError as exc:
        return _manifest_error_response(exc)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/batches/upload")
async def upload_batch_dataset(
    file: UploadFile = File(...),
    batch_id: str | None = Form(default=None),
    source: str = Form(default="other"),
):
    """Small-ZIP convenience path.

    Cloud Run has a request-size ceiling, so real collection packages should use the
    folder uploader on /batches/upload, which sends one source file per request.
    """
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="只支持 ZIP；大数据包请使用文件夹上传")

    try:
        final_batch = _validate_batch_id(batch_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    data = await file.read(MAX_SINGLE_FILE_BYTES + 1)
    if len(data) > MAX_SINGLE_FILE_BYTES:
        raise HTTPException(status_code=413, detail="ZIP 超过 25 MiB，请使用 /batches/upload 的文件夹上传模式")

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                raise ValueError("ZIP 为空")
            client = storage.Client()
            bucket = client.bucket(get_bucket_name())
            upload_counts = {"total": 0, "uploaded": 0, "skipped": 0, "conflict": 0, "failed": 0}
            conflicts = []
            for info in members:
                try:
                    name = _safe_relative_path(info.filename)
                except ValueError:
                    if info.filename.replace("\\", "/").startswith("__MACOSX/"):
                        continue
                    raise
                if info.file_size > MAX_SINGLE_FILE_BYTES:
                    raise ValueError(f"ZIP 内单文件超过 25 MiB：{name}")
                upload_counts["total"] += 1
                try:
                    result = _upload_resumable_blob(
                        bucket,
                        client,
                        _prefix(final_batch) + name,
                        archive.read(info),
                        content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                    )
                    outcome = {"SKIP": "skipped"}.get(result["status"], result["status"].lower())
                    upload_counts[outcome] = upload_counts.get(outcome, 0) + 1
                    if result["status"] == "CONFLICT":
                        conflicts.append(name)
                except Exception:
                    upload_counts["failed"] += 1
            if conflicts or upload_counts["failed"]:
                return {
                    "batch_id": final_batch,
                    "source": source,
                    "status": "CONFLICT" if conflicts else "FAILED",
                    "conflicts": conflicts,
                    "upload_summary": upload_counts,
                }
        result = _finalize_upload(final_batch, source)
        result["upload_summary"] = upload_counts
        result.update(
            {
                "uploaded": upload_counts["uploaded"],
                "skipped": upload_counts["skipped"],
                "conflict": upload_counts["conflict"],
                "failed": upload_counts["failed"],
            }
        )
        return result
    except ManifestNormalizationError as exc:
        return _manifest_error_response(exc)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="ZIP 文件损坏或格式不正确") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

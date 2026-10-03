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
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google.cloud import storage
from pydantic import BaseModel, Field
from sqlalchemy import select
from starlette.requests import Request

from app.db import SessionLocal
from app.exact_dedupe import (
    GLOBAL_EXACT_DUPLICATE,
    GlobalExactGuardUnavailable,
    claim_global_image,
    mark_global_image_active,
    mark_global_image_failed,
    sha256_bytes,
)
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
    batch_name: str | None = Field(default=None, max_length=128)


class UploadFinalizeRequest(BaseModel):
    batch_id: str = Field(min_length=4, max_length=128)
    source: str = Field(default="other", min_length=1, max_length=64)
    batch_name: str | None = Field(default=None, max_length=128)


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
    sha256: str | None = None,
) -> dict:
    """Upload one object idempotently and report UPLOADED/SKIP/CONFLICT."""

    sha256 = sha256 or _payload_sha256(data)
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


def _duplicate_paths(batch_id: str) -> set[str]:
    db = SessionLocal()
    try:
        from app.models import GlobalDuplicateAudit

        rows = db.scalars(
            select(GlobalDuplicateAudit).where(
                GlobalDuplicateAudit.incoming_batch_id == batch_id,
                GlobalDuplicateAudit.reason == GLOBAL_EXACT_DUPLICATE,
            )
        ).all()
        return {str(row.incoming_path).replace("\\", "/").lstrip("/") for row in rows}
    finally:
        db.close()


def _manifest_row_path(row: dict[str, str]) -> str:
    for key in ("file_name", "image_path", "filename", "image_name", "relative_path", "file_path", "path"):
        value = (row.get(key) or "").strip()
        if value:
            return value.replace("\\", "/").lstrip("/")
    return ""


def _exclude_duplicate_manifest_rows(source_text: str, batch_id: str) -> tuple[str, int]:
    duplicate_paths = _duplicate_paths(batch_id)
    if not duplicate_paths:
        return source_text, 0
    duplicate_basenames = {PurePosixPath(path).name for path in duplicate_paths}
    reader = csv.DictReader(io.StringIO(source_text.lstrip("\ufeff")))
    if not reader.fieldnames:
        return source_text, 0
    all_rows = list(csv.DictReader(io.StringIO(source_text.lstrip("\ufeff"))))
    rows = [
        row
        for row in all_rows
        if _manifest_row_path(row) not in duplicate_paths
        and PurePosixPath(_manifest_row_path(row)).name not in duplicate_basenames
    ]
    removed = len(all_rows) - len(rows)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(reader.fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue(), removed


def _guarded_image_upload(
    bucket: storage.Bucket,
    client: storage.Client,
    *,
    batch_id: str,
    relative_path: str,
    object_name: str,
    data: bytes,
    content_type: str,
    source: str,
) -> dict:
    """Claim content before GCS upload and finalize the claim after success."""

    digest = sha256_bytes(data)
    db = SessionLocal()
    claim = None
    try:
        claim = claim_global_image(
            db,
            sha256=digest,
            batch_id=batch_id,
            incoming_path=relative_path,
            object_name=object_name,
            source=source,
        )
        if claim.blocked:
            db.commit()
            return {
                "relative_path": relative_path,
                "size_bytes": len(data),
                "sha256": digest,
                "status": "DUPLICATE_BLOCKED",
                "duplicate": True,
                **claim.as_dict(),
            }
        result = _upload_resumable_blob(
            bucket,
            client,
            object_name,
            data,
            content_type=content_type,
            sha256=digest,
        )
        if result.get("status") in {"UPLOADED", "SKIP"}:
            mark_global_image_active(db, sha256=digest, batch_id=batch_id, object_name=object_name)
        else:
            mark_global_image_failed(db, sha256=digest, error=result.get("status", "upload failed"))
        db.commit()
        # Keep the storage outcome authoritative for upload counters.  The
        # registry claim is a separate state machine: a new claim followed by
        # an existing GCS object is a successful reconciliation (SKIP), not a
        # business state named CLAIMED.
        result.update(
            {
                "relative_path": relative_path,
                "claim_status": claim.status,
                "canonical": claim.as_dict()["canonical"],
            }
        )
        return result
    except Exception as exc:
        db.rollback()
        if claim is not None and claim.status == "CLAIMED":
            try:
                mark_global_image_failed(db, sha256=digest, error=str(exc))
                db.commit()
            except Exception:
                db.rollback()
        raise
    finally:
        db.close()


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
    batch_id: str | None = None,
) -> dict:
    """Validate or materialize the canonical metadata/fish_manifest.csv in GCS."""

    prefix = incoming_prefix.strip("/") + "/"
    existing = _existing_fish_manifest(blobs)
    if existing is not None:
        manifest_text = _download_manifest_text(existing)
        if batch_id:
            manifest_text, duplicate_rows_removed = _exclude_duplicate_manifest_rows(manifest_text, batch_id)
        else:
            duplicate_rows_removed = 0
        rows = validate_fish_manifest_text(manifest_text, source_name=existing.name)
        manifest_path = existing.name[len(prefix):] if existing.name.startswith(prefix) else existing.name
        if batch_id and duplicate_rows_removed:
            output_name = prefix + "metadata/fish_manifest.csv"
            output = bucket.blob(output_name)
            upload_kwargs = {}
            if getattr(existing, "generation", None) is not None and existing.name == output_name:
                upload_kwargs["if_generation_match"] = existing.generation
            output.upload_from_string(
                manifest_text,
                content_type="text/csv; charset=utf-8",
                **upload_kwargs,
            )
            manifest_path = "metadata/fish_manifest.csv"
        return {
            "status": "MANIFEST_READY",
            "manifest_path": manifest_path,
            "manifest_rows": rows,
            "generated": False,
            "duplicate_rows_removed": duplicate_rows_removed,
        }

    source = _source_manifest_blob(blobs)
    if source is None:
        raise ManifestNormalizationError("missing metadata/manifest.csv")
    source_text = _download_manifest_text(source)
    if batch_id:
        source_text, duplicate_rows_removed = _exclude_duplicate_manifest_rows(source_text, batch_id)
    else:
        duplicate_rows_removed = 0
    if not source_text.strip() or source_text.count("\n") <= 1:
        raise ManifestNormalizationError("no reviewable images remain after global exact duplicate filtering")
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
            "duplicate_rows_removed": duplicate_rows_removed,
        }
    return {
        "status": "MANIFEST_READY",
        "manifest_path": "metadata/fish_manifest.csv",
        "manifest_rows": rows,
        "generated": True,
        "duplicate_rows_removed": duplicate_rows_removed,
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
        batch_id=prefix.rstrip("/").split("/")[-1],
    )


def _manifest_error_response(exc: ManifestNormalizationError) -> JSONResponse:
    return JSONResponse(status_code=400, content=exc.as_dict())


def _finalize_upload(batch_id: str, source: str, batch_name: str | None = None) -> dict:
    batch_id = _validate_batch_id(batch_id)
    bucket_name = get_bucket_name()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = _list_blobs(client, bucket_name, batch_id)
    if not blobs:
        raise ValueError("上传目录为空，请先上传采集数据")

    images = [b for b in blobs if PurePosixPath(b.name).suffix.lower() in IMAGE_EXTS]
    duplicate_paths = _duplicate_paths(batch_id)
    duplicate_count = len(duplicate_paths)
    if not images:
        if duplicate_count:
            return {
                "batch_id": batch_id,
                "source": source,
                "status": "NO_NEW_IMAGES",
                "input_images": duplicate_count,
                "new_images": 0,
                "duplicates": duplicate_count,
                "skipped": 0,
                "retained": 0,
                "removed": duplicate_count,
                "duplicate_rows_removed": duplicate_count,
            }
        raise ValueError("没有发现 jpg/jpeg/png/webp 图片")

    manifest_info = _ensure_manifest_ready(
        client,
        bucket,
        bucket_name=bucket_name,
        incoming_prefix=_prefix(batch_id),
        blobs=blobs,
        batch_id=batch_id,
    )
    generated_manifest = bool(manifest_info["generated"])
    manifest_rows = int(manifest_info["manifest_rows"])
    status_history = ["UPLOADED", "MANIFEST_READY", "READY_FOR_AUDIT"]
    created_at = datetime.now(timezone.utc).isoformat()
    display_name = (batch_name or "").strip()[:128] or batch_id

    marker = {
        "batch_id": batch_id,
        "batch_name": display_name,
        "source": source,
        "created_at": created_at,
        "image_count": len(images),
        "input_image_count": len(images) + duplicate_count,
        "new_images": len(images),
        "duplicate_count": duplicate_count,
        "duplicates": duplicate_count,
        "skipped": 0,
        "retained": len(images),
        "removed": duplicate_count,
        "manifest_rows": manifest_rows,
        "duplicate_rows_removed": int(manifest_info.get("duplicate_rows_removed", 0)),
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
        "batch_name": display_name,
        "incoming_prefix": _prefix(batch_id),
        "source": source,
        "created_at": created_at,
        "uploaded_files": len(blobs),
        "image_count": len(images),
        "input_image_count": len(images) + duplicate_count,
        "new_images": len(images),
        "duplicates": duplicate_count,
        "skipped": 0,
        "retained": len(images),
        "removed": duplicate_count,
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
            "batch_name": (payload.batch_name or "").strip()[:128] or batch_id,
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
    source: str = Form(default="manual"),
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
        if PurePosixPath(relative_path).suffix.lower() in IMAGE_EXTS:
            result = _guarded_image_upload(
                bucket,
                client,
                batch_id=batch_id,
                relative_path=relative_path,
                object_name=object_name,
                data=data,
                content_type=content_type,
                source=source,
            )
        else:
            result = _upload_resumable_blob(bucket, client, object_name, data, content_type=content_type)
        result.update({"batch_id": batch_id, "relative_path": relative_path})
        return result
    except GlobalExactGuardUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/batches/upload-finalize")
def finalize_batch_upload(payload: UploadFinalizeRequest):
    try:
        return _finalize_upload(payload.batch_id, payload.source, payload.batch_name)
    except ManifestNormalizationError as exc:
        return _manifest_error_response(exc)
    except GlobalExactGuardUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/api/batches/upload")
async def upload_batch_dataset(
    file: UploadFile = File(...),
    batch_id: str | None = Form(default=None),
    source: str = Form(default="other"),
    batch_name: str | None = Form(default=None),
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
            upload_counts = {"total": 0, "uploaded": 0, "skipped": 0, "duplicates": 0, "conflict": 0, "failed": 0}
            conflicts = []
            duplicate_paths = []
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
                    payload = archive.read(info)
                    if PurePosixPath(name).suffix.lower() in IMAGE_EXTS:
                        result = _guarded_image_upload(
                            bucket,
                            client,
                            batch_id=final_batch,
                            relative_path=name,
                            object_name=_prefix(final_batch) + name,
                            data=payload,
                            content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                            source=source,
                        )
                    else:
                        result = _upload_resumable_blob(
                            bucket,
                            client,
                            _prefix(final_batch) + name,
                            payload,
                            content_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        )
                    outcome = {"SKIP": "skipped", "DUPLICATE_BLOCKED": "duplicates"}.get(
                        result["status"], result["status"].lower()
                    )
                    upload_counts[outcome] = upload_counts.get(outcome, 0) + 1
                    if result["status"] == "CONFLICT":
                        conflicts.append(name)
                    elif result["status"] == "DUPLICATE_BLOCKED":
                        duplicate_paths.append(name)
                except GlobalExactGuardUnavailable:
                    raise
                except Exception:
                    upload_counts["failed"] += 1
            if conflicts or upload_counts["failed"]:
                return {
                    "batch_id": final_batch,
                    "source": source,
                    "status": "CONFLICT" if conflicts else "FAILED",
                    "conflicts": conflicts,
                    "duplicates": duplicate_paths,
                    "upload_summary": upload_counts,
                }
        result = _finalize_upload(final_batch, source, batch_name)
        result["upload_summary"] = upload_counts
        result.update(
            {
                "uploaded": upload_counts["uploaded"],
                "skipped": upload_counts["skipped"],
                "conflict": upload_counts["conflict"],
                "failed": upload_counts["failed"],
                "duplicates": upload_counts["duplicates"],
                "duplicate_paths": duplicate_paths,
            }
        )
        return result
    except ManifestNormalizationError as exc:
        return _manifest_error_response(exc)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="ZIP 文件损坏或格式不正确") from exc
    except GlobalExactGuardUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

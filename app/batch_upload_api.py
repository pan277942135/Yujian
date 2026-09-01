import io
import os
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from google.cloud import storage

router = APIRouter()


def bucket_name():
    return os.getenv("GCS_BUCKET", "yujian-model-factory-571785698442")


@router.post("/api/batches/upload")
async def upload_batch_dataset(
    file: UploadFile = File(...),
    batch_id: str | None = Form(default=None),
    source: str = Form(default="other"),
):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="only zip dataset upload is supported")

    final_batch = batch_id or f"BATCH_{uuid4().hex[:12].upper()}"
    prefix = f"incoming/{final_batch}/"

    data = await file.read()
    client = storage.Client()
    bucket = client.bucket(bucket_name())

    image_count = 0
    uploaded = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            if name.startswith("__MACOSX/"):
                continue
            content = archive.read(info)
            target = prefix + name
            bucket.blob(target).upload_from_string(content)
            uploaded.append(target)
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                image_count += 1

    return {
        "batch_id": final_batch,
        "incoming_prefix": prefix,
        "source": source,
        "uploaded_files": len(uploaded),
        "image_count": image_count,
        "status": "READY_FOR_AUDIT",
    }

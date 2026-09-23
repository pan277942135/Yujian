from __future__ import annotations

import io
import json
from types import SimpleNamespace

from fastapi import BackgroundTasks
from PIL import Image, ImageDraw
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.catches_api import create_bside_job, get_bside_status
from app.db import Base
from app.models import AppUser, FishBsideJob, FishCatch
from app.platform.models import BsideBackground, BsideBackgroundOutlineProfile, BsideOutlineStyle
from app.services import fish_bside_jobs


class FakeBlob:
    def __init__(self, name: str):
        self.name = name
        self.data: bytes | None = None

    def exists(self, *_args, **_kwargs):
        return self.data is not None

    def upload_from_string(self, data, **_kwargs):
        self.data = bytes(data)

    def download_as_bytes(self, **_kwargs):
        return self.data or b""


class FakeBucket:
    def __init__(self):
        self.blobs: dict[str, FakeBlob] = {}

    def blob(self, name: str):
        return self.blobs.setdefault(name, FakeBlob(name))


def _png(size: tuple[int, int], mode: str, color) -> bytes:
    image = Image.new(mode, size, color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _fish_png() -> bytes:
    image = Image.new("RGBA", (180, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 15, 150, 65), fill=(170, 190, 188, 255))
    draw.polygon([(140, 40), (176, 10), (176, 70)], fill=(130, 160, 158, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_durable_bside_job_reuses_duplicate_and_runs_existing_pipeline(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'bside.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()
    bucket = FakeBucket()

    class Client:
        def bucket(self, _name):
            return bucket

    monkeypatch.setattr("app.catches_api.storage.Client", Client)
    monkeypatch.setattr("app.catches_api.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.services.fish_bside_jobs.storage.Client", Client)
    monkeypatch.setattr("app.services.fish_bside_jobs.get_bucket_name", lambda: "test-bucket")
    monkeypatch.setattr("app.platform.services.bside_assets.storage.Client", Client)
    monkeypatch.setattr(fish_bside_jobs, "SessionLocal", Session)
    monkeypatch.setattr(
        fish_bside_jobs,
        "process_qwen_output",
        lambda _data: SimpleNamespace(transparent_fish=_fish_png()),
    )

    user = AppUser(id="user-1", username="angler", password_hash="x", nickname="钓友")
    record = FishCatch(
        id="catch-1",
        user_id=user.id,
        image_url="/api/v1/catches/catch-1/media",
        image_object_name="user_catches/user-1/catch.jpg",
        species_id="carp",
        species_name="鲤鱼",
        confidence=0.9,
        model_version="test",
    )
    background = BsideBackground(
        code="test_lake",
        name="测试湖面",
        background_uri="gs://test-bucket/bside-assets/test_lake/background.webp",
        foreground_uri="gs://test-bucket/bside-assets/test_lake/foreground.png",
        light_uri="gs://test-bucket/bside-assets/test_lake/light.png",
        status="ACTIVE",
        fish_anchor_x=0.5,
        fish_anchor_y=0.5,
        fish_width_min=0.68,
        fish_width_max=0.74,
    )
    outline_style = BsideOutlineStyle(code="none", name="原生", status="ACTIVE")
    db.add_all([user, record, background, outline_style])
    db.commit()
    db.add(BsideBackgroundOutlineProfile(background_id=background.id, outline_style_id=outline_style.id, enabled=True, weight=100))
    db.commit()
    bucket.blob(record.image_object_name).upload_from_string(_png((300, 160), "RGB", (90, 120, 110)))
    bucket.blob("bside-assets/test_lake/background.webp").upload_from_string(_png((1080, 1350), "RGB", (68, 110, 120)))
    bucket.blob("bside-assets/test_lake/foreground.png").upload_from_string(_png((1080, 1350), "RGBA", (0, 0, 0, 0)))
    bucket.blob("bside-assets/test_lake/light.png").upload_from_string(_png((1080, 1350), "RGBA", (255, 255, 255, 0)))

    tasks = BackgroundTasks()
    first = create_bside_job(record.id, tasks, user, db)
    second = create_bside_job(record.id, BackgroundTasks(), user, db)
    assert first.status == second.status == "GENERATING"
    assert first.job_id == second.job_id
    assert len(tasks.tasks) == 1

    fish_bside_jobs.process_fish_bside_job(first.job_id)
    db.expire_all()
    row = db.get(FishCatch, record.id)
    job = db.get(FishBsideJob, first.job_id)
    assert row.bside_status == "READY"
    assert job.status == "SUCCESS"
    assert job.pose_metadata_json
    assert json.loads(job.pose_metadata_json)["mode"] == "RGBA"
    assert job.result_object_name
    assert bucket.blob(job.result_object_name).data
    metadata_object_name = job.result_object_name.replace("bside_result.png", "standardized_fish_metadata.json")
    assert bucket.blob(metadata_object_name).data
    status = get_bside_status(record.id, user, db)
    assert status.status == "READY"
    assert status.result_uri.endswith("/bside-media")

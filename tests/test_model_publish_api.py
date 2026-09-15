from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from app.db import Base
from app.model_publish_api import PublishCallback, publish_callback, publish_model, publish_status
from app.models import DatasetVersion, ModelPublishJob, ModelVersion, TrainingRun


def _session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'publish.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _model(db, version="MODEL_CROP_M1_v0.1"):
    db.add(
        DatasetVersion(
            dataset_version="DS_CROP_M1_v0.1",
            manifest_uri="gs://bucket/datasets/manifest.csv",
            train_count=10,
            val_count=2,
            test_count=2,
            species_count=2,
            git_commit="abc",
            status="FROZEN",
            pipeline_type="CROP_CLASSIFIER_V1",
        )
    )
    db.add(
        TrainingRun(
            run_id="RUN_CROP_M1_v0.1",
            dataset_version="DS_CROP_M1_v0.1",
            git_commit="abc",
            model_family="mobilenet_v3_small",
            params_json=f'{{"model_version":"{version}"}}',
            status="COMPLETED",
            artifact_uri=f"gs://bucket/models/{version}/model_torchscript.pt",
            metrics_uri=f"gs://bucket/models/{version}/metrics.json",
            pipeline_type="CROP_CLASSIFIER_V1",
        )
    )
    db.add(
        ModelVersion(
            model_version=version,
            run_id="RUN_CROP_M1_v0.1",
            artifact_uri=f"gs://bucket/models/{version}/model_torchscript.pt",
            metrics_uri=f"gs://bucket/models/{version}/metrics.json",
            status="CANDIDATE",
            pipeline_type="CROP_CLASSIFIER_V1",
            dataset_version="DS_CROP_M1_v0.1",
        )
    )
    db.commit()


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "server": ("console.example", 443),
            "path": "/api/models/MODEL_CROP_M1_v0.1/publish",
            "headers": [(b"host", b"console.example")],
        }
    )


def test_publish_dispatch_is_idempotent(monkeypatch, tmp_path):
    db = _session(tmp_path)
    _model(db)
    calls = []
    monkeypatch.setattr("app.model_publish_api._gcs_exists", lambda _uri: True)
    monkeypatch.setattr("app.model_publish_api._github_dispatch", lambda inputs: calls.append(inputs))

    first = publish_model("MODEL_CROP_M1_v0.1", _request(), db)
    second = publish_model("MODEL_CROP_M1_v0.1", _request(), db)

    assert first["status"] == "CONVERTING"
    assert second["already_running"] is True
    assert first["publish_job_id"] == second["publish_job_id"]
    assert len(calls) == 1
    assert calls[0]["model_prefix"] == "gs://bucket/models/MODEL_CROP_M1_v0.1"
    assert calls[0]["callback_url"] == "https://console.example/api/model-publish/callback"
    assert calls[0]["callback_token"]
    assert db.query(ModelPublishJob).count() == 1


def test_success_callback_switches_production_only_after_release(monkeypatch, tmp_path):
    db = _session(tmp_path)
    _model(db)
    monkeypatch.setattr("app.model_publish_api._gcs_exists", lambda _uri: True)
    dispatched = []
    monkeypatch.setattr("app.model_publish_api._github_dispatch", lambda inputs: dispatched.append(inputs))
    job = publish_model("MODEL_CROP_M1_v0.1", _request(), db)

    result = publish_callback(
        PublishCallback(
            publish_job_id=job["publish_job_id"],
            model_version="MODEL_CROP_M1_v0.1",
            status="PUBLISHED",
            stage="COMPLETE",
            sha256="a" * 64,
            github_release_url="https://github.com/pan277942135/Yujian/releases/tag/mobile-model-v0.2",
        ),
        dispatched[0]["callback_token"],
        db,
    )

    assert result["status"] == "PUBLISHED"
    assert db.get(ModelVersion, "MODEL_CROP_M1_v0.1").is_production is True
    assert db.get(ModelPublishJob, job["publish_job_id"]).active_lock is None


def test_dispatch_failure_does_not_promote_model(monkeypatch, tmp_path):
    db = _session(tmp_path)
    _model(db)
    monkeypatch.setattr("app.model_publish_api._gcs_exists", lambda _uri: True)
    monkeypatch.setattr(
        "app.model_publish_api._github_dispatch",
        lambda _inputs: (_ for _ in ()).throw(RuntimeError("GITHUB_AUTH_FAILED: denied")),
    )

    try:
        publish_model("MODEL_CROP_M1_v0.1", _request(), db)
        assert False, "expected HTTPException"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 502

    job = db.query(ModelPublishJob).one()
    assert job.status == "FAILED"
    assert job.error_code == "GITHUB_AUTH_FAILED"
    assert db.get(ModelVersion, "MODEL_CROP_M1_v0.1").is_production is False
    assert publish_status("MODEL_CROP_M1_v0.1", db)["status"] == "FAILED"

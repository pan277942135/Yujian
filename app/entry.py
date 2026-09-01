import os

from fastapi import HTTPException

from app.detector_runtime import load_detector
from app.main import app, templates as main_templates
from app.presence import router as presence_router
from app.dedupe import router as dedupe_router
from app.bulk_review import router as bulk_review_router, templates as bulk_review_templates
from app.inspect import router as inspect_router, templates as inspect_templates
from app.dataset_api import router as dataset_freeze_router
from app.training_api import router as training_router, templates as training_templates
from app.inference_api import router as inference_router, templates as inference_templates
from app.feedback_ingest_api import router as feedback_ingest_router
from app.batch_upload_api import router as batch_upload_router, templates as batch_upload_templates
from app.p0_automation import install_feedback_automation, router as automation_router
from app.unified_nav import install_unified_nav
from app.db import SessionLocal
from app.species_policy import ensure_target_species


for template_engine in (
    main_templates,
    bulk_review_templates,
    inspect_templates,
    training_templates,
    inference_templates,
    batch_upload_templates,
):
    install_unified_nav(template_engine)

install_feedback_automation(app)


@app.on_event("startup")
def seed_target_species_catalog() -> None:
    db = SessionLocal()
    try:
        ensure_target_species(db)
    finally:
        db.close()


@app.get("/health/deploy")
def deployment_health() -> dict:
    feedback_ingest_key_configured = bool(os.getenv("FEEDBACK_INGEST_KEY", "").strip())
    return {
        "status": "ok",
        "version": app.version,
        "git_commit": os.getenv("APP_GIT_COMMIT", "unknown"),
        "revision": os.getenv("K_REVISION", "unknown"),
        "service": os.getenv("K_SERVICE", "unknown"),
        "feedback_ingest_path": "/api/feedback/ingest",
        "feedback_ingest_key_configured": feedback_ingest_key_configured,
    }


@app.get("/health/detector")
def detector_health() -> dict:
    try:
        detector = load_detector()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="production detector is unavailable") from exc
    return {
        "status": "ok",
        "model_version": detector.model_version,
        "onnx_sha256": detector.onnx_sha256,
        "onnx_bytes": detector.onnx_bytes,
        "input_size": detector.input_size,
    }


app.include_router(presence_router)
app.include_router(dedupe_router)
app.include_router(bulk_review_router)
app.include_router(inspect_router)
app.include_router(dataset_freeze_router)
app.include_router(training_router)
app.include_router(inference_router)
app.include_router(feedback_ingest_router)
app.include_router(batch_upload_router)
app.include_router(automation_router)

from app.main import app, templates as main_templates
from app.presence import router as presence_router
from app.dedupe import router as dedupe_router
from app.bulk_review import router as bulk_review_router, templates as bulk_review_templates
from app.inspect import router as inspect_router, templates as inspect_templates
from app.dataset_api import router as dataset_freeze_router
from app.training_api import router as training_router, templates as training_templates
from app.inference_api import router as inference_router, templates as inference_templates
from app.unified_nav import install_unified_nav


for template_engine in (
    main_templates,
    bulk_review_templates,
    inspect_templates,
    training_templates,
    inference_templates,
):
    install_unified_nav(template_engine)

app.include_router(presence_router)
app.include_router(dedupe_router)
app.include_router(bulk_review_router)
app.include_router(inspect_router)
app.include_router(dataset_freeze_router)
app.include_router(training_router)
app.include_router(inference_router)

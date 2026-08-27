from app.main import app
from app.presence import router as presence_router
from app.dedupe import router as dedupe_router
from app.bulk_review import router as bulk_review_router
from app.inspect import router as inspect_router

app.include_router(presence_router)
app.include_router(dedupe_router)
app.include_router(bulk_review_router)
app.include_router(inspect_router)

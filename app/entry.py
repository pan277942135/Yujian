from app.main import app
from app.presence import router as presence_router

app.include_router(presence_router)

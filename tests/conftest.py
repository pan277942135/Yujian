import os

os.environ["REGISTRY_DB_URL"] = "sqlite:///:memory:"

import pytest

from app.db import Base, SessionLocal, engine
from app import models  # noqa: F401
from app.presence import FishPresenceResult  # noqa: F401


@pytest.fixture
def db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

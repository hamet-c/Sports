"""Test fixtures: in-memory SQLite DB with all schema created."""
import os
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


@pytest.fixture
def db() -> Generator[Session, None, None]:
    # Force in-memory DB for tests, regardless of .env.
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    # Re-import after env override.
    from app.db.session import Base
    from app.db import models  # noqa: F401  - register models on Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    Session_ = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = Session_()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()

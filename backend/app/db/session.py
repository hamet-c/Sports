"""SQLAlchemy session management."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings


def _resolve_url_and_args() -> tuple[str, dict]:
    if settings.use_turso:
        if not (settings.turso_database_url and settings.turso_auth_token):
            raise RuntimeError(
                "USE_TURSO is set but TURSO_DATABASE_URL/TURSO_AUTH_TOKEN are missing"
            )
        host = settings.turso_database_url.replace("libsql://", "", 1)
        url = f"sqlite+libsql://{host}?authToken={settings.turso_auth_token}&secure=true"
        return url, {}
    if settings.database_url.startswith("sqlite"):
        return settings.database_url, {"check_same_thread": False}
    return settings.database_url, {}


_url, _connect_args = _resolve_url_and_args()

engine = create_engine(
    _url,
    connect_args=_connect_args,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables. Imports models so they register on Base.metadata."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)

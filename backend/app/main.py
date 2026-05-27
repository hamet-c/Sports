"""
Application entry point.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1 import router as api_v1_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.session import init_db
from app.models.registry import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(level=settings.log_level)
    logger.info("Starting up NBA Props API")
    init_db()
    registry.load()
    logger.info(f"Loaded {len(registry)} models: {registry.all_stats()}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="NBA Player Props Prediction Engine",
    description=(
        "Predicts player stat distributions and identifies edges vs. sportsbook lines."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "nba-props", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": len(registry), "stats": registry.all_stats()}

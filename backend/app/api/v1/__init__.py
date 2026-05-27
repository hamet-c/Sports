from fastapi import APIRouter

from app.api.v1.performance import router as performance_router
from app.api.v1.players import router as players_router
from app.api.v1.predictions import router as predictions_router
from app.api.v1.slate import router as slate_router

router = APIRouter()
router.include_router(predictions_router, prefix="/predictions", tags=["predictions"])
router.include_router(players_router, prefix="/players", tags=["players"])
router.include_router(slate_router, prefix="/slate", tags=["slate"])
router.include_router(performance_router, prefix="/performance", tags=["performance"])

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Game
from app.db.session import get_db
from app.models.registry import registry
from app.services.prediction_service import PredictionService

router = APIRouter()


class PredictionRequest(BaseModel):
    player_id: int
    game_id: int
    stat_types: list[str] | None = None
    as_of: date | None = None


class StatPrediction(BaseModel):
    stat_type: str
    predicted_mean: float
    quantiles: dict[float, float]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    player_id: int
    game_id: int
    as_of: date
    predictions: list[StatPrediction]


class EdgeRequest(BaseModel):
    player_id: int
    game_id: int
    stat_type: str
    line: float
    over_odds: int
    under_odds: int
    book: str = "user"
    as_of: date | None = None


class EdgeResponse(BaseModel):
    player_id: int
    game_id: int
    stat_type: str
    line: float
    book: str
    predicted_mean: float
    over_probability: float
    under_probability: float
    raw_over_probability: float
    expected_value_over: float
    expected_value_under: float
    kelly_over: float
    kelly_under: float
    recommendation: str


def _resolve_as_of(db: Session, game_id: int, as_of: date | None) -> date:
    if as_of is not None:
        return as_of
    game = db.query(Game).filter(Game.id == game_id).one_or_none()
    if game is None:
        raise HTTPException(404, f"Game {game_id} not found")
    return game.game_date


@router.post("/", response_model=PredictionResponse)
def predict(req: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    if len(registry) == 0:
        raise HTTPException(503, "No models registered. Train models first.")
    as_of = _resolve_as_of(db, req.game_id, req.as_of)
    svc = PredictionService(
        db, registry,
        edge_threshold=settings.edge_threshold,
        over_edge_threshold=settings.edge_threshold_over,
    )
    _, dists = svc.predict_player_game(
        req.player_id, req.game_id, as_of, stat_types=req.stat_types,
    )
    return PredictionResponse(
        player_id=req.player_id,
        game_id=req.game_id,
        as_of=as_of,
        predictions=[
            StatPrediction(
                stat_type=stat,
                predicted_mean=dist.mean,
                quantiles=dist.quantiles,
            )
            for stat, dist in dists.items()
        ],
    )


@router.post("/edge", response_model=EdgeResponse)
def predict_edge(req: EdgeRequest, db: Session = Depends(get_db)) -> EdgeResponse:
    if req.stat_type not in registry:
        raise HTTPException(404, f"No trained model for stat '{req.stat_type}'")
    as_of = _resolve_as_of(db, req.game_id, req.as_of)
    svc = PredictionService(
        db, registry,
        edge_threshold=settings.edge_threshold,
        over_edge_threshold=settings.edge_threshold_over,
    )
    edge = svc.analyze_prop(
        player_id=req.player_id,
        game_id=req.game_id,
        as_of=as_of,
        stat_type=req.stat_type,
        line=req.line,
        over_odds=req.over_odds,
        under_odds=req.under_odds,
        book=req.book,
    )
    if edge is None:
        raise HTTPException(500, "Failed to compute edge")
    return EdgeResponse(
        player_id=edge.player_id,
        game_id=edge.game_id,
        stat_type=edge.stat_type,
        line=edge.line,
        book=edge.book,
        predicted_mean=edge.predicted_mean,
        over_probability=edge.over_probability,
        under_probability=edge.under_probability,
        raw_over_probability=edge.raw_over_probability,
        expected_value_over=edge.expected_value_over,
        expected_value_under=edge.expected_value_under,
        kelly_over=edge.kelly_over,
        kelly_under=edge.kelly_under,
        recommendation=edge.recommendation,
    )

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session, joinedload

from app.db.models import Game, Player, PlayerGameStats
from app.db.session import get_db
from app.models.registry import registry
from app.services.prediction_service import PredictionService

router = APIRouter()


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    full_name: str
    position: str | None
    team_id: int | None


class PlayerGameStatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    game_date: str
    minutes: float | None
    points: int | None
    rebounds: int | None
    assists: int | None
    threes_made: int | None
    is_home: bool | None


@router.get("/", response_model=list[PlayerOut])
def list_players(
    name_contains: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Player]:
    q = db.query(Player).filter(Player.is_active.is_(True))
    if name_contains:
        q = q.filter(Player.full_name.ilike(f"%{name_contains}%"))
    return q.order_by(Player.full_name).limit(limit).all()


@router.get("/{player_id}", response_model=PlayerOut)
def get_player(player_id: int, db: Session = Depends(get_db)) -> Player:
    player = db.query(Player).filter(Player.id == player_id).one_or_none()
    if player is None:
        raise HTTPException(404, "Player not found")
    return player


@router.get("/{player_id}/recent", response_model=list[PlayerGameStatOut])
def recent_games(
    player_id: int,
    limit: int = Query(15, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[PlayerGameStatOut]:
    rows = (
        db.query(PlayerGameStats)
        .filter(PlayerGameStats.player_id == player_id)
        .order_by(PlayerGameStats.game_date.desc())
        .limit(limit)
        .all()
    )
    return [
        PlayerGameStatOut(
            game_date=r.game_date.isoformat(),
            minutes=r.minutes,
            points=r.points,
            rebounds=r.rebounds,
            assists=r.assists,
            threes_made=r.threes_made,
            is_home=r.is_home,
        )
        for r in rows
    ]


class StatComparison(BaseModel):
    actual: float | None
    predicted: float
    error: float | None  # signed: predicted - actual; None if no actual


class RecentPredictionComparison(BaseModel):
    game_id: int
    game_date: str
    opponent_abbr: str | None
    is_home: bool | None
    minutes: float | None
    stats: dict[str, StatComparison]


@router.get(
    "/{player_id}/predictions_vs_actual",
    response_model=list[RecentPredictionComparison],
)
def predictions_vs_actual(
    player_id: int,
    limit: int = Query(3, ge=1, le=10),
    db: Session = Depends(get_db),
) -> list[RecentPredictionComparison]:
    """For each of this player's last `limit` completed games, run the model
    as-of the game date (no leakage) and compare to the actual result.

    Predictions are computed on demand — they reflect the *current* trained
    model's view of the historical situation, not whatever was in production
    on the night of the game.
    """
    if db.query(Player).filter(Player.id == player_id).one_or_none() is None:
        raise HTTPException(404, "Player not found")

    rows = (
        db.query(PlayerGameStats)
        .options(joinedload(PlayerGameStats.game))
        .filter(PlayerGameStats.player_id == player_id)
        .order_by(PlayerGameStats.game_date.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return []

    # Bulk-load opponent team abbreviations for the games involved.
    opp_ids = {r.opponent_team_id for r in rows if r.opponent_team_id is not None}
    from app.db.models import Team
    opp_abbr = {
        t.id: t.abbreviation
        for t in db.query(Team).filter(Team.id.in_(opp_ids)).all()
    } if opp_ids else {}

    svc = PredictionService(db, registry)
    out: list[RecentPredictionComparison] = []
    stat_keys = registry.all_stats()

    for r in rows:
        _, dists = svc.predict_player_game(
            r.player_id, r.game_id, r.game_date, stat_types=stat_keys,
        )
        stats: dict[str, StatComparison] = {}
        for stat in stat_keys:
            dist = dists.get(stat)
            if dist is None:
                continue
            actual_raw = getattr(r, stat, None)
            actual = float(actual_raw) if actual_raw is not None else None
            predicted = float(dist.mean)
            stats[stat] = StatComparison(
                actual=actual,
                predicted=predicted,
                error=(predicted - actual) if actual is not None else None,
            )
        out.append(RecentPredictionComparison(
            game_id=r.game_id,
            game_date=r.game_date.isoformat(),
            opponent_abbr=opp_abbr.get(r.opponent_team_id) if r.opponent_team_id else None,
            is_home=r.is_home,
            minutes=r.minutes,
            stats=stats,
        ))
    return out

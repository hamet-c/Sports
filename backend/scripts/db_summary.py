"""Quick DB summary — date range and row counts per table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func

from app.db.models import (
    Game,
    InjuryReport,
    Player,
    PlayerGameStats,
    PropLine,
    Team,
    TeamGameStats,
)
from app.db.session import SessionLocal


def main() -> None:
    db = SessionLocal()
    try:
        for model in (Team, Player, Game, PlayerGameStats, TeamGameStats, PropLine, InjuryReport):
            n = db.query(func.count(model.id)).scalar()
            print(f"{model.__tablename__:24s} {n:>8} rows")

        date_col = PlayerGameStats.game_date
        lo, hi = db.query(func.min(date_col), func.max(date_col)).first()
        print(f"\nplayer_game_stats date range: {lo}  ->  {hi}")

        # Per-season counts (derived from game.season).
        per_season = (
            db.query(Game.season, func.count(Game.id))
            .group_by(Game.season)
            .order_by(Game.season)
            .all()
        )
        print("\nGames per season:")
        for season, n in per_season:
            print(f"  {season}: {n}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

"""
Merge stub Games (nba_id IS NULL, created by prop ingest before the real game
log arrived) into their real twins matched on (game_date, home, away).

For each stub/real pair: re-point prop_lines, game_markets and predictions
from the stub to the real game, then delete the stub. Stubs never carry
player/team stats (those are keyed to nba_id games by bootstrap), but the
script verifies that before deleting and skips the pair loudly if violated.

Default is dry-run (prints planned merges, writes nothing). Pass --apply to
execute everything in one transaction.

    python scripts/reconcile_stub_games.py            # dry run
    python scripts/reconcile_stub_games.py --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from sqlalchemy import and_
from sqlalchemy.orm import aliased

from app.core.logging import configure_logging
from app.db.models import Game, GameMarket, PlayerGameStats, Prediction, PropLine, TeamGameStats
from app.db.session import SessionLocal, init_db


def find_twins(db) -> list[tuple[Game, Game]]:
    """(stub, real) pairs sharing (game_date, home_team_id, away_team_id)."""
    real = aliased(Game)
    rows = (
        db.query(Game, real)
        .join(
            real,
            and_(
                Game.game_date == real.game_date,
                Game.home_team_id == real.home_team_id,
                Game.away_team_id == real.away_team_id,
            ),
        )
        .filter(Game.nba_id.is_(None))
        .filter(real.nba_id.isnot(None))
        .all()
    )
    return rows


def merge_pair(db, stub: Game, real: Game) -> dict[str, int]:
    counts = {
        "prop_lines": 0,
        "game_markets": 0,
        "predictions": 0,
        "game_markets_dropped": 0,
    }

    # game_markets has UNIQUE(game_id, book, captured_at): drop stub rows that
    # would collide with an existing real-game row before re-pointing.
    stub_markets = db.query(GameMarket).filter(GameMarket.game_id == stub.id).all()
    for m in stub_markets:
        collision = (
            db.query(GameMarket)
            .filter(GameMarket.game_id == real.id)
            .filter(GameMarket.book == m.book)
            .filter(GameMarket.captured_at == m.captured_at)
            .first()
        )
        if collision is not None:
            db.delete(m)
            counts["game_markets_dropped"] += 1
        else:
            m.game_id = real.id
            counts["game_markets"] += 1

    counts["prop_lines"] = (
        db.query(PropLine)
        .filter(PropLine.game_id == stub.id)
        .update({PropLine.game_id: real.id}, synchronize_session=False)
    )
    counts["predictions"] = (
        db.query(Prediction)
        .filter(Prediction.game_id == stub.id)
        .update({Prediction.game_id: real.id}, synchronize_session=False)
    )
    db.delete(stub)
    return counts


def main(apply: bool) -> None:
    init_db()
    db = SessionLocal()
    try:
        twins = find_twins(db)
        if not twins:
            logger.info("No stub/real twins found — nothing to reconcile.")
            return

        merged = 0
        for stub, real in twins:
            n_pgs = db.query(PlayerGameStats).filter(PlayerGameStats.game_id == stub.id).count()
            n_tgs = db.query(TeamGameStats).filter(TeamGameStats.game_id == stub.id).count()
            if n_pgs or n_tgs:
                logger.error(
                    f"Stub game id={stub.id} ({stub.game_date} h={stub.home_team_id} "
                    f"a={stub.away_team_id}) unexpectedly has stats rows "
                    f"(pgs={n_pgs}, tgs={n_tgs}) — skipping this pair, investigate."
                )
                continue

            n_props = db.query(PropLine).filter(PropLine.game_id == stub.id).count()
            n_markets = db.query(GameMarket).filter(GameMarket.game_id == stub.id).count()
            n_preds = db.query(Prediction).filter(Prediction.game_id == stub.id).count()
            logger.info(
                f"{'MERGE' if apply else 'WOULD MERGE'} stub id={stub.id} -> real "
                f"id={real.id} nba_id={real.nba_id} ({stub.game_date}): "
                f"{n_props} prop_lines, {n_markets} game_markets, {n_preds} predictions"
            )
            if apply:
                counts = merge_pair(db, stub, real)
                if counts["game_markets_dropped"]:
                    logger.warning(
                        f"  dropped {counts['game_markets_dropped']} colliding game_markets rows"
                    )
            merged += 1

        if apply:
            db.commit()
            logger.info(f"Reconciled {merged} stub game(s).")
        else:
            db.rollback()
            logger.info(f"DRY RUN — {merged} pair(s) would merge. Pass --apply to execute.")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Execute the merges (default: dry run).")
    args = parser.parse_args()
    main(args.apply)

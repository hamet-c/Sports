"""
Grade ungraded RecommendationLog rows against actual PlayerGameStats.

Idempotent: only rows with graded_at IS NULL and game_date strictly before
today are touched; rows whose actuals aren't in the DB yet are left ungraded
for the next run. OVER/UNDER rows get result WIN/LOSS/PUSH; PASS rows get
actual_value only (their hypothetical result under any threshold is
re-derivable from the stored p_over/EVs).

    python scripts/grade_recommendations.py
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from app.core.logging import configure_logging
from app.core.timeutil import utcnow
from app.db.models import Game, PlayerGameStats, RecommendationLog
from app.db.session import SessionLocal, init_db


def _result(recommendation: str, actual: float, line: float) -> str | None:
    if actual == line:
        return "PUSH"
    over_hit = actual > line
    if recommendation == "OVER":
        return "WIN" if over_hit else "LOSS"
    if recommendation == "UNDER":
        return "LOSS" if over_hit else "WIN"
    return None  # PASS


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        today = date.today()
        pending = (
            db.query(RecommendationLog)
            .filter(RecommendationLog.graded_at.is_(None))
            .filter(RecommendationLog.game_date < today)
            .all()
        )
        if not pending:
            logger.info("No ungraded recommendations — nothing to do.")
            return

        # Stub->real reconciliation by (date, home, away): a rec may have been
        # logged against a prop-ingest stub game whose actuals live on the
        # real (nba_id) twin. Same pattern as slate.recommendation_record.
        game_ids = {r.game_id for r in pending}
        games = db.query(Game).filter(Game.id.in_(game_ids)).all()
        stub_games = [g for g in games if g.nba_id is None]
        stub_to_real: dict[int, int] = {}
        if stub_games:
            match_keys = {(g.game_date, g.home_team_id, g.away_team_id) for g in stub_games}
            reals = (
                db.query(Game)
                .filter(Game.nba_id.isnot(None))
                .filter(Game.game_date.in_({k[0] for k in match_keys}))
                .all()
            )
            by_match = {(g.game_date, g.home_team_id, g.away_team_id): g.id for g in reals}
            for g in stub_games:
                real_id = by_match.get((g.game_date, g.home_team_id, g.away_team_id))
                if real_id is not None:
                    stub_to_real[g.id] = real_id

        real_ids = {stub_to_real.get(gid, gid) for gid in game_ids}
        pgs_rows = (
            db.query(PlayerGameStats)
            .filter(PlayerGameStats.game_id.in_(real_ids))
            .all()
        )
        actuals: dict[tuple[int, int], PlayerGameStats] = {
            (r.player_id, r.game_id): r for r in pgs_rows
        }

        now = utcnow()
        graded = wins = losses = pushes = no_actual = 0
        for rec in pending:
            real_game_id = stub_to_real.get(rec.game_id, rec.game_id)
            row = actuals.get((rec.player_id, real_game_id))
            actual = getattr(row, rec.stat_type, None) if row is not None else None
            if actual is None:
                no_actual += 1
                continue
            rec.actual_value = float(actual)
            rec.result = _result(rec.recommendation, float(actual), rec.line)
            rec.graded_at = now
            graded += 1
            if rec.result == "WIN":
                wins += 1
            elif rec.result == "LOSS":
                losses += 1
            elif rec.result == "PUSH":
                pushes += 1

        db.commit()
        rate = f"{wins / (wins + losses):.1%}" if (wins + losses) else "n/a"
        logger.info(
            f"Graded {graded} rows: {wins}W-{losses}L-{pushes}P "
            f"(win rate {rate} on decided OVER/UNDER); "
            f"{no_actual} left ungraded (no actuals yet)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    argparse.ArgumentParser().parse_args()
    main()

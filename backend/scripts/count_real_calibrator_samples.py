"""
Quick count of PropLine rows with completed actuals, per stat.

Tells us whether we have enough sample to fit per-stat real-line calibrators
without overfitting the isotonic curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Game, PlayerGameStats, PropLine
from app.db.session import SessionLocal, init_db


STATS = ("points", "rebounds", "assists", "threes_made")


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        # Build the same stub->real game-id reconciliation slate.py uses.
        games = db.query(Game).all()
        by_match: dict[tuple, int] = {}
        for g in games:
            if g.nba_id is not None:
                by_match[(g.game_date, g.home_team_id, g.away_team_id)] = g.id
        stub_to_real: dict[int, int] = {}
        for g in games:
            if g.nba_id is None:
                real_id = by_match.get((g.game_date, g.home_team_id, g.away_team_id))
                if real_id is not None:
                    stub_to_real[g.id] = real_id

        prop_rows = db.query(PropLine).all()
        # Dedupe to latest line per (player, game, stat, book) — matches the
        # graders elsewhere in the code so the count reflects what we'd
        # actually use in the calibrator fit.
        latest: dict[tuple, PropLine] = {}
        for p in prop_rows:
            key = (p.player_id, p.game_id, p.stat_type, p.book)
            if key not in latest or p.captured_at > latest[key].captured_at:
                latest[key] = p

        # Bulk-load actuals for any real game id that might be referenced.
        all_game_ids = {p.game_id for p in latest.values()}
        real_game_ids = list({stub_to_real.get(gid, gid) for gid in all_game_ids})
        actuals_rows = (
            db.query(PlayerGameStats)
            .filter(PlayerGameStats.game_id.in_(real_game_ids))
            .all()
        )
        actuals: dict[tuple[int, int], PlayerGameStats] = {
            (r.player_id, r.game_id): r for r in actuals_rows
        }

        per_stat_total: dict[str, int] = {s: 0 for s in STATS}
        per_stat_graded: dict[str, int] = {s: 0 for s in STATS}
        per_stat_by_date: dict[str, dict[str, int]] = {s: {} for s in STATS}

        for prop in latest.values():
            if prop.stat_type not in STATS:
                continue
            per_stat_total[prop.stat_type] += 1
            real_game_id = stub_to_real.get(prop.game_id, prop.game_id)
            actual_row = actuals.get((prop.player_id, real_game_id))
            if actual_row is None:
                continue
            if getattr(actual_row, prop.stat_type, None) is None:
                continue
            per_stat_graded[prop.stat_type] += 1
            date_key = actual_row.game_date.isoformat()
            per_stat_by_date[prop.stat_type][date_key] = (
                per_stat_by_date[prop.stat_type].get(date_key, 0) + 1
            )

        print("=" * 60)
        print("Graded PropLines per stat (latest line per book/stat/game/player):")
        print("=" * 60)
        for stat in STATS:
            print(
                f"  {stat:14s}  graded={per_stat_graded[stat]:5d} / "
                f"total={per_stat_total[stat]:5d}"
            )
        print()
        print("Per-stat graded counts by date (top 14 days):")
        for stat in STATS:
            print(f"\n  {stat}:")
            for date_str, n in sorted(per_stat_by_date[stat].items())[-14:]:
                print(f"    {date_str}  {n}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

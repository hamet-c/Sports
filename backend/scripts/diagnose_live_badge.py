"""
One-off diagnostic: re-grade every PropLine in the last N days against actuals
under three calibration configurations × two recommendation-threshold schemes:

    Calibration:
        1. cal_on   — *_xgbq_calibration.joblib (Phase 5 fit)
        2. cal_off  — no calibrator, raw P(over)
        3. cal_old  — *_xgbq_calibration.joblib.old (pre-Phase-5 backup)
    Thresholds:
        sym_05   — OVER & UNDER both at +5% EV (current production setting)
        asym_510 — UNDER at +5%, OVER at +10% (Plan B mitigation for mean-head
                   bias: demand more evidence on the side the model overshoots)

Six configs per prop. We cache distributions per (player, game) per cal config
so prediction cost is 3× per prop (not 6×). Goal: pick the best config before
flipping any production switches.

Run from backend/:
    .venv\\Scripts\\python.exe scripts\\diagnose_live_badge.py --days 7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.models import Game, PlayerGameStats, PropLine
from app.db.session import SessionLocal, init_db
from app.models.calibration import IsotonicCalibrator
from app.models.registry import ModelRegistry, RegisteredModel
from app.services.edge import expected_value
from app.services.prediction_service import PredictionService


CAL_CONFIGS = ("cal_on", "cal_off", "cal_old")
# (under_threshold, over_threshold) — symmetric +5% is the production default,
# asym_510 is the Plan B mitigation for mean-head bias on OVER.
THRESHOLD_CONFIGS: dict[str, tuple[float, float]] = {
    "sym_05": (0.05, 0.05),
    "asym_510": (0.05, 0.10),
}
# Cross product of cal × threshold. Stable order for stable JSON output.
CONFIGS = tuple(f"{c}__{t}" for c in CAL_CONFIGS for t in THRESHOLD_CONFIGS)
STATS = ("points", "rebounds", "assists", "threes_made")


def _empty_bucket() -> dict:
    return {"n": 0, "wins": 0, "losses": 0, "pushes": 0,
            "over_n": 0, "over_wins": 0,
            "under_n": 0, "under_wins": 0}


def _finalize(b: dict) -> dict:
    settled = b["n"] - b["pushes"]
    out = dict(b)
    out["win_rate"] = (b["wins"] / settled) if settled > 0 else None
    out["over_win_rate"] = (b["over_wins"] / b["over_n"]) if b["over_n"] else None
    out["under_win_rate"] = (b["under_wins"] / b["under_n"]) if b["under_n"] else None
    return out


def _load_registries() -> dict[str, ModelRegistry]:
    """Three registries pointing at the same predictors but different calibrators."""
    base = ModelRegistry(use_calibrators=True)
    base.load()
    if len(base) == 0:
        raise RuntimeError("No models loaded — train first.")

    # cal_off: same predictors, calibrator=None
    cal_off = ModelRegistry(use_calibrators=False)
    cal_off._models = {                                                   # type: ignore[attr-defined]
        stat: RegisteredModel(predictor=rm.predictor, calibrator=None)
        for stat, rm in base._models.items()                              # type: ignore[attr-defined]
    }

    # cal_old: load *_xgbq_calibration.joblib.old where present
    cal_old = ModelRegistry(use_calibrators=True)
    cal_old._models = {}                                                  # type: ignore[attr-defined]
    for stat, rm in base._models.items():                                 # type: ignore[attr-defined]
        old_path = settings.models_dir / f"{stat}_xgbq_calibration.joblib.old"
        old_cal: IsotonicCalibrator | None = None
        if old_path.exists():
            try:
                old_cal = IsotonicCalibrator.load(str(old_path))
            except Exception as e:
                logger.warning(f"failed to load {old_path}: {e}")
        cal_old._models[stat] = RegisteredModel(predictor=rm.predictor, calibrator=old_cal)  # type: ignore[attr-defined]

    return {"cal_on": base, "cal_off": cal_off, "cal_old": cal_old}


def _p_over_for(reg, prop: PropLine, game: Game, svc: PredictionService) -> float | None:
    """Return calibrated P(over) for this prop under the registry's calibrator
    (which may be None for cal_off). Caching is the caller's responsibility."""
    rm = reg.get(prop.stat_type)
    if rm is None:
        return None
    _, dists = svc.predict_player_game(
        prop.player_id, prop.game_id, game.game_date, stat_types=[prop.stat_type],
    )
    dist = dists.get(prop.stat_type)
    if dist is None:
        return None
    raw_over = dist.probability_over(prop.line)
    if rm.calibrator is not None and rm.calibrator.is_fitted():
        return float(rm.calibrator.transform(np.array([raw_over]))[0])
    return float(raw_over)


def _grade_at_threshold(
    p_over: float,
    prop: PropLine,
    actual: float,
    under_thresh: float,
    over_thresh: float,
) -> tuple[str, bool, bool] | None:
    """Apply EV thresholds; return (rec, won, push) or None for PASS."""
    p_under = 1.0 - p_over
    ev_over = expected_value(p_over, prop.over_odds)
    ev_under = expected_value(p_under, prop.under_odds)
    if ev_over >= over_thresh and ev_over > ev_under:
        rec = "OVER"
    elif ev_under >= under_thresh and ev_under > ev_over:
        rec = "UNDER"
    else:
        return None
    push = actual == prop.line
    won = (rec == "OVER" and actual > prop.line) or (rec == "UNDER" and actual < prop.line)
    return rec, won, push


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    # Explicit window overrides --days; needed to grade historical slates
    # (e.g. the May playoff sample from the offseason) reproducibly.
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    args = parser.parse_args()
    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be given together")

    init_db()
    registries = _load_registries()

    if args.start is not None:
        start, end = args.start, args.end
    else:
        today = date.today()
        start = today - timedelta(days=args.days)
        end = today - timedelta(days=1)

    db = SessionLocal()
    try:
        games = (
            db.query(Game)
            .filter(Game.game_date >= start)
            .filter(Game.game_date <= end)
            .all()
        )
        if not games:
            print("No games in window.")
            return

        # Same stub->real reconciliation as slate.recommendation_record.
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

        game_ids = [g.id for g in games]
        game_by_id = {g.id: g for g in games}
        prop_rows = db.query(PropLine).filter(PropLine.game_id.in_(game_ids)).all()
        # Latest line per (player, game, stat, book).
        latest: dict[tuple, PropLine] = {}
        for p in prop_rows:
            key = (p.player_id, p.game_id, p.stat_type, p.book)
            if key not in latest or p.captured_at > latest[key].captured_at:
                latest[key] = p

        real_game_ids = list({stub_to_real.get(gid, gid) for gid in game_ids})
        pgs_rows = (
            db.query(PlayerGameStats)
            .filter(PlayerGameStats.game_id.in_(real_game_ids))
            .all()
        )
        actuals: dict[tuple[int, int], PlayerGameStats] = {
            (r.player_id, r.game_id): r for r in pgs_rows
        }

        # Bucketed counters: stats[stat][config] = bucket, totals[config] = bucket
        # `config` is "cal_X__thresh_Y" so we get the full cross-product.
        per_stat: dict[str, dict[str, dict]] = {
            s: {c: _empty_bucket() for c in CONFIGS} for s in STATS
        }
        totals: dict[str, dict] = {c: _empty_bucket() for c in CONFIGS}

        # Per-cal-config PredictionService. edge_threshold here is irrelevant —
        # we apply thresholds directly via _grade_at_threshold.
        svcs = {
            c: PredictionService(db, reg, edge_threshold=settings.edge_threshold)
            for c, reg in registries.items()
        }

        n_props_graded = 0
        n_props_skipped = 0

        for prop in latest.values():
            if prop.stat_type not in STATS:
                continue
            game = game_by_id.get(prop.game_id)
            if game is None:
                continue
            real_game_id = stub_to_real.get(prop.game_id, prop.game_id)
            actual_row = actuals.get((prop.player_id, real_game_id))
            if actual_row is None:
                n_props_skipped += 1
                continue
            actual_raw = getattr(actual_row, prop.stat_type, None)
            if actual_raw is None:
                n_props_skipped += 1
                continue
            actual = float(actual_raw)

            # Compute P(over) once per cal config; reuse across threshold variants.
            for cal_name in CAL_CONFIGS:
                p_over = _p_over_for(registries[cal_name], prop, game, svcs[cal_name])
                if p_over is None:
                    continue
                for thresh_name, (under_t, over_t) in THRESHOLD_CONFIGS.items():
                    cfg = f"{cal_name}__{thresh_name}"
                    graded = _grade_at_threshold(p_over, prop, actual, under_t, over_t)
                    if graded is None:
                        continue
                    rec, won, push = graded
                    for bucket in (per_stat[prop.stat_type][cfg], totals[cfg]):
                        bucket["n"] += 1
                        if push:
                            bucket["pushes"] += 1
                        elif won:
                            bucket["wins"] += 1
                        else:
                            bucket["losses"] += 1
                        if rec == "OVER":
                            bucket["over_n"] += 1
                            if won:
                                bucket["over_wins"] += 1
                        else:
                            bucket["under_n"] += 1
                            if won:
                                bucket["under_wins"] += 1
            n_props_graded += 1

        out = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_props_graded": n_props_graded,
            "n_props_skipped_no_actual": n_props_skipped,
            "totals": {c: _finalize(totals[c]) for c in CONFIGS},
            "by_stat": {
                s: {c: _finalize(per_stat[s][c]) for c in CONFIGS}
                for s in STATS
            },
        }
        print(json.dumps(out, indent=2, default=float))
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    main()

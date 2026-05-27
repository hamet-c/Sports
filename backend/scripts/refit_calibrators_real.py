"""
Refit per-stat IsotonicCalibrators against **real PropLine outcomes**, not
synthetic L10-rounded lines.

This is the script Phase 5 should have produced. The original
refit_calibrators.py fit on synthetic lines (round(L10 * 2) / 2), which biased
below actuals on counting stats — so the calibrator learned "boost P(over)"
and that boost manufactured false OVER edges against real books. See Phase
5.5 in NEXT_SESSION.md for the rollback.

For each PropLine in the window:
    raw_p_over = model.predict_distribution(features).probability_over(line)
    actual_over = 1 if actual > line else 0

The (raw_p_over, actual_over) pairs go straight into IsotonicCalibrator.fit
— same isotonic class used in Phase 5, just trained on a different target.

By default the script is dry-run: it fits, evaluates, prints a summary, but
DOES NOT save the joblibs. Pass --save to overwrite
data/models/{stat}_xgbq_calibration.joblib. Backups of the Phase 5 fit are
already in *.joblib.old.

Usage (from backend/):
    .venv\\Scripts\\python.exe scripts\\refit_calibrators_real.py \\
        --start 2026-05-09 --end 2026-05-15
    # add --save when ready to ship
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger

from app.core.config import settings
from app.core.logging import configure_logging
from app.db.models import Game, PlayerGameStats, PropLine
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.calibration import IsotonicCalibrator
from app.models.registry import ModelRegistry


STATS = ("points", "rebounds", "assists", "threes_made")
# Minimum (fit_rows, eval_rows) per stat before we trust a fit enough to
# recommend --save. Below these thresholds the script prints a warning and
# refuses to write artifacts even with --save.
MIN_FIT_ROWS = 300
MIN_EVAL_ROWS = 100


def _to_dataframe_row(features: dict):
    import pandas as pd
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


def _bucket_label(p: float) -> str:
    lo = int(p * 10) * 10
    if lo >= 100:
        lo = 90
    return f"{lo:02d}-{lo + 10:02d}"


def _eval_bucket_table(probs: np.ndarray, outs: np.ndarray) -> dict[str, tuple[int, float]]:
    """Per-decile bucket: (n, hit_rate)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for p, o in zip(probs, outs):
        buckets[_bucket_label(float(p))].append(int(o))
    return {b: (len(v), float(sum(v) / len(v)) if v else 0.0) for b, v in buckets.items()}


def collect_pairs(
    db, start: date, end: date,
) -> dict[str, list[tuple[float, int, date]]]:
    """
    For every graded PropLine in [start, end], compute raw P(over) at the book
    line. Returns {stat: [(raw_p_over, actual_over, game_date), ...]} sorted
    by game_date so the caller can chronologically split.
    """
    # Load registry without calibrators — we need raw P(over).
    registry = ModelRegistry(use_calibrators=False)
    registry.load()
    if len(registry) == 0:
        raise RuntimeError("No models loaded — train first.")

    builder = FeatureBuilder(db)

    # Same stub->real game reconciliation slate.py and the diagnostic use.
    games = db.query(Game).filter(Game.game_date >= start).filter(Game.game_date <= end).all()
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
    # Latest line per (player, game, stat, book) — matches the graders.
    latest: dict[tuple, PropLine] = {}
    for p in prop_rows:
        key = (p.player_id, p.game_id, p.stat_type, p.book)
        if key not in latest or p.captured_at > latest[key].captured_at:
            latest[key] = p

    real_game_ids = list({stub_to_real.get(gid, gid) for gid in game_ids})
    pgs_rows = (
        db.query(PlayerGameStats).filter(PlayerGameStats.game_id.in_(real_game_ids)).all()
    )
    actuals: dict[tuple[int, int], PlayerGameStats] = {
        (r.player_id, r.game_id): r for r in pgs_rows
    }

    # Cache distributions per (player, game) since they don't depend on
    # book/line/odds — same stat across multiple books reuses the prediction.
    dist_cache: dict[tuple[int, int, str], "Distribution"] = {}

    out: dict[str, list[tuple[float, int, date]]] = {s: [] for s in STATS}
    n_total = 0
    n_skipped = 0

    for prop in latest.values():
        if prop.stat_type not in STATS:
            continue
        rm = registry.get(prop.stat_type)
        if rm is None:
            continue
        game = game_by_id.get(prop.game_id)
        if game is None:
            continue
        real_gid = stub_to_real.get(prop.game_id, prop.game_id)
        actual_row = actuals.get((prop.player_id, real_gid))
        if actual_row is None:
            n_skipped += 1
            continue
        actual_raw = getattr(actual_row, prop.stat_type, None)
        if actual_raw is None:
            n_skipped += 1
            continue
        actual = float(actual_raw)

        key = (prop.player_id, prop.game_id, prop.stat_type)
        if key in dist_cache:
            dist = dist_cache[key]
        else:
            feat_vec = builder.build(prop.player_id, prop.game_id, as_of=game.game_date)
            X = coerce_feature_frame(_to_dataframe_row(feat_vec.features))
            dist = rm.predictor.predict(X)[0]
            dist_cache[key] = dist

        raw_p = float(dist.probability_over(prop.line))
        # Note: we drop pushes (actual == line). Calibrator can't learn from
        # them and they're rare on .5-spaced book lines anyway.
        if actual == prop.line:
            continue
        actual_over = 1 if actual > prop.line else 0
        out[prop.stat_type].append((raw_p, actual_over, actual_row.game_date))
        n_total += 1

    logger.info(f"Collected {n_total} graded (raw_p, actual_over) pairs ({n_skipped} skipped)")
    for stat in STATS:
        out[stat].sort(key=lambda t: t[2])  # chronological
    return out


def chronological_split(
    pairs: list[tuple[float, int, date]], eval_fraction: float = 0.30,
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """
    Split chronologically: oldest 1-eval_fraction goes to fit, newest
    eval_fraction goes to eval. We split on row count, not date boundary,
    because the data is bursty (~150 rows/day on game days, 0 on off days).
    Returns ([(p, y) for fit], [(p, y) for eval]).
    """
    n = len(pairs)
    if n == 0:
        return [], []
    eval_n = max(1, int(n * eval_fraction))
    fit = [(p, y) for (p, y, _) in pairs[: n - eval_n]]
    ev = [(p, y) for (p, y, _) in pairs[n - eval_n:]]
    return fit, ev


def fit_and_report(
    stat: str,
    fit_rows: list[tuple[float, int]],
    eval_rows: list[tuple[float, int]],
) -> tuple[IsotonicCalibrator | None, dict]:
    """Fit isotonic on fit_rows, evaluate on eval_rows, return calibrator + report."""
    report: dict = {
        "stat": stat,
        "n_fit": len(fit_rows),
        "n_eval": len(eval_rows),
        "fit_over_rate": None,
        "eval_over_rate": None,
        "probe_points": {},
        "eval_buckets_raw": {},
        "eval_buckets_cal": {},
    }
    if not fit_rows:
        logger.warning(f"{stat}: zero fit rows — skipping")
        return None, report

    fit_p = np.array([p for p, _ in fit_rows], dtype=float)
    fit_y = np.array([y for _, y in fit_rows], dtype=int)
    report["fit_over_rate"] = float(fit_y.mean())

    cal = IsotonicCalibrator()
    try:
        cal.fit(fit_p, fit_y)
    except Exception as e:
        logger.error(f"{stat}: isotonic fit failed: {e}")
        return None, report

    # Probe points to see the shape of the curve at a glance.
    probes = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])
    mapped = cal.transform(probes)
    report["probe_points"] = {
        f"{p:.2f}": round(float(m), 3) for p, m in zip(probes, mapped)
    }

    # Eval: per-decile bucket of raw vs. calibrated, and the implied lift
    # on a +5% EV rec rule.
    if eval_rows:
        eval_p = np.array([p for p, _ in eval_rows], dtype=float)
        eval_y = np.array([y for _, y in eval_rows], dtype=int)
        report["eval_over_rate"] = float(eval_y.mean())
        cal_p = cal.transform(eval_p)
        report["eval_buckets_raw"] = _eval_bucket_table(eval_p, eval_y)
        report["eval_buckets_cal"] = _eval_bucket_table(cal_p, eval_y)
    return cal, report


def print_report(report: dict) -> None:
    s = report["stat"]
    logger.info(
        f"  {s}: fit_n={report['n_fit']} (over_rate={report['fit_over_rate']:.3f})  "
        f"eval_n={report['n_eval']} (over_rate={report['eval_over_rate']})"
    )
    if report["probe_points"]:
        logger.info(f"    isotonic curve: {report['probe_points']}")
    if report["eval_buckets_raw"]:
        logger.info(f"    eval buckets RAW : {dict(sorted(report['eval_buckets_raw'].items()))}")
    if report["eval_buckets_cal"]:
        logger.info(f"    eval buckets CAL : {dict(sorted(report['eval_buckets_cal'].items()))}")


def main(start: date, end: date, eval_fraction: float, save: bool) -> None:
    init_db()
    db = SessionLocal()
    try:
        pairs_by_stat = collect_pairs(db, start, end)

        all_reports: dict[str, dict] = {}
        fitted: dict[str, IsotonicCalibrator] = {}
        for stat in STATS:
            pairs = pairs_by_stat[stat]
            fit_rows, eval_rows = chronological_split(pairs, eval_fraction=eval_fraction)
            cal, report = fit_and_report(stat, fit_rows, eval_rows)
            all_reports[stat] = report
            if cal is not None:
                fitted[stat] = cal
            print_report(report)

        # Decide whether to save. Refuse if any stat is below the minimum
        # sample threshold — the user can override by editing MIN_* constants
        # at the top if they have a good reason.
        below_threshold = [
            stat for stat in STATS
            if all_reports[stat]["n_fit"] < MIN_FIT_ROWS
            or all_reports[stat]["n_eval"] < MIN_EVAL_ROWS
        ]
        if below_threshold:
            logger.warning(
                f"Below sample thresholds (need n_fit≥{MIN_FIT_ROWS}, n_eval≥{MIN_EVAL_ROWS}): "
                f"{', '.join(below_threshold)}"
            )

        if not save:
            logger.info("DRY RUN — not writing joblibs. Pass --save when ready.")
            return

        if below_threshold:
            logger.error(
                "Refusing to --save with stats below threshold. "
                "Wait for more graded PropLines and re-run, "
                "or edit MIN_FIT_ROWS/MIN_EVAL_ROWS at the top of this script "
                "if you have a good reason to override."
            )
            return

        for stat, cal in fitted.items():
            out = settings.models_dir / f"{stat}_xgbq_calibration.joblib"
            cal.save(str(out))
            logger.info(f"Saved {out}")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.30,
        help="Fraction of the chronologically-sorted pairs reserved for eval (default 0.30).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Overwrite data/models/{stat}_xgbq_calibration.joblib. Default is "
            "dry-run (fit + eval + print, no writes). Refused below sample "
            "thresholds."
        ),
    )
    args = parser.parse_args()
    main(args.start, args.end, args.eval_fraction, args.save)

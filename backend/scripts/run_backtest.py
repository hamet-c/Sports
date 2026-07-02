"""
Backtest driver.

Two modes:

    # Real mode — uses captured prop_lines + actual outcomes.
    python scripts/run_backtest.py real --start 2024-01-01 --end 2024-04-14

    # Synthetic mode — no prop_lines required. For every completed
    # (player, game), predict and grade against actual using the player's
    # rolling median as a stand-in line. Gives model-quality metrics
    # (MAE / RMSE / log-loss / calibration) without sportsbook data.
    python scripts/run_backtest.py synthetic --start 2024-01-01 --end 2024-04-14

Why two modes: the real backtester needs historical line data we won't
accumulate until the daily ingest has been running for a while. Synthetic
mode gives an immediate model-quality signal off the bootstrap data.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from math import log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from loguru import logger
from tqdm import tqdm

from app.core.logging import configure_logging
from app.db.models import Game, PlayerGameStats
from app.db.session import SessionLocal, init_db
from app.features.builder import FEATURE_COLUMNS, FeatureBuilder, coerce_feature_frame
from app.models.registry import registry
from app.services.backtester import Backtester
from app.services.prediction_service import PredictionService


STATS_TO_EVAL = ["points", "rebounds", "assists", "threes_made"]
# NBA game_id prefix taxonomy. 0022 = regular season, 0042 = playoffs, 0012 =
# preseason, 0052 = play-in tournament. Anything else (or no nba_id at all,
# e.g. stub games from prop_ingest) is classified as "unknown".
_CONTEXT_ALL = "all"
_CONTEXT_RS = "rs"
_CONTEXT_PO = "po"
_CONTEXT_UNKNOWN = "unknown"


def _classify_game(nba_id: str | None) -> str:
    if not nba_id:
        return _CONTEXT_UNKNOWN
    if nba_id.startswith("0022"):
        return _CONTEXT_RS
    if nba_id.startswith("0042"):
        return _CONTEXT_PO
    return _CONTEXT_UNKNOWN


# ============================ Real-line backtest ============================ #

def run_real(start: date, end: date, edge_threshold: float, stake: float) -> dict:
    db = SessionLocal()
    try:
        if len(registry) == 0:
            logger.error("No models loaded — train first.")
            return {}
        svc = PredictionService(db, registry, edge_threshold=edge_threshold)
        bt = Backtester(db, svc, edge_threshold=edge_threshold, stake_per_bet=stake)
        result = bt.run(start, end)
        out = result.to_dict()
        print(json.dumps(out, indent=2, default=float))
        return out
    finally:
        db.close()


# ============================ Synthetic backtest ============================ #

@dataclass
class StatBucket:
    n: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    log_loss_sum: float = 0.0
    log_loss_n: int = 0
    # Calibration: for each 10% bucket of predicted P(over), tally how often actual was over.
    cal_n: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cal_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Baseline (L10 mean) for comparison.
    baseline_abs_err_sum: float = 0.0
    baseline_n: int = 0
    # Recommendation tally: how often the OVER/UNDER we'd actually bet (EV @ -110
    # exceeds the edge threshold) hit. PASS rows excluded.
    rec_n: int = 0
    rec_wins: int = 0
    rec_pushes: int = 0
    rec_over_n: int = 0
    rec_over_wins: int = 0
    rec_under_n: int = 0
    rec_under_wins: int = 0

    def to_dict(self) -> dict:
        mae = self.abs_err_sum / self.n if self.n else 0.0
        rmse = (self.sq_err_sum / self.n) ** 0.5 if self.n else 0.0
        ll = self.log_loss_sum / self.log_loss_n if self.log_loss_n else 0.0
        baseline_mae = self.baseline_abs_err_sum / self.baseline_n if self.baseline_n else None
        cal = {}
        for bucket in sorted(self.cal_n):
            n = self.cal_n[bucket]
            hits = self.cal_hits[bucket]
            cal[bucket] = {"n": n, "hit_rate": hits / n if n else 0.0}
        rec_settled = self.rec_n - self.rec_pushes
        return {
            "n": self.n,
            "mae": mae,
            "rmse": rmse,
            "log_loss": ll,
            "baseline_l10_mae": baseline_mae,
            "vs_baseline_lift": (baseline_mae - mae) if baseline_mae is not None else None,
            "calibration": cal,
            "recommendations": {
                "n": self.rec_n,
                "pushes": self.rec_pushes,
                "wins": self.rec_wins,
                "win_rate": (self.rec_wins / rec_settled) if rec_settled else None,
                "over": {
                    "n": self.rec_over_n,
                    "wins": self.rec_over_wins,
                    "win_rate": (self.rec_over_wins / self.rec_over_n) if self.rec_over_n else None,
                },
                "under": {
                    "n": self.rec_under_n,
                    "wins": self.rec_under_wins,
                    "win_rate": (self.rec_under_wins / self.rec_under_n) if self.rec_under_n else None,
                },
            },
        }


_BASELINE_COL = {
    "points": "pts_avg_10",
    "rebounds": "reb_avg_10",
    "assists": "ast_avg_10",
    "threes_made": "threes_avg_10",
}

# Standard sportsbook juice. Each side of an over/under is typically -110, so
# decimal payout is 1.909... and EV >= 0 requires P >= ~52.4%. We use -110 for
# all synthetic recommendation grading so the win-rate is comparable to what
# you'd actually need to clear vig on a real book.
_SYNTHETIC_ODDS = -110
_SYNTHETIC_DECIMAL = 1.0 + 100.0 / 110.0  # 1.9090909...
_REC_EDGE_THRESHOLD = 0.05  # match settings.edge_threshold default


def _synthetic_ev(p: float) -> float:
    """EV per $1 stake at -110 with calibrated probability p."""
    return p * (_SYNTHETIC_DECIMAL - 1.0) - (1.0 - p)


def _bucket(p: float) -> str:
    lo = int(p * 10) * 10
    if lo >= 100:
        lo = 90
    return f"{lo:02d}-{lo + 10:02d}"


def run_synthetic(start: date, end: date, min_minutes: float) -> dict:
    """
    Walk completed games in the window. For each game, build features
    AS-OF the game date (no leakage), predict the distribution, and score
    the prediction against the realized stat using the predicted median
    as the synthetic over/under line.
    """
    db = SessionLocal()
    try:
        if len(registry) == 0:
            logger.error("No models loaded — train first.")
            return {}
        builder = FeatureBuilder(db)
        rows = (
            db.query(PlayerGameStats)
            .filter(PlayerGameStats.game_date >= start)
            .filter(PlayerGameStats.game_date <= end)
            .filter(PlayerGameStats.minutes != None)  # noqa: E711
            .order_by(PlayerGameStats.game_date)
            .all()
        )
        rows = [r for r in rows if r.minutes is not None and r.minutes >= min_minutes]
        logger.info(f"Synthetic backtest over {len(rows)} player-games")

        # Classify each game once (RS vs PO) so the inner loop only does a dict
        # lookup. The nba_id prefix distinguishes 0022 (RS) from 0042 (PO).
        needed_game_ids = {r.game_id for r in rows}
        game_context: dict[int, str] = {}
        if needed_game_ids:
            for g in db.query(Game.id, Game.nba_id).filter(Game.id.in_(needed_game_ids)).all():
                game_context[g.id] = _classify_game(g.nba_id)

        # Two buckets per stat: an "all" rollup (preserves the legacy report
        # shape) and one per context.
        buckets: dict[tuple[str, str], StatBucket] = {
            (ctx, stat): StatBucket()
            for ctx in (_CONTEXT_ALL, _CONTEXT_RS, _CONTEXT_PO, _CONTEXT_UNKNOWN)
            for stat in STATS_TO_EVAL
        }
        context_counts: dict[str, int] = {
            _CONTEXT_RS: 0, _CONTEXT_PO: 0, _CONTEXT_UNKNOWN: 0,
        }

        for r in tqdm(rows, desc="scoring"):
            row_ctx = game_context.get(r.game_id, _CONTEXT_UNKNOWN)
            context_counts[row_ctx] = context_counts.get(row_ctx, 0) + 1
            feat_vec = builder.build(r.player_id, r.game_id, as_of=r.game_date)
            X = coerce_feature_frame(
                _to_dataframe_row(feat_vec.features)
            )
            for stat in STATS_TO_EVAL:
                rm = registry.get(stat)
                if rm is None:
                    continue
                actual = getattr(r, stat, None)
                if actual is None:
                    continue
                actual = float(actual)
                dist = rm.predictor.predict(X)[0]

                # Each row contributes to the "all" rollup AND its specific
                # context (RS or PO). Targets is the list of buckets we apply
                # every per-row increment to.
                targets = [buckets[(_CONTEXT_ALL, stat)], buckets[(row_ctx, stat)]]

                err = dist.mean - actual
                for bucket in targets:
                    bucket.n += 1
                    bucket.abs_err_sum += abs(err)
                    bucket.sq_err_sum += err * err

                # Synthetic line: round the player's L10 mean to the nearest
                # 0.5 — that's how books anchor lines, and it produces a
                # distribution of P(over) values across [0,1] so calibration
                # buckets are actually informative. Falls back to predicted
                # median if no L10 baseline exists yet.
                baseline = feat_vec.features.get(_BASELINE_COL[stat])
                if baseline is not None:
                    line = round(float(baseline) * 2) / 2
                    if line == int(line):
                        line += 0.5
                else:
                    line = dist.quantiles.get(0.5, dist.mean)
                p_over = dist.probability_over(line)
                if rm.calibrator is not None and rm.calibrator.is_fitted():
                    p_over = float(rm.calibrator.transform(np.array([p_over]))[0])
                p = min(max(p_over, 1e-6), 1 - 1e-6)
                actual_over = 1 if actual > line else 0
                ll_term = -(actual_over * log(p) + (1 - actual_over) * log(1 - p))
                cb = _bucket(p)
                for bucket in targets:
                    bucket.log_loss_sum += ll_term
                    bucket.log_loss_n += 1
                    bucket.cal_n[cb] += 1
                    bucket.cal_hits[cb] += actual_over

                # Recommendation tally: compute EV at synthetic -110 odds for
                # both sides; if either clears the edge threshold, that side is
                # the "bet we'd have made". Grade against actual. A push
                # (actual == line) settles to no win/no loss but counts toward
                # the bet count for transparency.
                ev_o = _synthetic_ev(p)
                ev_u = _synthetic_ev(1.0 - p)
                rec_side: str | None = None
                if ev_o >= _REC_EDGE_THRESHOLD and ev_o > ev_u:
                    rec_side = "over"
                elif ev_u >= _REC_EDGE_THRESHOLD and ev_u > ev_o:
                    rec_side = "under"
                if rec_side is not None:
                    is_push = actual == line
                    won_over = rec_side == "over" and actual > line
                    won_under = rec_side == "under" and actual < line
                    for bucket in targets:
                        bucket.rec_n += 1
                        if is_push:
                            bucket.rec_pushes += 1
                        if rec_side == "over":
                            bucket.rec_over_n += 1
                            if won_over:
                                bucket.rec_wins += 1
                                bucket.rec_over_wins += 1
                        else:
                            bucket.rec_under_n += 1
                            if won_under:
                                bucket.rec_wins += 1
                                bucket.rec_under_wins += 1

                # Baseline: per-row L10 mean (already in the feature row).
                baseline_col = _BASELINE_COL[stat]
                baseline = feat_vec.features.get(baseline_col)
                if baseline is not None:
                    bl_err = abs(float(baseline) - actual)
                    for bucket in targets:
                        bucket.baseline_abs_err_sum += bl_err
                        bucket.baseline_n += 1

        # Build the report. The top-level stat keys remain the "all" rollup
        # so existing consumers (performance API, diff_backtests, frontend
        # cards) keep working unchanged. _by_context is additive and only
        # populated for non-empty contexts. The leading underscore signals
        # "metadata / supplementary" so legacy iterators that walk stat keys
        # skip it naturally.
        out: dict = {stat: buckets[(_CONTEXT_ALL, stat)].to_dict() for stat in STATS_TO_EVAL}
        by_context: dict[str, dict] = {}
        for ctx in (_CONTEXT_RS, _CONTEXT_PO):
            ctx_result = {stat: buckets[(ctx, stat)].to_dict() for stat in STATS_TO_EVAL}
            if any(ctx_result[s]["n"] > 0 for s in STATS_TO_EVAL):
                by_context[ctx] = ctx_result
        if by_context:
            out["_by_context"] = by_context
        out["_context_counts"] = {k: v for k, v in context_counts.items() if v > 0}
        print(json.dumps(out, indent=2, default=float))
        return out
    finally:
        db.close()


def _to_dataframe_row(features: dict) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


# ================================== entry =================================== #

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    real = sub.add_parser("real", help="Backtest using captured prop_lines.")
    real.add_argument("--start", type=date.fromisoformat, required=True)
    real.add_argument("--end", type=date.fromisoformat, required=True)
    real.add_argument("--edge-threshold", type=float, default=0.05)
    real.add_argument("--stake", type=float, default=1.0)
    real.add_argument("--save", action="store_true", help="Persist report JSON for the API to serve.")

    syn = sub.add_parser("synthetic", help="Model-quality eval at synthetic median line.")
    syn.add_argument("--start", type=date.fromisoformat, required=True)
    syn.add_argument("--end", type=date.fromisoformat, required=True)
    syn.add_argument("--min-minutes", type=float, default=8.0)
    syn.add_argument("--save", action="store_true", help="Persist report JSON for the API to serve.")

    args = parser.parse_args()
    init_db()
    registry.load()

    if args.mode == "real":
        result = run_real(args.start, args.end, args.edge_threshold, args.stake)
    else:
        result = run_synthetic(args.start, args.end, args.min_minutes)

    if args.save and result:
        from app.core.config import settings as _settings
        from app.core.timeutil import utcnow_iso_z
        reports_dir = _settings.data_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"{args.mode}_backtest.json"
        envelope = {
            "mode": args.mode,
            "generated_at": utcnow_iso_z(),
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "result": result,
        }
        out_path.write_text(json.dumps(envelope, indent=2, default=float))
        logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    configure_logging()
    main()

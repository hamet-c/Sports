"""
Diagnostic: why do UNDER recommendations outperform OVER recommendations?

Reuses the same scoring loop as run_backtest.py synthetic mode but captures
per-row diagnostic data instead of aggregates:
  - prediction (mean, median, p10/p25/p75/p90)
  - synthetic line + L10 baseline
  - p_over (and calibrated p_over if available)
  - actual outcome
  - rec side (over / under / pass) + EV at -110
  - context (rs vs po)

Then it slices the data several ways to test specific hypotheses:

  H1: prediction is biased high.    If mean(pred - actual) > 0, the model
      systematically over-predicts. That inflates p_over and pulls OVER recs
      onto false-positive lines.

  H2: synthetic-line bias.          Compare actual to line directly: is
      P(actual > line) materially != 50% even before the model says anything?
      If yes, our "line" construction is biased and the rec asymmetry might
      mostly come from the line itself.

  H3: calibration asymmetry.        Re-bucket the calibration table at the
      HIGH end (p_over > 60%) vs the LOW end (p_over < 40%). If the high end
      under-hits and the low end over-hits, predicted P(over) is too extreme
      in both directions — a classic miscalibration that would punish OVER
      recs (which fire from p_over high) more than UNDER recs (which fire
      from p_over low and convert via 1-p).

  H4: line distance asymmetry.      For OVER recs, p_over is high which means
      the line sits low in the predicted distribution. For UNDER recs,
      p_over is low which means the line sits high. Are these "distances"
      symmetric or does one side push further? Asymmetric line distance
      could explain the win-rate gap.

  H5: threshold sensitivity.        If we raise the OVER edge threshold,
      does OVER win rate climb? (= the marginal +5% OVER recs are the bad
      ones; the strong-signal OVERs are fine.)

Run:
    cd backend
    .venv\\Scripts\\python.exe scripts\\investigate_asymmetry.py \\
        --start 2026-02-15 --end 2026-04-13
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
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


STATS = ["points", "rebounds", "assists", "threes_made"]
BASELINE_COL = {
    "points": "pts_avg_10",
    "rebounds": "reb_avg_10",
    "assists": "ast_avg_10",
    "threes_made": "threes_avg_10",
}
SYNTHETIC_DECIMAL = 1.0 + 100.0 / 110.0
EDGE_THRESHOLD = 0.05


def _to_dataframe_row(features: dict):
    import pandas as pd
    return pd.DataFrame([{c: features.get(c) for c in FEATURE_COLUMNS}])


def _ev(p: float) -> float:
    return p * (SYNTHETIC_DECIMAL - 1.0) - (1.0 - p)


def _classify_game(nba_id):
    if not nba_id:
        return "unknown"
    if nba_id.startswith("0022"):
        return "rs"
    if nba_id.startswith("0042"):
        return "po"
    return "unknown"


def collect(start: date, end: date, min_minutes: float) -> dict[str, list[dict]]:
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
        logger.info(f"Collecting diagnostic rows over {len(rows)} player-games")

        game_ids = {r.game_id for r in rows}
        game_ctx: dict[int, str] = {}
        for g in db.query(Game.id, Game.nba_id).filter(Game.id.in_(game_ids)).all():
            game_ctx[g.id] = _classify_game(g.nba_id)

        per_stat: dict[str, list[dict]] = {s: [] for s in STATS}

        for r in tqdm(rows, desc="scoring"):
            ctx = game_ctx.get(r.game_id, "unknown")
            feat_vec = builder.build(r.player_id, r.game_id, as_of=r.game_date)
            X = coerce_feature_frame(_to_dataframe_row(feat_vec.features))

            for stat in STATS:
                rm = registry.get(stat)
                if rm is None:
                    continue
                actual = getattr(r, stat, None)
                if actual is None:
                    continue
                actual = float(actual)
                dist = rm.predictor.predict(X)[0]

                baseline = feat_vec.features.get(BASELINE_COL[stat])
                if baseline is not None:
                    line = round(float(baseline) * 2) / 2
                    if line == int(line):
                        line += 0.5
                else:
                    line = dist.quantiles.get(0.5, dist.mean)

                p_over_raw = dist.probability_over(line)
                if rm.calibrator is not None and rm.calibrator.is_fitted():
                    p_over = float(rm.calibrator.transform(np.array([p_over_raw]))[0])
                else:
                    p_over = p_over_raw
                p_over = min(max(p_over, 1e-6), 1 - 1e-6)

                ev_o = _ev(p_over)
                ev_u = _ev(1.0 - p_over)
                rec = "pass"
                if ev_o >= EDGE_THRESHOLD and ev_o > ev_u:
                    rec = "over"
                elif ev_u >= EDGE_THRESHOLD and ev_u > ev_o:
                    rec = "under"

                per_stat[stat].append({
                    "actual": actual,
                    "pred_mean": float(dist.mean),
                    "q10": float(dist.quantiles.get(0.1, np.nan)),
                    "q25": float(dist.quantiles.get(0.25, np.nan)),
                    "q50": float(dist.quantiles.get(0.5, np.nan)),
                    "q75": float(dist.quantiles.get(0.75, np.nan)),
                    "q90": float(dist.quantiles.get(0.9, np.nan)),
                    "line": float(line),
                    "baseline_l10": float(baseline) if baseline is not None else None,
                    "p_over_raw": float(p_over_raw),
                    "p_over": float(p_over),
                    "ev_over": float(ev_o),
                    "ev_under": float(ev_u),
                    "rec": rec,
                    "ctx": ctx,
                })

        return per_stat
    finally:
        db.close()


# ----------------------------- hypothesis tests ----------------------------- #

def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def report(per_stat: dict[str, list[dict]]) -> dict:
    out = {}
    for stat, rows in per_stat.items():
        if not rows:
            continue
        n = len(rows)
        actuals = np.array([r["actual"] for r in rows], dtype=float)
        preds = np.array([r["pred_mean"] for r in rows], dtype=float)
        lines = np.array([r["line"] for r in rows], dtype=float)
        baselines = np.array(
            [r["baseline_l10"] if r["baseline_l10"] is not None else np.nan for r in rows],
            dtype=float,
        )
        p_overs = np.array([r["p_over"] for r in rows], dtype=float)
        recs = np.array([r["rec"] for r in rows])

        # H1: prediction bias.
        h1 = {
            "mean_pred_minus_actual": float((preds - actuals).mean()),
            "median_pred_minus_actual": float(np.median(preds - actuals)),
            "frac_pred_gt_actual": float((preds > actuals).mean()),
        }

        # H2: line bias (independent of model).
        h2 = {
            "mean_line_minus_actual": float((lines - actuals).mean()),
            "frac_actual_over_line": float((actuals > lines).mean()),
            "frac_actual_under_line": float((actuals < lines).mean()),
            "mean_baseline_minus_actual": (
                float(np.nanmean(baselines - actuals))
                if not np.all(np.isnan(baselines)) else None
            ),
        }

        # H3: calibration asymmetry. Re-bucket at 10% buckets, but also
        # compute aggregate hit rate for "low" (p < 0.4) and "high" (p > 0.6).
        actual_over = (actuals > lines).astype(float)
        cal_buckets = defaultdict(list)
        for p, a in zip(p_overs, actual_over):
            lo = int(p * 10) * 10
            if lo >= 100:
                lo = 90
            cal_buckets[f"{lo:02d}-{lo+10:02d}"].append(a)
        cal = {
            k: {"n": len(v), "hit_rate": float(np.mean(v))}
            for k, v in sorted(cal_buckets.items())
        }
        low_mask = p_overs < 0.4
        high_mask = p_overs > 0.6
        h3 = {
            "calibration": cal,
            "low_p_over_n": int(low_mask.sum()),
            "low_p_over_mean_pred_prob": (
                float(p_overs[low_mask].mean()) if low_mask.any() else None
            ),
            "low_p_over_actual_hit_rate": (
                float(actual_over[low_mask].mean()) if low_mask.any() else None
            ),
            "high_p_over_n": int(high_mask.sum()),
            "high_p_over_mean_pred_prob": (
                float(p_overs[high_mask].mean()) if high_mask.any() else None
            ),
            "high_p_over_actual_hit_rate": (
                float(actual_over[high_mask].mean()) if high_mask.any() else None
            ),
        }

        # H4: line distance asymmetry for actual recs.
        over_mask = recs == "over"
        under_mask = recs == "under"
        # standardize line position within predicted distribution
        # measure as (line - q50) / (q75 - q25) (interquantile distance).
        q25 = np.array([r["q25"] for r in rows], dtype=float)
        q50 = np.array([r["q50"] for r in rows], dtype=float)
        q75 = np.array([r["q75"] for r in rows], dtype=float)
        iqr = np.maximum(q75 - q25, 1e-6)
        std_line_pos = (lines - q50) / iqr
        h4 = {
            "over_rec_n": int(over_mask.sum()),
            "over_rec_mean_p_over": (
                float(p_overs[over_mask].mean()) if over_mask.any() else None
            ),
            "over_rec_mean_std_line_pos": (
                float(std_line_pos[over_mask].mean()) if over_mask.any() else None
            ),  # negative = line is below median (favors OVER)
            "over_rec_mean_actual_minus_line": (
                float((actuals[over_mask] - lines[over_mask]).mean())
                if over_mask.any() else None
            ),
            "under_rec_n": int(under_mask.sum()),
            "under_rec_mean_p_over": (
                float(p_overs[under_mask].mean()) if under_mask.any() else None
            ),
            "under_rec_mean_std_line_pos": (
                float(std_line_pos[under_mask].mean()) if under_mask.any() else None
            ),  # positive = line is above median (favors UNDER)
            "under_rec_mean_actual_minus_line": (
                float((actuals[under_mask] - lines[under_mask]).mean())
                if under_mask.any() else None
            ),
        }

        # H5: threshold sensitivity. Walk EV thresholds and report
        # OVER / UNDER win rates at each.
        win = actuals > lines
        loss = actuals < lines
        thresh = {}
        for t in [0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]:
            ev_o = p_overs * (SYNTHETIC_DECIMAL - 1.0) - (1.0 - p_overs)
            ev_u = (1 - p_overs) * (SYNTHETIC_DECIMAL - 1.0) - p_overs
            o_mask = (ev_o >= t) & (ev_o > ev_u)
            u_mask = (ev_u >= t) & (ev_u > ev_o)
            thresh[f"{t:.3f}"] = {
                "over_n": int(o_mask.sum()),
                "over_win_rate": (
                    float(win[o_mask].mean()) if o_mask.any() else None
                ),
                "under_n": int(u_mask.sum()),
                "under_win_rate": (
                    float(loss[u_mask].mean()) if u_mask.any() else None
                ),
            }
        h5 = {"by_ev_threshold": thresh}

        # H6 (bonus): bias broken out by context (rs vs po).
        ctxs = np.array([r["ctx"] for r in rows])
        h6 = {}
        for c in ("rs", "po"):
            m = ctxs == c
            if not m.any():
                continue
            h6[c] = {
                "n": int(m.sum()),
                "mean_pred_minus_actual": float((preds[m] - actuals[m]).mean()),
                "mean_line_minus_actual": float((lines[m] - actuals[m]).mean()),
                "frac_actual_over_line": float((actuals[m] > lines[m]).mean()),
            }

        out[stat] = {
            "n": n,
            "h1_pred_bias": h1,
            "h2_line_bias": h2,
            "h3_calibration": h3,
            "h4_rec_geometry": h4,
            "h5_threshold_sweep": h5,
            "h6_by_context": h6,
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--min-minutes", type=float, default=8.0)
    p.add_argument("--out", type=Path, default=Path("../data/reports/asymmetry_diagnostic.json"))
    args = p.parse_args()

    configure_logging()
    init_db()
    registry.load()

    rows = collect(args.start, args.end, args.min_minutes)
    diagnostic = report(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(diagnostic, indent=2, default=float))
    logger.info(f"Wrote {args.out}")
    print(json.dumps(diagnostic, indent=2, default=float))


if __name__ == "__main__":
    main()

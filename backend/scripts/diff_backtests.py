"""
Compare two synthetic_backtest.json reports side-by-side.

    python scripts/diff_backtests.py \\
        --base data/reports/synthetic_backtest_phase3.json \\
        --new  data/reports/synthetic_backtest.json

Prints a per-stat table with MAE / baseline / lift / log-loss deltas. Designed
for end-of-phase wrap-ups so we can see at a glance which retrains actually
improved things and which were noise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


STATS = ("points", "rebounds", "assists", "threes_made")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        sys.exit(f"Report not found: {path}")
    return json.loads(path.read_text())


def _fmt_delta(new: float | None, base: float | None, digits: int = 3) -> str:
    if new is None or base is None:
        return "       -"
    d = new - base
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{digits}f}"


def _row(stat: str, base: dict, new: dict) -> str:
    bn = base.get("result", {}).get(stat) or {}
    nn = new.get("result", {}).get(stat) or {}
    bm = bn.get("mae"); nm = nn.get("mae")
    bl = bn.get("vs_baseline_lift"); nl = nn.get("vs_baseline_lift")
    bll = bn.get("log_loss"); nll = nn.get("log_loss")
    bn_ = bn.get("n") or 0
    nn_ = nn.get("n") or 0

    def fmt(x: float | None, digits: int = 3) -> str:
        return f"{x:.{digits}f}" if x is not None else "  -  "

    return (
        f"{stat:<13} "
        f"n={bn_:>5} -> {nn_:<5}   "
        f"MAE {fmt(bm)} -> {fmt(nm)} ({_fmt_delta(nm, bm)})   "
        f"lift {fmt(bl)} -> {fmt(nl)} ({_fmt_delta(nl, bl)})   "
        f"logloss {fmt(bll, 4)} -> {fmt(nll, 4)} ({_fmt_delta(nll, bll, 4)})"
    )


def _rec_row(stat: str, base: dict, new: dict) -> str | None:
    """Recommendation hit-rate row. None if the bucket doesn't exist yet
    (e.g. comparing against an older report from before this feature)."""
    bn = (base.get("result", {}).get(stat) or {}).get("recommendations")
    nn = (new.get("result", {}).get(stat) or {}).get("recommendations")
    if bn is None and nn is None:
        return None
    bn = bn or {}
    nn = nn or {}
    bw = bn.get("win_rate"); nw = nn.get("win_rate")
    b_n = bn.get("n", 0); n_n = nn.get("n", 0)
    b_wins = bn.get("wins", 0); n_wins = nn.get("wins", 0)

    def fmt_rate(r: float | None) -> str:
        return f"{r * 100:5.1f}%" if r is not None else "  -  "

    def fmt_delta_rate(new_r: float | None, base_r: float | None) -> str:
        if new_r is None or base_r is None:
            return "      -"
        d = (new_r - base_r) * 100
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}pp"

    return (
        f"{stat:<13} "
        f"recs {b_n:>4}->{n_n:<4}  "
        f"W {b_wins:>4}->{n_wins:<4}  "
        f"win_rate {fmt_rate(bw)} -> {fmt_rate(nw)} ({fmt_delta_rate(nw, bw)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="Older report (baseline).")
    parser.add_argument("--new", type=Path, required=True, help="Newer report (after retrain).")
    args = parser.parse_args()

    base = _load(args.base)
    new = _load(args.new)

    print(f"Base: {args.base.name}  ({base.get('start')} -> {base.get('end')})  "
          f"generated {base.get('generated_at', '?')[:19]}")
    print(f"New:  {args.new.name}   ({new.get('start')} -> {new.get('end')})   "
          f"generated {new.get('generated_at', '?')[:19]}")
    if base.get("start") != new.get("start") or base.get("end") != new.get("end"):
        print("\nWARNING: backtest windows differ - comparison may not be apples-to-apples.\n")
    else:
        print()

    for stat in STATS:
        print(_row(stat, base, new))

    rec_rows = [_rec_row(s, base, new) for s in STATS]
    if any(r is not None for r in rec_rows):
        print()
        print("Recommendations (EV >= 5% at -110 odds):")
        for r in rec_rows:
            if r is not None:
                print("  " + r)

    # Quick verdict line.
    print()
    n_better, n_worse = 0, 0
    for stat in STATS:
        bm = base.get("result", {}).get(stat, {}).get("mae")
        nm = new.get("result", {}).get(stat, {}).get("mae")
        if bm is None or nm is None:
            continue
        if nm < bm:
            n_better += 1
        elif nm > bm:
            n_worse += 1
    print(f"MAE improved on {n_better}/{len(STATS)} stats, regressed on {n_worse}.")


if __name__ == "__main__":
    main()

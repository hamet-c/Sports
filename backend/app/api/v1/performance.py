"""
Backtest report endpoint.

Reads JSON written by `scripts/run_backtest.py --save`. Computing the report
on demand is slow (minutes to walk thousands of player-games), so we serve
cached files instead. Re-run the script to refresh.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.registry import registry

router = APIRouter()


def _load_report(mode: str) -> dict[str, Any] | None:
    path = settings.data_dir / "reports" / f"{mode}_backtest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@router.get("/")
def get_performance() -> dict[str, Any]:
    """Return both real and synthetic reports if present.

    Shape:
        {
            "synthetic": {generated_at, start, end, result: {<stat>: {...}}} | null,
            "real":      {generated_at, start, end, result: {...}} | null
        }
    """
    return {
        "synthetic": _load_report("synthetic"),
        "real": _load_report("real"),
    }


@router.get("/feature_importance")
def get_feature_importance(top_k: int = 0) -> dict[str, Any]:
    """
    Per-stat feature importance from the loaded models.

    Returns:
        {
            "stats": {
                "<stat>": [{"feature": str, "importance": float}, ...]
            }
        }

    `importance` sums to ~1 per stat (normalized). Sorted descending.
    `top_k=0` returns every feature.
    """
    if len(registry) == 0:
        raise HTTPException(status_code=503, detail="No models loaded")
    out: dict[str, list[dict[str, float]]] = {}
    for stat in registry.all_stats():
        rm = registry.get(stat)
        if rm is None:
            continue
        try:
            imp = rm.predictor.feature_importance()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"importance failed for {stat}: {e}")
        ranked = sorted(
            [{"feature": k, "importance": v} for k, v in imp.items()],
            key=lambda r: r["importance"], reverse=True,
        )
        if top_k > 0:
            ranked = ranked[:top_k]
        out[stat] = ranked
    return {"stats": out}


@router.get("/coverage")
def get_coverage() -> dict[str, Any]:
    """
    Per-stat per-feature non-null fraction over the training set used by
    the most recent `train_models.py` run. Surfaces dead features (coverage
    near 0 means the model effectively never sees that signal).
    """
    path = settings.data_dir / "reports" / "feature_coverage.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No coverage report yet. Run `python scripts/train_models.py ...` to generate one.",
        )
    return json.loads(path.read_text())


@router.get("/{mode}")
def get_performance_mode(mode: str) -> dict[str, Any]:
    if mode not in ("synthetic", "real"):
        raise HTTPException(status_code=404, detail="mode must be 'synthetic' or 'real'")
    report = _load_report(mode)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {mode} report yet. Run `python scripts/run_backtest.py {mode} --start ... --end ... --save`.",
        )
    return report

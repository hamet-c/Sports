"""
Usage / role features.

We compute the player's own usage_rate trend (when available) plus a
"teammate usage cascade" signal: the share of trailing-window team
points/assists/usage held by inactive top-3 usage teammates.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class UsageFeatures:
    usage_rate_l10: float | None
    started_pct_l10: float | None
    minutes_share_l10: float | None  # player's mean minutes / 240 team minutes
    teammate_top_usage_out_share: float | None  # 0..1, weighted by missing teammates
    high_usage_teammate_out: int | None
    is_starter_today: int | None  # if known from injury / lineup data

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


def compute_player_usage(
    player_log: pd.DataFrame,
    as_of: date,
) -> dict[str, float | int | None]:
    """
    player_log columns expected: game_date, minutes, usage_rate (optional), started (optional).
    """
    if player_log is None or player_log.empty:
        return {
            "usage_rate_l10": None,
            "started_pct_l10": None,
            "minutes_share_l10": None,
        }

    df = player_log.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[df["game_date"] < as_of].sort_values("game_date").tail(10)
    if df.empty:
        return {
            "usage_rate_l10": None,
            "started_pct_l10": None,
            "minutes_share_l10": None,
        }

    out: dict[str, float | int | None] = {}
    if "usage_rate" in df.columns and df["usage_rate"].notna().any():
        out["usage_rate_l10"] = float(df["usage_rate"].dropna().mean())
    else:
        out["usage_rate_l10"] = None

    if "started" in df.columns and df["started"].notna().any():
        out["started_pct_l10"] = float(df["started"].dropna().astype(int).mean())
    else:
        out["started_pct_l10"] = None

    if "minutes" in df.columns and df["minutes"].notna().any():
        out["minutes_share_l10"] = float(df["minutes"].dropna().mean()) / 48.0
    else:
        out["minutes_share_l10"] = None

    return out


def compute_teammate_cascade(
    teammate_usage_l10: dict[int, float],
    teammates_out_ids: set[int],
) -> UsageFeatures:
    """
    teammate_usage_l10: {player_id: l10 mean usage_rate} for each rostered teammate.
    teammates_out_ids: set of player_ids confirmed OUT for the upcoming game.
    """
    if not teammate_usage_l10:
        return UsageFeatures(
            usage_rate_l10=None,
            started_pct_l10=None,
            minutes_share_l10=None,
            teammate_top_usage_out_share=None,
            high_usage_teammate_out=None,
            is_starter_today=None,
        )

    items = sorted(teammate_usage_l10.items(), key=lambda kv: kv[1] or 0.0, reverse=True)
    top3 = [pid for pid, _ in items[:3]]
    top3_out = [pid for pid in top3 if pid in teammates_out_ids]

    total_team_usage = sum(v for v in teammate_usage_l10.values() if v is not None) or 1.0
    out_usage = sum(
        teammate_usage_l10[pid]
        for pid in teammates_out_ids
        if teammate_usage_l10.get(pid) is not None
    )

    return UsageFeatures(
        usage_rate_l10=None,
        started_pct_l10=None,
        minutes_share_l10=None,
        teammate_top_usage_out_share=float(out_usage / total_team_usage),
        high_usage_teammate_out=int(len(top3_out) > 0),
        is_starter_today=None,
    )

"""
Player form / recency features.

CRITICAL: every function takes `as_of: date` and only uses games strictly
before that date. Violating this leaks future info into training.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class PlayerFormFeatures:
    pts_avg_5: float | None
    pts_avg_10: float | None
    pts_avg_season: float | None
    reb_avg_5: float | None
    reb_avg_10: float | None
    ast_avg_5: float | None
    ast_avg_10: float | None
    threes_avg_5: float | None
    threes_avg_10: float | None
    min_avg_5: float | None
    min_avg_10: float | None
    pts_ewma: float | None
    reb_ewma: float | None
    ast_ewma: float | None
    threes_ewma: float | None
    min_ewma: float | None
    pts_std_10: float | None
    reb_std_10: float | None
    ast_std_10: float | None
    games_played_l10: int
    games_played_season: int
    # Phase 5.9 — threes-specific volume + efficiency. The model was using
    # `threes_avg_N` (makes only), which conflates volume and efficiency.
    # Splitting them lets the tree learn that a 6/15 shooter (40% on 15
    # attempts) has a tighter make distribution than a 2/4 shooter (50% on 4
    # attempts) even if both currently average 2 makes.
    threes_attempted_l10: float | None
    threes_pct_l10: float | None      # makes / attempts, or None if <5 cumulative attempts
    fg_share_threes_l10: float | None # threes_attempted / fga, or None if no FGA history

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


# threes_attempted and field_goals_attempted are required to compute the new
# threes-volume / 3p% / share features. If they're missing in the loaded log
# (older rows from before the advanced backfill), the new features fall back
# to None and XGBoost treats them as missing.
REQUIRED_COLS = ("game_date", "points", "rebounds", "assists", "threes_made", "minutes")
_MIN_ATTEMPTS_FOR_PCT = 5  # below this, 3P% is too noisy to be useful


def compute_player_form(
    game_log: pd.DataFrame,
    as_of: date,
    ewma_alpha: float = 0.3,
) -> PlayerFormFeatures:
    """
    Compute form features as of a given date. Uses only games strictly before `as_of`.
    """
    if game_log.empty or not all(c in game_log.columns for c in REQUIRED_COLS):
        return _empty_form_features()

    df = game_log.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[df["game_date"] < as_of].sort_values("game_date")

    if df.empty:
        return _empty_form_features()

    last5 = df.tail(5)
    last10 = df.tail(10)

    def _safe_mean(s: pd.Series) -> float | None:
        s = s.dropna()
        return float(s.mean()) if len(s) > 0 else None

    def _safe_std(s: pd.Series) -> float | None:
        s = s.dropna()
        return float(s.std(ddof=1)) if len(s) > 1 else None

    def _ewma(s: pd.Series) -> float | None:
        s = s.dropna()
        if len(s) == 0:
            return None
        return float(s.ewm(alpha=ewma_alpha, adjust=False).mean().iloc[-1])

    # Threes volume / efficiency / role share. Each is optional — older rows
    # may not have threes_attempted or field_goals_attempted populated.
    threes_att_l10 = (
        _safe_mean(last10["threes_attempted"])
        if "threes_attempted" in last10.columns else None
    )
    threes_pct_l10 = _safe_rate(
        last10.get("threes_made"), last10.get("threes_attempted"),
        min_denom=_MIN_ATTEMPTS_FOR_PCT,
    )
    fg_share_threes_l10 = _safe_rate(
        last10.get("threes_attempted"), last10.get("field_goals_attempted"),
        min_denom=_MIN_ATTEMPTS_FOR_PCT,
    )

    return PlayerFormFeatures(
        pts_avg_5=_safe_mean(last5["points"]),
        pts_avg_10=_safe_mean(last10["points"]),
        pts_avg_season=_safe_mean(df["points"]),
        reb_avg_5=_safe_mean(last5["rebounds"]),
        reb_avg_10=_safe_mean(last10["rebounds"]),
        ast_avg_5=_safe_mean(last5["assists"]),
        ast_avg_10=_safe_mean(last10["assists"]),
        threes_avg_5=_safe_mean(last5["threes_made"]),
        threes_avg_10=_safe_mean(last10["threes_made"]),
        min_avg_5=_safe_mean(last5["minutes"]),
        min_avg_10=_safe_mean(last10["minutes"]),
        pts_ewma=_ewma(df["points"]),
        reb_ewma=_ewma(df["rebounds"]),
        ast_ewma=_ewma(df["assists"]),
        threes_ewma=_ewma(df["threes_made"]),
        min_ewma=_ewma(df["minutes"]),
        pts_std_10=_safe_std(last10["points"]),
        reb_std_10=_safe_std(last10["rebounds"]),
        ast_std_10=_safe_std(last10["assists"]),
        games_played_l10=int(len(last10)),
        games_played_season=int(len(df)),
        threes_attempted_l10=threes_att_l10,
        threes_pct_l10=threes_pct_l10,
        fg_share_threes_l10=fg_share_threes_l10,
    )


def _safe_rate(
    numer: pd.Series | None,
    denom: pd.Series | None,
    *,
    min_denom: int,
) -> float | None:
    """Sum-based rate over the window with a minimum-attempts guard.

    Returns None if either series is missing, or if the total denominator
    falls below min_denom (the rate would be too noisy to be useful).
    """
    if numer is None or denom is None:
        return None
    n = pd.to_numeric(numer, errors="coerce").dropna()
    d = pd.to_numeric(denom, errors="coerce").dropna()
    if d.empty or d.sum() < min_denom:
        return None
    # Align lengths defensively in case the two series came from different
    # column slices with non-matching NaN patterns.
    paired = pd.concat([n, d], axis=1, join="inner").dropna()
    if paired.empty or paired.iloc[:, 1].sum() < min_denom:
        return None
    return float(paired.iloc[:, 0].sum() / paired.iloc[:, 1].sum())


def _empty_form_features() -> PlayerFormFeatures:
    return PlayerFormFeatures(
        pts_avg_5=None, pts_avg_10=None, pts_avg_season=None,
        reb_avg_5=None, reb_avg_10=None,
        ast_avg_5=None, ast_avg_10=None,
        threes_avg_5=None, threes_avg_10=None,
        min_avg_5=None, min_avg_10=None,
        pts_ewma=None, reb_ewma=None, ast_ewma=None, threes_ewma=None, min_ewma=None,
        pts_std_10=None, reb_std_10=None, ast_std_10=None,
        games_played_l10=0, games_played_season=0,
        threes_attempted_l10=None, threes_pct_l10=None, fg_share_threes_l10=None,
    )

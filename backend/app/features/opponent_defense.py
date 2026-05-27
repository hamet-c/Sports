"""
Opponent-defense features computed from a TeamGameStats log.

Cardinal rule: only use games strictly before `as_of`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class OpponentDefenseFeatures:
    opp_def_rtg_l10: float | None
    opp_def_rtg_season: float | None
    opp_pts_allowed_l10: float | None
    opp_pace_l10: float | None
    opp_pace_season: float | None
    opp_threes_allowed_l10: float | None
    opp_3p_attempted_allowed_l10: float | None
    opp_games_played_l10: int
    # Phase 5.9 — threes-specific opponent quality.
    # opp_3p_pct_allowed_l10 = makes_allowed / attempts_allowed (efficiency,
    #   not just raw count). Distinguishes "defense gives up 12/30 (40%)"
    #   from "defense gives up 12/40 (30%)" — same allowed count, very
    #   different defensive quality.
    # opp_3pa_allowed_per_pace_l10 = attempts_allowed / pace. Pace-normalises
    #   3PA-allowed so fast-paced defenses don't look weak just because they
    #   play more possessions.
    opp_3p_pct_allowed_l10: float | None
    opp_3pa_allowed_per_pace_l10: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


_MIN_ATTEMPTS_FOR_PCT = 20  # opp 3PA-allowed over L10; below this 3P% too noisy


def compute_opponent_defense(
    team_log: pd.DataFrame,
    as_of: date,
) -> OpponentDefenseFeatures:
    """team_log: rows of TeamGameStats for the OPPONENT team."""
    if team_log.empty:
        return _empty()

    df = team_log.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[df["game_date"] < as_of].sort_values("game_date")
    if df.empty:
        return _empty()

    last10 = df.tail(10)

    def _mean(s: pd.Series) -> float | None:
        s = s.dropna()
        return float(s.mean()) if len(s) > 0 else None

    # Threes-specific opponent quality. Sum-based rates with min-attempts
    # guard for 3P%; per-pace rate for attempts allowed.
    threes_made = pd.to_numeric(
        last10.get("threes_allowed", pd.Series(dtype=float)), errors="coerce",
    ).dropna()
    threes_att = pd.to_numeric(
        last10.get("threes_attempted_allowed", pd.Series(dtype=float)), errors="coerce",
    ).dropna()
    pace = pd.to_numeric(
        last10.get("pace", pd.Series(dtype=float)), errors="coerce",
    ).dropna()

    if not threes_att.empty and threes_att.sum() >= _MIN_ATTEMPTS_FOR_PCT and not threes_made.empty:
        opp_3p_pct = float(threes_made.sum() / threes_att.sum())
    else:
        opp_3p_pct = None

    if not threes_att.empty and not pace.empty:
        # Mean of per-game ratio (attempts / pace) — robust to one weird game.
        paired = pd.concat([threes_att, pace], axis=1, join="inner").dropna()
        paired = paired[paired.iloc[:, 1] > 0]
        if not paired.empty:
            opp_3pa_per_pace = float(
                (paired.iloc[:, 0] / paired.iloc[:, 1]).mean()
            )
        else:
            opp_3pa_per_pace = None
    else:
        opp_3pa_per_pace = None

    return OpponentDefenseFeatures(
        opp_def_rtg_l10=_mean(last10.get("def_rating", pd.Series(dtype=float))),
        opp_def_rtg_season=_mean(df.get("def_rating", pd.Series(dtype=float))),
        opp_pts_allowed_l10=_mean(last10.get("points_allowed", pd.Series(dtype=float))),
        opp_pace_l10=_mean(last10.get("pace", pd.Series(dtype=float))),
        opp_pace_season=_mean(df.get("pace", pd.Series(dtype=float))),
        opp_threes_allowed_l10=_mean(last10.get("threes_allowed", pd.Series(dtype=float))),
        opp_3p_attempted_allowed_l10=_mean(
            last10.get("threes_attempted_allowed", pd.Series(dtype=float))
        ),
        opp_games_played_l10=int(len(last10)),
        opp_3p_pct_allowed_l10=opp_3p_pct,
        opp_3pa_allowed_per_pace_l10=opp_3pa_per_pace,
    )


def _empty() -> OpponentDefenseFeatures:
    return OpponentDefenseFeatures(
        opp_def_rtg_l10=None,
        opp_def_rtg_season=None,
        opp_pts_allowed_l10=None,
        opp_pace_l10=None,
        opp_pace_season=None,
        opp_threes_allowed_l10=None,
        opp_3p_attempted_allowed_l10=None,
        opp_games_played_l10=0,
        opp_3p_pct_allowed_l10=None,
        opp_3pa_allowed_per_pace_l10=None,
    )

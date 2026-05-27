"""
Game-context features: rest days, b2b, home/away, playoff flag.

These don't require historical aggregation, only the player's most recent
prior game date relative to `as_of`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class GameContextFeatures:
    is_home: int | None
    rest_days: int | None
    is_b2b: int | None
    games_in_last_3_days: int
    games_in_last_7_days: int
    is_playoff: int | None

    def to_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


# NBA game_id prefixes — "00" + a 2-digit season-type code, then 5 digits of
# season-relative id. 0022 = regular season, 0042 = playoffs, 0052 = play-in,
# 0012 = preseason. Slate-time stubs created by prop_ingest have no nba_id at
# all; we fall back to a date window for those.
_PLAYOFF_PREFIX = "0042"
_KNOWN_NON_PLAYOFF_PREFIXES = ("0022", "0012", "0052")
# Inclusive [start, end] window for playoff games. Covers play-in (mid-April)
# through Finals (mid-June). Wider than reality on purpose — false positives
# here become rare RS rows tagged as playoff, which the prefix-based path
# overrides whenever nba_id is present.
_PLAYOFF_START_MONTH_DAY = (4, 15)
_PLAYOFF_END_MONTH_DAY = (6, 25)


def classify_is_playoff(nba_id: str | None, game_date: date | None) -> int | None:
    """
    Return 1 if the game is a playoff game, 0 if regular season / preseason /
    play-in, None if neither signal is available.

    Prefers the NBA `game_id` prefix when present (deterministic). Falls back
    to a date window so slate-time stubs (no nba_id yet) still get classified
    — without that fallback, every prop_ingest row would feed the model None,
    erasing the feature exactly when we need it.
    """
    if nba_id:
        if nba_id.startswith(_PLAYOFF_PREFIX):
            return 1
        if nba_id.startswith(_KNOWN_NON_PLAYOFF_PREFIXES):
            return 0
        # Unknown prefix — fall through to the date heuristic rather than
        # silently classifying as 0.
    if game_date is None:
        return None
    md = (game_date.month, game_date.day)
    return 1 if _PLAYOFF_START_MONTH_DAY <= md <= _PLAYOFF_END_MONTH_DAY else 0


def compute_game_context(
    game_dates: pd.Series,
    as_of: date,
    is_home: bool | None,
    nba_id: str | None = None,
) -> GameContextFeatures:
    """
    game_dates: a series of dates for player's prior games.
    as_of: the date of the upcoming game we're predicting.
    nba_id: external NBA game id for the upcoming game (when known). Used
        with `as_of` to set the playoff flag.
    """
    is_playoff = classify_is_playoff(nba_id, as_of)
    if game_dates is None or len(game_dates) == 0:
        return GameContextFeatures(
            is_home=int(is_home) if is_home is not None else None,
            rest_days=None,
            is_b2b=None,
            games_in_last_3_days=0,
            games_in_last_7_days=0,
            is_playoff=is_playoff,
        )

    s = pd.to_datetime(pd.Series(game_dates)).dt.date
    s = s[s < as_of].sort_values()

    if s.empty:
        return GameContextFeatures(
            is_home=int(is_home) if is_home is not None else None,
            rest_days=None,
            is_b2b=None,
            games_in_last_3_days=0,
            games_in_last_7_days=0,
            is_playoff=is_playoff,
        )

    last_game = s.iloc[-1]
    rest = (as_of - last_game).days
    games_3 = int(((as_of - s).map(lambda d: d.days) <= 3).sum())
    games_7 = int(((as_of - s).map(lambda d: d.days) <= 7).sum())

    return GameContextFeatures(
        is_home=int(is_home) if is_home is not None else None,
        rest_days=int(rest),
        is_b2b=int(rest <= 1),
        games_in_last_3_days=games_3,
        games_in_last_7_days=games_7,
        is_playoff=is_playoff,
    )

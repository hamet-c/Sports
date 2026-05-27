"""
Critical test: features must NEVER use information from on or after `as_of`.
If this test ever fails, treat it as a P0 leakage bug. Fix the feature, not the test.
"""
from datetime import date, datetime

import pandas as pd

from app.features.game_context import (
    classify_is_playoff,
    compute_game_context,
)
from app.features.injury_features import compute_injury_features
from app.features.opponent_defense import compute_opponent_defense
from app.features.player_form import compute_player_form


def _toy_player_log(dates: list[str], pts: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "game_date": pd.to_datetime(dates).date,
        "points": pts,
        "rebounds": [5] * len(pts),
        "assists": [3] * len(pts),
        "threes_made": [1] * len(pts),
        "minutes": [30.0] * len(pts),
    })


# -------------------------------- player form ------------------------------- #

def test_form_excludes_on_and_after_as_of():
    log = _toy_player_log(
        ["2024-01-01", "2024-01-03", "2024-01-05"],
        [10, 20, 999],  # 999 must not be used if as_of <= 2024-01-05
    )
    feats = compute_player_form(log, as_of=date(2024, 1, 5))
    assert feats.pts_avg_season == 15.0
    assert feats.pts_avg_season != (10 + 20 + 999) / 3


def test_form_includes_strictly_before_as_of():
    log = _toy_player_log(["2024-01-01", "2024-01-03"], [10, 20])
    feats = compute_player_form(log, as_of=date(2024, 1, 4))
    assert feats.pts_avg_season == 15.0


def test_form_empty_log_returns_none_features():
    feats = compute_player_form(pd.DataFrame(), as_of=date(2024, 1, 1))
    assert feats.pts_avg_5 is None
    assert feats.games_played_season == 0


def test_form_last5_window_only():
    log = _toy_player_log(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
         "2024-01-06", "2024-01-07"],
        [50, 50, 0, 0, 0, 0, 0],
    )
    # as_of after all: last5 should be only the last 5 -> all zeros
    feats = compute_player_form(log, as_of=date(2024, 1, 8))
    assert feats.pts_avg_5 == 0.0


# ----------------------- threes-specific player features --------------------- #

def _log_with_threes(
    dates: list[str],
    threes_made: list[int],
    threes_att: list[int],
    fga: list[int],
) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "game_date": pd.to_datetime(dates).date,
        "points": [10] * n,
        "rebounds": [5] * n,
        "assists": [3] * n,
        "threes_made": threes_made,
        "threes_attempted": threes_att,
        "field_goals_attempted": fga,
        "minutes": [30.0] * n,
    })


def test_threes_pct_uses_sum_based_rate():
    # Three games, 6 makes / 20 attempts overall = 30%. A row-mean-of-rates
    # would give different answer if any single game had zero attempts.
    log = _log_with_threes(
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        threes_made=[2, 4, 0],
        threes_att=[5, 10, 5],
        fga=[15, 20, 10],
    )
    feats = compute_player_form(log, as_of=date(2024, 1, 4))
    assert feats.threes_attempted_l10 == (5 + 10 + 5) / 3
    assert abs(feats.threes_pct_l10 - 6 / 20) < 1e-9
    # fg_share = 20 attempts / 45 fga
    assert abs(feats.fg_share_threes_l10 - 20 / 45) < 1e-9


def test_threes_pct_none_when_low_attempts():
    # Total of 3 attempts across the window — below the noise guard for 3P%.
    log = _log_with_threes(
        ["2024-01-01", "2024-01-02"],
        threes_made=[1, 1],
        threes_att=[1, 2],
        fga=[10, 10],
    )
    feats = compute_player_form(log, as_of=date(2024, 1, 3))
    # 3P% is too noisy on 3 attempts -> None
    assert feats.threes_pct_l10 is None
    # fg_share is still meaningful even with low threes_att — a center who
    # almost never shoots threes has share ≈ 0, which is real signal, not noise.
    # Denominator here (FGA=20) is well above the guard.
    assert abs(feats.fg_share_threes_l10 - 3 / 20) < 1e-9
    # Volume is also meaningful
    assert feats.threes_attempted_l10 == 1.5


def test_threes_features_none_when_columns_missing():
    # Old-style log without threes_attempted / field_goals_attempted
    log = _toy_player_log(["2024-01-01", "2024-01-02"], [10, 12])
    feats = compute_player_form(log, as_of=date(2024, 1, 3))
    assert feats.threes_attempted_l10 is None
    assert feats.threes_pct_l10 is None
    assert feats.fg_share_threes_l10 is None
    # Existing fields still populated
    assert feats.pts_avg_season == 11.0


# ------------------------------ opponent defense ---------------------------- #

def test_opponent_defense_excludes_future():
    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]).date,
        "def_rating": [110.0, 112.0, 999.0],
        "points_allowed": [115, 110, 999],
        "pace": [98.0, 99.0, 999.0],
        "threes_made": [10, 12, 99],
        "threes_attempted": [25, 30, 99],
        "threes_allowed": [11, 13, 99],
        "threes_attempted_allowed": [26, 31, 99],
    })
    feats = compute_opponent_defense(df, as_of=date(2024, 1, 10))
    assert feats.opp_def_rtg_l10 == 111.0
    assert feats.opp_def_rtg_l10 != (110 + 112 + 999) / 3


def test_opponent_threes_pct_allowed_sum_based():
    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-08"]).date,
        "pace": [98.0, 99.0, 100.0],
        "threes_allowed": [10, 12, 14],
        "threes_attempted_allowed": [30, 30, 40],
    })
    feats = compute_opponent_defense(df, as_of=date(2024, 1, 10))
    # 36/100 = 36% allowed
    assert abs(feats.opp_3p_pct_allowed_l10 - 36 / 100) < 1e-9
    # per-pace: mean of (30/98, 30/99, 40/100) = mean(0.306, 0.303, 0.400)
    expected = (30 / 98 + 30 / 99 + 40 / 100) / 3
    assert abs(feats.opp_3pa_allowed_per_pace_l10 - expected) < 1e-9


def test_opponent_threes_pct_none_when_few_attempts():
    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-01-01"]).date,
        "pace": [98.0],
        "threes_allowed": [3],
        "threes_attempted_allowed": [10],  # < 20 threshold
    })
    feats = compute_opponent_defense(df, as_of=date(2024, 1, 10))
    assert feats.opp_3p_pct_allowed_l10 is None
    # Per-pace is still meaningful even on small samples
    assert feats.opp_3pa_allowed_per_pace_l10 is not None


def test_opponent_threes_features_exclude_future():
    # The future row at 2024-01-10 carries fake-massive 3PA so we can verify
    # it does NOT flow into the as_of=2024-01-10 (strictly-before) window.
    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]).date,
        "pace": [98.0, 99.0, 100.0],
        "threes_allowed": [10, 12, 9999],
        "threes_attempted_allowed": [30, 30, 9999],
    })
    feats = compute_opponent_defense(df, as_of=date(2024, 1, 10))
    # Only first two rows contribute: 22/60
    assert abs(feats.opp_3p_pct_allowed_l10 - 22 / 60) < 1e-9


# -------------------------------- game context ------------------------------ #

def test_game_context_rest_days():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"]).date
    feats = compute_game_context(pd.Series(dates), as_of=date(2024, 1, 5), is_home=True)
    assert feats.rest_days == 2
    assert feats.is_b2b == 0
    assert feats.is_home == 1


def test_game_context_b2b():
    feats = compute_game_context(
        pd.Series(pd.to_datetime(["2024-01-04"]).date),
        as_of=date(2024, 1, 5),
        is_home=False,
    )
    assert feats.is_b2b == 1
    assert feats.rest_days == 1


# ---------------------------------- playoff --------------------------------- #

def test_classify_is_playoff_prefix_wins_over_date():
    # 0042 prefix says playoff even though Dec 1 is well outside the date window.
    assert classify_is_playoff("0042200401", date(2024, 12, 1)) == 1


def test_classify_is_playoff_regular_season_prefix():
    assert classify_is_playoff("0022300456", date(2024, 5, 1)) == 0


def test_classify_is_playoff_date_fallback_when_no_nba_id():
    # No nba_id (slate-time stub) — must fall back to date window.
    assert classify_is_playoff(None, date(2026, 5, 12)) == 1
    assert classify_is_playoff(None, date(2026, 1, 15)) == 0


def test_classify_is_playoff_none_when_no_signal():
    assert classify_is_playoff(None, None) is None


def test_game_context_emits_is_playoff_from_nba_id():
    feats = compute_game_context(
        pd.Series(pd.to_datetime(["2024-01-04"]).date),
        as_of=date(2024, 1, 5),
        is_home=True,
        nba_id="0042300101",
    )
    assert feats.is_playoff == 1


def test_game_context_emits_is_playoff_from_date_when_id_missing():
    feats = compute_game_context(
        pd.Series(pd.to_datetime(["2026-05-10"]).date),
        as_of=date(2026, 5, 12),
        is_home=True,
        nba_id=None,
    )
    assert feats.is_playoff == 1


# ---------------------------------- injury ---------------------------------- #

def test_injury_status_uses_latest_before_as_of():
    df = pd.DataFrame({
        "report_datetime": pd.to_datetime([
            "2024-01-01 09:00",
            "2024-01-04 14:00",
            "2024-01-05 18:00",
        ]),
        "status": ["QUESTIONABLE", "OUT", "AVAILABLE"],
    })
    feats = compute_injury_features(df, as_of_dt=datetime(2024, 1, 5, 12, 0))
    assert feats.is_out == 1
    assert feats.is_questionable == 0

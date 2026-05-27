"""
Wrapper around `nba_api`.

Why a wrapper:
    1. Centralizes rate limiting & retry behavior
    2. Returns clean DataFrames with consistent column names
    3. Easy to mock in tests
    4. Single place to swap data source if nba_api breaks

Reference: https://github.com/swar/nba_api
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings


class NBADataClient:
    def __init__(self, request_delay_seconds: float | None = None) -> None:
        self.request_delay = (
            request_delay_seconds
            if request_delay_seconds is not None
            else settings.nba_api_min_delay_seconds
        )
        self._last_request_at: float = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_player_game_log(
        self, player_id: int, season: str, season_type: str = "Regular Season",
    ) -> pd.DataFrame:
        from nba_api.stats.endpoints import playergamelog
        self._throttle()
        logger.debug(f"Fetching game log: player={player_id} season={season}")
        response = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
            timeout=settings.nba_api_request_timeout,
        )
        df = response.get_data_frames()[0]
        return self._normalize_player_game_log(df)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_team_game_log(
        self, team_id: int, season: str, season_type: str = "Regular Season",
    ) -> pd.DataFrame:
        from nba_api.stats.endpoints import teamgamelog
        self._throttle()
        df = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
            timeout=settings.nba_api_request_timeout,
        ).get_data_frames()[0]
        return self._normalize_team_game_log(df)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_team_advanced(self, team_id: int, season: str) -> pd.DataFrame:
        """Per-game advanced (pace, ortg, drtg) — used for opponent context."""
        from nba_api.stats.endpoints import teamdashboardbygeneralsplits
        self._throttle()
        return teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(
            team_id=team_id,
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=settings.nba_api_request_timeout,
        ).get_data_frames()[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_active_players(self) -> pd.DataFrame:
        from nba_api.stats.static import players
        return pd.DataFrame(players.get_active_players())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_all_teams(self) -> pd.DataFrame:
        from nba_api.stats.static import teams
        return pd.DataFrame(teams.get_teams())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_common_player_info(self, player_id: int) -> pd.DataFrame:
        from nba_api.stats.endpoints import commonplayerinfo
        self._throttle()
        return commonplayerinfo.CommonPlayerInfo(
            player_id=player_id, timeout=settings.nba_api_request_timeout,
        ).get_data_frames()[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_scoreboard(self, game_date: date) -> pd.DataFrame:
        from nba_api.stats.endpoints import scoreboardv2
        self._throttle()
        response = scoreboardv2.ScoreboardV2(
            game_date=game_date.strftime("%m/%d/%Y"),
            timeout=settings.nba_api_request_timeout,
        )
        return response.get_data_frames()[0]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    def get_box_score_advanced(self, game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Advanced box score for a single game.

        Uses BoxScoreAdvancedV3. The v2 endpoint started returning empty `{}`
        in 2026 — stats.nba.com has migrated this resource to the nba.cloud
        backend exposed through v3. Field names are camelCase (`personId`,
        `usagePercentage`, `position`, `offensiveRating`, `defensiveRating`,
        `pace`) rather than v2's screaming-snake-case.

        Returns ``(player_stats, team_stats)``.
        """
        from nba_api.stats.endpoints.boxscoreadvancedv3 import BoxScoreAdvancedV3
        self._throttle()
        ep = BoxScoreAdvancedV3(
            game_id=game_id, timeout=settings.nba_api_request_timeout,
        )
        return ep.player_stats.get_data_frame(), ep.team_stats.get_data_frame()

    # ----------------------------- normalization ---------------------------- #

    @staticmethod
    def _normalize_player_game_log(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # nba_api endpoints are inconsistent in casing across endpoints (e.g.
        # PlayerGameLog uses GAME_ID while TeamGameLog uses Game_ID). Uppercase
        # everything once so the rename map is the single source of truth.
        df.columns = [c.upper() for c in df.columns]
        df["GAME_DATE"] = _parse_game_date(df["GAME_DATE"])
        rename_map = {
            "GAME_ID": "game_id_nba",
            "PLAYER_ID": "player_id_nba",
            "PTS": "points",
            "REB": "rebounds",
            "AST": "assists",
            "STL": "steals",
            "BLK": "blocks",
            "TOV": "turnovers",
            "FG3M": "threes_made",
            "FG3A": "threes_attempted",
            "FGM": "field_goals_made",
            "FGA": "field_goals_attempted",
            "FTM": "free_throws_made",
            "FTA": "free_throws_attempted",
            "MIN": "minutes",
            "PLUS_MINUS": "plus_minus",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "win_loss",
            "SEASON_ID": "season_id",
        }
        return df.rename(columns=rename_map)

    @staticmethod
    def _normalize_team_game_log(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df.columns = [c.upper() for c in df.columns]
        df["GAME_DATE"] = _parse_game_date(df["GAME_DATE"])
        rename_map = {
            "GAME_ID": "game_id_nba",
            "TEAM_ID": "team_id_nba",
            "PTS": "points",
            "REB": "rebounds",
            "AST": "assists",
            "FG3M": "threes_made",
            "FG3A": "threes_attempted",
            "FGM": "field_goals_made",
            "FGA": "field_goals_attempted",
            "MIN": "minutes",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "win_loss",
        }
        return df.rename(columns=rename_map)


def _parse_game_date(series: pd.Series) -> pd.Series:
    """
    nba_api returns GAME_DATE in a few formats depending on the endpoint
    ('OCT 24, 2023', '2023-10-24T00:00:00', etc.). Try the most common
    explicit format first to avoid the dateutil fallback warning, then fall
    back to general parsing for anything that didn't match.
    """
    parsed = pd.to_datetime(series, format="%b %d, %Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(series[missing], errors="coerce")
        parsed = parsed.copy()
        parsed.loc[missing] = fallback
    return parsed


nba_client = NBADataClient()

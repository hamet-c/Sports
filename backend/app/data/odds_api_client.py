"""
Sportsbook lines via The Odds API (https://the-odds-api.com).
Free tier: 500 requests/month — cache aggressively.

Markets:
    - player_points
    - player_rebounds
    - player_assists
    - player_threes
    - player_points_rebounds_assists
"""
from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.core.config import settings


class OddsAPIClient:
    SUPPORTED_MARKETS = [
        "player_points",
        "player_rebounds",
        "player_assists",
        "player_threes",
        "player_points_rebounds_assists",
    ]

    # Map odds-api market keys to our internal stat_type strings.
    MARKET_TO_STAT_TYPE = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "threes_made",
        "player_points_rebounds_assists": "pra",
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.odds_api_key
        self.base_url = settings.odds_api_base
        self._client = httpx.Client(timeout=20.0)

    def _require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "ODDS_API_KEY missing — set it in .env to use the Odds API client."
            )

    def get_nba_events(self) -> list[dict[str, Any]]:
        self._require_key()
        url = f"{self.base_url}/sports/basketball_nba/events"
        r = self._client.get(url, params={"apiKey": self.api_key})
        r.raise_for_status()
        return r.json()

    def get_nba_odds(self, markets: list[str] | None = None) -> list[dict[str, Any]]:
        """Game-level h2h/spread/totals — used for vegas-signal features."""
        self._require_key()
        url = f"{self.base_url}/sports/basketball_nba/odds"
        r = self._client.get(
            url,
            params={
                "apiKey": self.api_key,
                "regions": "us",
                "markets": ",".join(markets or ["spreads", "totals", "h2h"]),
                "oddsFormat": "american",
            },
        )
        r.raise_for_status()
        return r.json()

    def get_player_props(
        self,
        event_id: str,
        markets: list[str] | None = None,
        bookmakers: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_key()
        markets = markets or self.SUPPORTED_MARKETS
        url = f"{self.base_url}/sports/basketball_nba/events/{event_id}/odds"
        params: dict[str, Any] = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        r = self._client.get(url, params=params)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info(f"Odds API requests remaining: {remaining}")
        return r.json()

    def __enter__(self) -> "OddsAPIClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.close()


odds_client = OddsAPIClient()

"""
Injury report scraper.

Pulls the ESPN NBA injuries feed. ESPN exposes a public JSON endpoint
that returns by-team injury rows, which is far more reliable than
HTML scraping.

    https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries

Status normalization to a fixed vocabulary:
    OUT, DOUBTFUL, QUESTIONABLE, PROBABLE, DAY_TO_DAY, AVAILABLE
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


_STATUS_MAP = {
    "out": "OUT",
    "doubtful": "DOUBTFUL",
    "questionable": "QUESTIONABLE",
    "probable": "PROBABLE",
    "day-to-day": "DAY_TO_DAY",
    "day to day": "DAY_TO_DAY",
    "available": "AVAILABLE",
}


@dataclass
class InjuryRecord:
    player_name: str
    team_abbreviation: str | None
    status: str
    description: str | None
    report_datetime: datetime


class ESPNInjuryScraper:
    URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

    def __init__(self) -> None:
        self._client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "nba-props/0.1 (+https://localhost)"},
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _fetch(self) -> dict[str, Any]:
        r = self._client.get(self.URL)
        r.raise_for_status()
        return r.json()

    def fetch_all(self) -> list[InjuryRecord]:
        payload = self._fetch()
        captured_at = datetime.utcnow()
        out: list[InjuryRecord] = []
        for team in payload.get("injuries", []):
            team_abbr = (team.get("team") or {}).get("abbreviation")
            for entry in team.get("injuries", []):
                athlete = entry.get("athlete") or {}
                name = athlete.get("displayName")
                if not name:
                    continue
                raw_status = (entry.get("status") or "").strip().lower()
                status = _STATUS_MAP.get(raw_status, raw_status.upper() or "UNKNOWN")
                description = (
                    (entry.get("longComment") or entry.get("shortComment") or "").strip()
                    or None
                )
                out.append(
                    InjuryRecord(
                        player_name=name,
                        team_abbreviation=team_abbr,
                        status=status,
                        description=description,
                        report_datetime=captured_at,
                    )
                )
        logger.info(f"ESPN injuries: {len(out)} records")
        return out

    def __enter__(self) -> "ESPNInjuryScraper":
        return self

    def __exit__(self, *_: Any) -> None:
        self._client.close()


injury_scraper = ESPNInjuryScraper()

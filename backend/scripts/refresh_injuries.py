"""
Refresh injury report rows from ESPN.

Maps each ESPN injury entry to a Player by `full_name` (case-insensitive,
exact match). Unmatched names are logged so you can patch the player table.

    python scripts/refresh_injuries.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from app.core.logging import configure_logging
from app.data.injury_scraper import injury_scraper
from app.db.models import InjuryReport, Player
from app.db.session import SessionLocal, init_db


def _name_index(db) -> dict[str, int]:
    return {p.full_name.lower(): p.id for p in db.query(Player).all()}


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        with injury_scraper as s:
            records = s.fetch_all()
        idx = _name_index(db)
        inserted = 0
        unmatched = 0
        rows: list[InjuryReport] = []
        for rec in records:
            pid = idx.get(rec.player_name.lower())
            if pid is None:
                unmatched += 1
                continue
            rows.append(
                InjuryReport(
                    player_id=pid,
                    report_datetime=rec.report_datetime,
                    status=rec.status,
                    description=rec.description,
                )
            )
        db.add_all(rows)
        db.commit()
        inserted = len(rows)
        logger.info(f"Inserted {inserted} injury rows; {unmatched} names unmatched")
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    main()

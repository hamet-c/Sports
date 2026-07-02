"""
Ingest player props + game-level lines from The Odds API into the DB.

    python scripts/ingest_props.py                   # today's slate
    python scripts/ingest_props.py --date 2024-04-12 # backfill specific date

Per Odds API event we:
    1. Ensure a Game row exists (matched by date + home/away team full_name).
       If absent, we create a placeholder Game so the props can attach.
    2. Pull player props for points / rebounds / assists / threes.
    3. For each (player, book, market), pair the Over and Under outcomes
       and insert a PropLine row.
    4. Pull spreads/totals into GameMarket rows for the vegas-signal features.

Free-tier note: each event burns one request for player props plus a
shared one for the game odds list. 8 games ≈ 9 requests.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from app.core.logging import configure_logging
from app.core.timeutil import utcnow
from app.data.odds_api_client import OddsAPIClient, odds_client
from app.db.models import Game, GameMarket, Player, PropLine, Team
from app.db.session import SessionLocal, init_db


_NBA_TZ = ZoneInfo("America/New_York")


def _local_date_for_event(commence_iso: str) -> date:
    """Odds API commence_time is UTC ISO; NBA convention indexes a game by
    its US Eastern calendar date (a 7pm ET tipoff is part of *that* night's
    slate, not the UTC day that already started). Convert to ET before
    extracting the date so we line up with nba_api's game_date column."""
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00")).astimezone(_NBA_TZ)
    return dt.date()


def _team_index(db) -> dict[str, Team]:
    """full_name -> Team."""
    return {t.full_name: t for t in db.query(Team).all()}


def _player_index(db) -> dict[str, Player]:
    """lowercase full_name -> Player. Active first, falls back to all."""
    return {p.full_name.lower(): p for p in db.query(Player).all()}


def _ensure_game(
    db,
    *,
    game_date: date,
    home_team: Team,
    away_team: Team,
    season_hint: str,
) -> Game:
    matches = (
        db.query(Game)
        .filter(Game.game_date == game_date)
        .filter(Game.home_team_id == home_team.id)
        .filter(Game.away_team_id == away_team.id)
        .all()
    )
    if matches:
        # Stub/real twins can coexist until reconcile_stub_games.py merges
        # them — prefer the real (nba_id-bearing) row so new props attach to
        # the game that has actuals.
        for g in matches:
            if g.nba_id is not None:
                return g
        return matches[0]
    # Some games (today / future slate) don't exist yet. Create a placeholder
    # without an nba_id; bootstrap_data will adopt it when the season's game
    # logs come in.
    g = Game(
        sport="nba",
        season=season_hint,
        game_date=game_date,
        home_team_id=home_team.id,
        away_team_id=away_team.id,
        is_completed=False,
    )
    db.add(g)
    db.flush()
    return g


def _season_for_date(d: date) -> str:
    """NBA season label like '2024-25' for any date in that season."""
    if d.month >= 10:
        return f"{d.year}-{str(d.year + 1)[-2:]}"
    return f"{d.year - 1}-{str(d.year)[-2:]}"


def _pair_over_under_outcomes(outcomes: list[dict]) -> dict[tuple[str, float], dict[str, int]]:
    """
    Group outcomes from a single market into {(player_name, line): {"Over": price, "Under": price}}.

    Odds API player-prop outcomes have shape:
        {"name": "Over"|"Under", "description": "<player full name>", "price": -110, "point": 25.5}
    """
    grouped: dict[tuple[str, float], dict[str, int]] = defaultdict(dict)
    for o in outcomes:
        side = o.get("name")
        player_name = o.get("description")
        line = o.get("point")
        price = o.get("price")
        if not (side in ("Over", "Under") and player_name and line is not None and price is not None):
            continue
        grouped[(player_name, float(line))][side] = int(price)
    return grouped


def ingest_event_props(
    db,
    client: OddsAPIClient,
    event: dict[str, Any],
    *,
    teams_by_name: dict[str, Team],
    players_by_lower: dict[str, Player],
    target_date: date | None,
) -> tuple[int, int, int]:
    """
    Returns (props_inserted, players_unmatched, books_seen).
    """
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    commence = event.get("commence_time")
    if not (home_name and away_name and commence):
        return 0, 0, 0

    event_date = _local_date_for_event(commence)
    if target_date is not None and event_date != target_date:
        return 0, 0, 0

    home = teams_by_name.get(home_name)
    away = teams_by_name.get(away_name)
    if home is None or away is None:
        logger.warning(f"Unknown teams in event: home={home_name!r} away={away_name!r}")
        return 0, 0, 0

    game = _ensure_game(
        db,
        game_date=event_date,
        home_team=home,
        away_team=away,
        season_hint=_season_for_date(event_date),
    )

    payload = client.get_player_props(event["id"])
    bookmakers = payload.get("bookmakers", [])
    rows: list[PropLine] = []
    unmatched: set[str] = set()
    captured_at = utcnow()

    for bm in bookmakers:
        book = bm.get("key", "")
        for market in bm.get("markets", []):
            stat = client.MARKET_TO_STAT_TYPE.get(market.get("key", ""))
            if not stat:
                continue
            grouped = _pair_over_under_outcomes(market.get("outcomes", []))
            for (player_name, line), prices in grouped.items():
                if "Over" not in prices or "Under" not in prices:
                    continue
                player = players_by_lower.get(player_name.lower())
                if player is None:
                    unmatched.add(player_name)
                    continue
                rows.append(
                    PropLine(
                        player_id=player.id,
                        game_id=game.id,
                        book=book,
                        stat_type=stat,
                        line=line,
                        over_odds=prices["Over"],
                        under_odds=prices["Under"],
                        captured_at=captured_at,
                    )
                )

    if rows:
        db.bulk_save_objects(rows)
        db.commit()

    if unmatched:
        logger.info(f"{game.id} {away_name}@{home_name}: {len(unmatched)} player names unmatched")
        for n in sorted(unmatched)[:5]:
            logger.debug(f"  unmatched: {n}")

    return len(rows), len(unmatched), len(bookmakers)


def ingest_game_lines(
    db,
    client: OddsAPIClient,
    *,
    teams_by_name: dict[str, Team],
    target_date: date | None,
) -> int:
    """
    Pull spreads/totals for all NBA games and insert GameMarket rows.

    GameMarket has UniqueConstraint(game_id, book, captured_at) so each call
    creates a new snapshot. Use sparingly to conserve free-tier requests.
    """
    games_payload = client.get_nba_odds(markets=["spreads", "totals", "h2h"])
    captured_at = utcnow()
    rows: list[GameMarket] = []

    for ev in games_payload:
        commence = ev.get("commence_time")
        if not commence:
            continue
        ev_date = _local_date_for_event(commence)
        if target_date is not None and ev_date != target_date:
            continue
        home = teams_by_name.get(ev.get("home_team", ""))
        away = teams_by_name.get(ev.get("away_team", ""))
        if home is None or away is None:
            continue
        game = _ensure_game(
            db,
            game_date=ev_date,
            home_team=home,
            away_team=away,
            season_hint=_season_for_date(ev_date),
        )
        for bm in ev.get("bookmakers", []):
            book = bm.get("key", "")
            spread_home: float | None = None
            total: float | None = None
            home_ml: int | None = None
            away_ml: int | None = None
            for m in bm.get("markets", []):
                key = m.get("key")
                outcomes = m.get("outcomes", [])
                if key == "spreads":
                    for o in outcomes:
                        if o.get("name") == ev.get("home_team"):
                            spread_home = o.get("point")
                elif key == "totals":
                    for o in outcomes:
                        if o.get("name") == "Over":
                            total = o.get("point")
                elif key == "h2h":
                    for o in outcomes:
                        if o.get("name") == ev.get("home_team"):
                            home_ml = o.get("price")
                        elif o.get("name") == ev.get("away_team"):
                            away_ml = o.get("price")
            if spread_home is None and total is None and home_ml is None and away_ml is None:
                continue
            rows.append(
                GameMarket(
                    game_id=game.id,
                    book=book,
                    spread_home=spread_home,
                    total=total,
                    home_moneyline=home_ml,
                    away_moneyline=away_ml,
                    captured_at=captured_at,
                )
            )

    if rows:
        db.bulk_save_objects(rows)
        db.commit()
    return len(rows)


def main(target: date | None) -> None:
    init_db()
    db = SessionLocal()
    try:
        with odds_client as client:
            teams_by_name = _team_index(db)
            players_by_lower = _player_index(db)

            # /events is quota-free; check it before the quota-counted /odds
            # call so offseason runs (empty events list) burn zero quota.
            logger.info("Fetching NBA events")
            events = client.get_nba_events()
            logger.info(f"{len(events)} events returned")
            if not events:
                logger.info("No NBA events (offseason or dark day) — nothing to ingest.")
                return

            n_game_lines = ingest_game_lines(
                db, client, teams_by_name=teams_by_name, target_date=target,
            )
            logger.info(f"Inserted {n_game_lines} GameMarket rows")

            total_props = 0
            total_unmatched = 0
            for ev in events:
                try:
                    n, u, _ = ingest_event_props(
                        db,
                        client,
                        ev,
                        teams_by_name=teams_by_name,
                        players_by_lower=players_by_lower,
                        target_date=target,
                    )
                    total_props += n
                    total_unmatched += u
                except Exception as e:
                    logger.warning(f"Event {ev.get('id')} failed: {e}")

            logger.info(
                f"Props ingest complete: {total_props} rows; {total_unmatched} unmatched player names"
            )
    finally:
        db.close()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Restrict to a single UTC date (YYYY-MM-DD). Default: ingest all returned events.",
    )
    args = parser.parse_args()
    main(args.date)

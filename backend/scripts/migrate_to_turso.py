"""One-shot: copy local SQLite DB -> Turso.

Creates schema via SQLAlchemy on the remote, then bulk-inserts every row from
each table in chunks. Idempotent in the sense that you can rerun after fixing
a partial-failure -- it skips rows whose primary keys already exist on remote.

Usage:
    python scripts/migrate_to_turso.py            # full run
    python scripts/migrate_to_turso.py --verify   # row-count check only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.db import models  # noqa: E402, F401  -- register models
from app.db.session import Base  # noqa: E402

CHUNK = 500


def _make_turso_engine():
    if not settings.turso_database_url or not settings.turso_auth_token:
        raise SystemExit("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not set in .env")
    host = settings.turso_database_url.replace("libsql://", "", 1)
    url = f"sqlite+libsql://{host}?authToken={settings.turso_auth_token}&secure=true"
    return create_engine(url, future=True)


def _make_local_engine():
    local_path = settings.project_root / "data" / "nba_props.db"
    if not local_path.exists():
        raise SystemExit(f"Local DB not found at {local_path}")
    return create_engine(f"sqlite:///{local_path.as_posix()}", future=True)


def _table_order() -> list[str]:
    """Topological order so FKs resolve."""
    return [
        "teams",
        "players",
        "games",
        "team_game_stats",
        "player_game_stats",
        "injury_reports",
        "prop_lines",
        "game_markets",
        "predictions",
    ]


def verify(local_engine, turso_engine) -> None:
    print("\nRow-count verification:")
    print(f"{'table':<25} {'local':>10} {'turso':>10}  match")
    print("-" * 60)
    all_match = True
    for table in _table_order():
        with local_engine.connect() as lc, turso_engine.connect() as tc:
            local_n = lc.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            try:
                turso_n = tc.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            except Exception as exc:  # noqa: BLE001
                turso_n = f"ERR ({exc.__class__.__name__})"
        match = local_n == turso_n
        all_match = all_match and match
        print(f"{table:<25} {local_n:>10} {str(turso_n):>10}  {'OK' if match else 'MISMATCH'}")
    print("-" * 60)
    print("All tables match." if all_match else "Mismatches present.")


def migrate() -> None:
    local_engine = _make_local_engine()
    turso_engine = _make_turso_engine()

    print(f"Local : {local_engine.url}")
    print(f"Turso : {settings.turso_database_url}\n")

    print("Creating schema on Turso (no-op if tables exist)...")
    Base.metadata.create_all(bind=turso_engine)
    print("  done.\n")

    LocalSession = sessionmaker(bind=local_engine, future=True)
    TursoSession = sessionmaker(bind=turso_engine, future=True)
    inspector = inspect(local_engine)

    for table_name in _table_order():
        cols = [c["name"] for c in inspector.get_columns(table_name)]
        col_list = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        insert_sql = text(
            f"INSERT OR IGNORE INTO {table_name} ({col_list}) VALUES ({placeholders})"
        )

        with local_engine.connect() as lc:
            total = lc.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
        if total == 0:
            print(f"{table_name:<25} (empty, skip)")
            continue

        print(f"{table_name:<25} migrating {total:,} rows...")
        start = time.time()
        copied = 0
        with LocalSession() as lsess:
            cursor = lsess.execute(text(f"SELECT * FROM {table_name}"))
            batch: list[dict] = []
            for row in cursor.mappings():
                batch.append(dict(row))
                if len(batch) >= CHUNK:
                    with TursoSession() as tsess:
                        tsess.execute(insert_sql, batch)
                        tsess.commit()
                    copied += len(batch)
                    print(f"  ... {copied:,}/{total:,}", end="\r")
                    batch = []
            if batch:
                with TursoSession() as tsess:
                    tsess.execute(insert_sql, batch)
                    tsess.commit()
                copied += len(batch)
        elapsed = time.time() - start
        print(f"  {copied:,} rows in {elapsed:.1f}s ({copied / max(elapsed, 0.01):.0f}/s)")

    print("\nDone. Run with --verify to confirm row counts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Only run row-count verification")
    args = parser.parse_args()

    if args.verify:
        verify(_make_local_engine(), _make_turso_engine())
        return

    migrate()
    verify(_make_local_engine(), _make_turso_engine())


if __name__ == "__main__":
    main()

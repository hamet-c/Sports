"""Copy local SQLite -> Turso over the libsql HTTP protocol.

Fallback to migrate_to_turso.py: the sqlite+libsql SQLAlchemy dialect needs
either libsql-experimental (no Windows wheels) or libsql-client's dbapi2
whose hrana WebSocket handshake Turso now rejects with HTTP 400. Core
libsql_client over HTTPS works everywhere, including this Windows/3.14 venv.

Schema is cloned verbatim from local sqlite_master DDL; rows are copied as
INSERT OR IGNORE in server-side transactional batches, so reruns after a
partial failure are safe (PK collisions skip).

    python scripts/migrate_to_turso_http.py            # migrate + verify
    python scripts/migrate_to_turso_http.py --verify   # row-count check only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import libsql_client

from app.core.config import settings

CHUNK = 500

TABLE_ORDER = [
    "teams",
    "players",
    "games",
    "team_game_stats",
    "player_game_stats",
    "injury_reports",
    "prop_lines",
    "game_markets",
    "predictions",
    "recommendation_log",
]


def _local_conn() -> sqlite3.Connection:
    path = settings.project_root / "data" / "nba_props.db"
    if not path.exists():
        raise SystemExit(f"Local DB not found at {path}")
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _turso_client() -> libsql_client.ClientSync:
    if not settings.turso_database_url or not settings.turso_auth_token:
        raise SystemExit("TURSO_DATABASE_URL / TURSO_AUTH_TOKEN not set in .env")
    url = settings.turso_database_url.replace("libsql://", "https://", 1)
    return libsql_client.create_client_sync(url, auth_token=settings.turso_auth_token)


def create_schema(local: sqlite3.Connection, turso: libsql_client.ClientSync) -> None:
    remote_tables = {
        r[0] for r in turso.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).rows
    }
    ddl_rows = local.execute(
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
        "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END"
    ).fetchall()
    for row in ddl_rows:
        if row["type"] == "table" and row["name"] in remote_tables:
            continue
        sql = row["sql"]
        if row["type"] != "table":
            sql = sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
            sql = sql.replace("CREATE UNIQUE INDEX", "CREATE UNIQUE INDEX IF NOT EXISTS", 1)
        turso.execute(sql)
    print("Schema created / confirmed on Turso.")


def migrate(local: sqlite3.Connection, turso: libsql_client.ClientSync) -> None:
    create_schema(local, turso)
    for table in TABLE_ORDER:
        cols = [r["name"] for r in local.execute(f"PRAGMA table_info({table})").fetchall()]
        placeholders = ", ".join("?" for _ in cols)
        insert_sql = (
            f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
        )
        total = local.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if total == 0:
            print(f"{table:<25} (empty, skip)")
            continue
        print(f"{table:<25} migrating {total:,} rows...")
        start = time.time()
        copied = 0
        batch: list[libsql_client.Statement] = []
        for row in local.execute(f"SELECT * FROM {table}"):
            batch.append(libsql_client.Statement(insert_sql, tuple(row)))
            if len(batch) >= CHUNK:
                turso.batch(batch)
                copied += len(batch)
                print(f"  ... {copied:,}/{total:,}", end="\r")
                batch = []
        if batch:
            turso.batch(batch)
            copied += len(batch)
        elapsed = time.time() - start
        print(f"  {copied:,} rows in {elapsed:.1f}s ({copied / max(elapsed, 0.01):.0f}/s)")
    print("\nDone.")


def verify(local: sqlite3.Connection, turso: libsql_client.ClientSync) -> bool:
    print("\nRow-count verification:")
    print(f"{'table':<25} {'local':>10} {'turso':>10}  match")
    print("-" * 60)
    all_match = True
    for table in TABLE_ORDER:
        local_n = local.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        try:
            turso_n = turso.execute(f"SELECT COUNT(*) FROM {table}").rows[0][0]
        except Exception as exc:  # noqa: BLE001
            turso_n = f"ERR ({exc.__class__.__name__})"
        match = local_n == turso_n
        all_match = all_match and match
        print(f"{table:<25} {local_n:>10} {str(turso_n):>10}  {'OK' if match else 'MISMATCH'}")
    print("-" * 60)
    print("All tables match." if all_match else "Mismatches present.")
    return all_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Only run row-count verification")
    args = parser.parse_args()

    local = _local_conn()
    turso = _turso_client()
    try:
        if args.verify:
            ok = verify(local, turso)
        else:
            migrate(local, turso)
            ok = verify(local, turso)
        sys.exit(0 if ok else 1)
    finally:
        turso.close()
        local.close()


if __name__ == "__main__":
    main()

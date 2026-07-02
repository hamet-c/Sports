# NBA Player Props Prediction Engine

Predicts NBA player stat lines (points, rebounds, assists, threes-made) and surfaces +EV plays against sportsbook prop markets. Trained on four seasons of game logs plus advanced box scores; serves a daily slate through a FastAPI + React dashboard.

## Stack

Python 3.14 · FastAPI · SQLAlchemy · SQLite (local) + Turso/libSQL (CI ingest) · XGBoost · pandas · React 18 · Vite · TanStack Query · Tailwind · Recharts.

## Layout

```
backend/    FastAPI app, feature builders, models, scripts
frontend/   React + Vite dashboard
data/       SQLite DB, model artifacts, backtest reports
```

## Setup

```powershell
# Backend
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# One-time historical pull (~60-90 min)
.venv\Scripts\python.exe scripts\bootstrap_data.py --seasons 2022-23 2023-24 2024-25 2025-26

# Train models
.venv\Scripts\python.exe scripts\train_models.py --train-end 2026-02-15 --val-end 2026-04-13 --tune small

# Run API
uvicorn app.main:app --reload --port 8000
```

```powershell
# Frontend (separate terminal)
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Backend needs a `.env` (see `backend/.env.example`) with an `ODDS_API_KEY` from [the-odds-api.com](https://the-odds-api.com).

## Daily ingest

```powershell
cd backend
.venv\Scripts\python.exe scripts\ingest_props.py
.venv\Scripts\python.exe scripts\refresh_injuries.py
.venv\Scripts\python.exe scripts\log_recommendations.py   # freeze today's recs
.venv\Scripts\python.exe scripts\grade_recommendations.py # grade past recs
```

Automated via GitHub Actions (`.github/workflows/daily-ingest.yml`): a scheduled
job ingests props + injuries into **Turso** (hosted libSQL) daily during the
season using the `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` / `ODDS_API_KEY`
repo secrets. The DB file itself is not in git.

### Local vs Turso

The engine talks to the local `data/nba_props.db` unless `USE_TURSO=1` is set
(plus both Turso vars — see `backend/.env.example`). Keep local as default on
this machine: the `sqlite+libsql` dialect has no Windows / Python 3.14 wheels
(CI pins Python 3.12), and training/backtests should never run over the
network. `scripts/migrate_to_turso.py` copies local → Turso one-way
(idempotent, `--verify` for row counts); run it from a Python 3.12 venv.
Stats tables stay locally-mastered (bootstrap); odds/injury tables accumulate
on Turso via CI — sync down before the next training season.

## Backtests

```powershell
.venv\Scripts\python.exe scripts\run_backtest.py synthetic --start 2026-02-15 --end 2026-04-13 --save
```

Reports land in `data/reports/`. The Performance page in the dashboard surfaces them.

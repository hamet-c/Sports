# NBA Player Props Prediction Engine

Predicts NBA player stat lines (points, rebounds, assists, threes-made) and surfaces +EV plays against sportsbook prop markets. Trained on four seasons of game logs plus advanced box scores; serves a daily slate through a FastAPI + React dashboard.

## Stack

Python 3.14 · FastAPI · SQLAlchemy · SQLite · XGBoost · pandas · React 18 · Vite · TanStack Query · Tailwind · Recharts.

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
```

Automated via GitHub Actions — see `.github/workflows/daily-ingest.yml`. Pull latest before working locally so the DB is current.

## Backtests

```powershell
.venv\Scripts\python.exe scripts\run_backtest.py synthetic --start 2026-02-15 --end 2026-04-13 --save
```

Reports land in `data/reports/`. The Performance page in the dashboard surfaces them.

"""Centralized configuration via pydantic-settings."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"

    database_url: str = f"sqlite:///{(project_root / 'data' / 'nba_props.db').as_posix()}"

    odds_api_key: str = ""
    odds_api_base: str = "https://api.the-odds-api.com/v4"

    nba_api_request_timeout: int = 30
    nba_api_max_retries: int = 3
    nba_api_min_delay_seconds: float = 0.6

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    edge_threshold: float = 0.05
    # Asymmetric threshold for OVER recs only. When the model over-predicts
    # means (Phase 5.5 diagnostic: cal_off OVER win-rate 46.2% vs UNDER 53.3%
    # on real lines May 11-17), raising this lets the model demand more
    # evidence before recommending an OVER. Leave at edge_threshold for
    # symmetric behavior. Set via env: EDGE_THRESHOLD_OVER=0.10
    edge_threshold_over: float = 0.05
    min_predictions_for_calibration: int = 200
    # Phase 5 fit calibrators on per-row synthetic-line outcomes. Synthetic
    # lines (L10 rounded to .5) bias below the actual on counting stats, so the
    # calibrator learned to boost P(over). Against real sportsbook lines that
    # boost manufactures false OVER edges — diagnose_live_badge.py on the
    # 2026-05-11..05-17 window showed cal_on dragged OVER win-rate from 46.2%
    # (raw) to 32.7%, total win-rate 49.6% → 45.5%. Disabled until the
    # calibrator is refit against real PropLine outcomes.
    use_calibrators: bool = False

    log_level: str = "INFO"


settings = Settings()

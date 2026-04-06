"""
scripts/shared/db.py

TimescaleDB connection helper for research_strats.

Loads credentials from environment variables (or a .env file in the project root)
and returns a psycopg2 connection.

Connection resolution order:
  1. TIMESCALE_DSN  — full DSN string (preferred in Docker / CI environments)
  2. Individual PGHOST / PGDATABASE / PGUSER / PGPASSWORD / PGPORT vars

No Redis, IG, Airflow, or live-trading logic is present here.
"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Load .env if present (silent no-op when python-dotenv is not installed)
# ---------------------------------------------------------------------------
try:
    _project_root = Path(__file__).resolve().parents[2]
    _env_path = _project_root / ".env"
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Connection parameters
# ---------------------------------------------------------------------------
TIMESCALE_DSN = os.getenv("TIMESCALE_DSN")
PGHOST        = os.getenv("PGHOST",     "localhost")
PGDATABASE    = os.getenv("PGDATABASE", "market_data")
PGUSER        = os.getenv("PGUSER",     "backtesting")
PGPASSWORD    = os.getenv("PGPASSWORD", "backtesting_pass")
PGPORT        = int(os.getenv("PGPORT", "5434"))


def get_connection() -> psycopg2.extensions.connection:
    """
    Return a psycopg2 connection to TimescaleDB.

    Prefers TIMESCALE_DSN if set; falls back to individual PG* env vars.
    """
    if TIMESCALE_DSN:
        return psycopg2.connect(TIMESCALE_DSN)
    return psycopg2.connect(
        host=PGHOST,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        port=PGPORT,
    )

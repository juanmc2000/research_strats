"""
scripts/backfill_daily_market_metrics.py

Backfill features.daily_market_metrics for FX symbols.

Aggregates market_data.hourly_prices into daily bars and computes the
rolling 30-trading-day stddev of daily range (range_sd_30d) — the SD30d
risk unit used by the trailing-stop outcome labeling in issue #75.

Source : market_data.hourly_prices  (TimescaleDB, FX symbols only)
Target : features.daily_market_metrics (TimescaleDB)

Usage
-----
    python scripts/backfill_daily_market_metrics.py
    python scripts/backfill_daily_market_metrics.py --symbol "USD/JPY"
    python scripts/backfill_daily_market_metrics.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
try:
    _project_root = Path(__file__).resolve().parents[1]
    _env_path = _project_root / ".env"
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path)
except Exception:
    pass

TIMESCALE_DSN = os.getenv("TIMESCALE_DSN")
PGHOST        = os.getenv("PGHOST",     "localhost")
PGDATABASE    = os.getenv("PGDATABASE", "market_data")
PGUSER        = os.getenv("PGUSER",     "backtesting")
PGPASSWORD    = os.getenv("PGPASSWORD", "backtesting_pass")
PGPORT        = int(os.getenv("PGPORT", "5434"))


def get_connection() -> psycopg2.extensions.connection:
    if TIMESCALE_DSN:
        return psycopg2.connect(TIMESCALE_DSN)
    return psycopg2.connect(
        host=PGHOST, dbname=PGDATABASE,
        user=PGUSER, password=PGPASSWORD, port=PGPORT,
    )


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
_SQL = """
INSERT INTO features.daily_market_metrics (
    symbol,
    day_ts,
    daily_high,
    daily_low,
    daily_range,
    range_sd_30d
)

WITH

-- ---------------------------------------------------------------------------
-- Aggregate hourly bars to daily OHLC
-- Day boundary: UTC calendar date.
-- Session-close gaps produce no row for that calendar date.
-- ---------------------------------------------------------------------------
daily_ohlc AS (
    SELECT
        s.symbol,
        DATE(hp.date AT TIME ZONE 'UTC')            AS day_ts,
        MAX((hp.bid_high + hp.ask_high) / 2.0)      AS daily_high,
        MIN((hp.bid_low  + hp.ask_low)  / 2.0)      AS daily_low
    FROM market_data.hourly_prices hp
    JOIN market_data.symbols s ON s.id = hp.symbol_id
    WHERE {type_filter}
      AND hp.date < DATE_TRUNC('day', NOW() AT TIME ZONE 'UTC')
    {symbol_filter}
    GROUP BY s.symbol, DATE(hp.date AT TIME ZONE 'UTC')
),

with_range AS (
    SELECT
        symbol,
        day_ts,
        daily_high,
        daily_low,
        daily_high - daily_low AS daily_range
    FROM daily_ohlc
),

-- ---------------------------------------------------------------------------
-- Rolling 30-trading-day stddev of daily_range.
-- ROWS BETWEEN 29 PRECEDING AND CURRENT ROW = 30 trading days.
-- NULL for first 29 days (insufficient history).
-- ---------------------------------------------------------------------------
with_sd AS (
    SELECT
        symbol,
        day_ts,
        daily_high,
        daily_low,
        daily_range,
        STDDEV(daily_range) OVER (
            PARTITION BY symbol
            ORDER BY day_ts
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS range_sd_30d
    FROM with_range
)

SELECT
    symbol,
    day_ts,
    daily_high,
    daily_low,
    daily_range,
    range_sd_30d
FROM with_sd

ON CONFLICT (symbol, day_ts) DO UPDATE SET
    daily_high   = EXCLUDED.daily_high,
    daily_low    = EXCLUDED.daily_low,
    daily_range  = EXCLUDED.daily_range,
    range_sd_30d = EXCLUDED.range_sd_30d,
    updated_at   = NOW()
"""

_VALIDATION_SQL = """
SELECT
    symbol,
    COUNT(*)                                         AS total_days,
    COUNT(*) FILTER (WHERE range_sd_30d IS NOT NULL) AS days_with_sd30d,
    MIN(day_ts)                                      AS earliest,
    MAX(day_ts)                                      AS latest,
    ROUND(AVG(daily_range)::numeric, 6)              AS avg_daily_range,
    ROUND(AVG(range_sd_30d)::numeric, 6)             AS avg_range_sd_30d
FROM features.daily_market_metrics
{where_clause}
GROUP BY symbol
ORDER BY symbol;
"""


def _build_sql(symbol: Optional[str], non_fx: bool = False) -> str:
    if symbol is not None:
        if len(symbol) > 20 or not all(c.isalnum() or c in ("/_-. ") for c in symbol):
            raise ValueError(f"Symbol looks unsafe: {symbol!r}")
        symbol_filter = f"AND s.symbol = '{symbol}'"
    else:
        symbol_filter = ""
    type_filter = "s.type != 'forex'" if non_fx else "s.type = 'forex'"
    return _SQL.format(symbol_filter=symbol_filter, type_filter=type_filter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill features.daily_market_metrics."
    )
    parser.add_argument("--symbol",  default=None, help='Single symbol, e.g. "USD/JPY"')
    parser.add_argument("--non-fx",  action="store_true", help="Process non-FX instruments instead of FX")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL and exit")
    args = parser.parse_args()

    sql = _build_sql(args.symbol, non_fx=args.non_fx)

    if args.dry_run:
        print(sql)
        return

    scope = args.symbol or ("all non-FX instruments" if args.non_fx else "all FX symbols")
    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info("Building daily_market_metrics for %s", scope)

    conn = get_connection()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(sql)
            inserted = cur.rowcount
            conn.commit()
        log.info("Done: %d rows inserted/updated.", inserted)

        with conn.cursor() as cur:
            where = f"WHERE symbol = '{args.symbol}'" if args.symbol else ""
            cur.execute(_VALIDATION_SQL.format(where_clause=where))
            rows = cur.fetchall()
            log.info("Validation:")
            log.info(
                "  %-20s %10s %14s  %-12s  %-12s  %16s  %16s",
                "symbol", "days", "with_sd30d", "earliest", "latest",
                "avg_daily_range", "avg_range_sd30d",
            )
            for r in rows:
                log.info(
                    "  %-20s %10d %14d  %-12s  %-12s  %16s  %16s",
                    r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                )
    except Exception:
        conn.rollback()
        log.exception("Backfill failed — rolled back.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

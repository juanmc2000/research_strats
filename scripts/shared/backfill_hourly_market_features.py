"""
scripts/backfill_hourly_market_features.py

Historical backfill for features.hourly_market_features.

Reads market_data.hourly_prices (pre-aggregated H1 bars), computes all
derived features and inserts them into the feature table in a single
idempotent pass.

Usage
-----
    # Backfill full history for all symbols
    python backfill_hourly_market_features.py

    # Backfill a specific symbol
    python backfill_hourly_market_features.py --symbol "USD/JPY"

    # Backfill a date range
    python backfill_hourly_market_features.py --from 2023-01-01 --to 2024-01-01

Standards
---------
    - psycopg2 only (no SQLAlchemy, no pandas)
    - ON CONFLICT DO UPDATE: safe for reruns
    - All feature logic lives in SQL CTEs (see db/features/ for reference files)
    - Layer: Postgres / Background (no hot-path impact)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timezone
from pathlib import Path
from typing import Optional

import psycopg2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
# Config is loaded from environment. In Docker Compose the .env file is
# injected automatically; for standalone runs load it via python-dotenv.
try:
    _project_root = Path(__file__).resolve().parents[1]
    _env_path = _project_root / ".env"
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path)
except Exception:
    pass

# Feature pipeline runs against the same database that holds minute_prices.
# That is TimescaleDB, identified by TIMESCALE_DSN in .env.
# Individual PG* vars are kept as a fallback for environments where
# minute_prices and the feature table share the trading database.
TIMESCALE_DSN = os.getenv("TIMESCALE_DSN")

PGHOST     = os.getenv("PGHOST",     "localhost")
PGDATABASE = os.getenv("PGDATABASE", "trading")
PGUSER     = os.getenv("PGUSER",     "myuser")
PGPASSWORD = os.getenv("PGPASSWORD", "mypassword")
PGPORT     = int(os.getenv("PGPORT", "55432"))


def get_connection() -> psycopg2.extensions.connection:
    if TIMESCALE_DSN:
        return psycopg2.connect(TIMESCALE_DSN)
    return psycopg2.connect(
        host=PGHOST,
        dbname=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        port=PGPORT,
    )


# ---------------------------------------------------------------------------
# SQL: full feature pipeline as a single INSERT ... WITH ... SELECT
# ---------------------------------------------------------------------------
# The CTE chain mirrors the reference SQL files in db/features/:
#   hourly_ohlc -> reads market_data.hourly_prices (H1 bars already aggregated)
#   003 -> volatility and drift features
#   004 -> efficiency, flip count, shock features
#   005 -> 20-hour breakout features
#
# ON CONFLICT DO UPDATE ensures reruns are safe and rows are refreshed.

_BACKFILL_SQL = """
INSERT INTO features.hourly_market_features (
    symbol,
    hour_ts,
    open,
    high,
    low,
    close,
    return_1h,
    stddev_20,
    stddev_100,
    vol_ratio_20_100,
    range_20_sd_units,
    drift_20_sd_units,
    ma_20,
    distance_from_ma20_sd,
    efficiency_20,
    flip_count_10,
    flip_count_20,
    shock_1h_sd,
    shock_3h_max_sd,
    high_20,
    low_20,
    breakout_up_20h_flag,
    breakout_down_20h_flag,
    breakout_strength_sd
)

WITH

-- ===========================================================================
-- Hourly OHLC from market_data.hourly_prices
-- Midpoint = (bid + ask) / 2 applied per OHLC component independently.
-- Source table is already aggregated to H1 bars — no minute aggregation needed.
-- ===========================================================================

hourly_ohlc AS (
    SELECT
        s.symbol,
        hp.date                                      AS hour_ts,
        (hp.bid_open  + hp.ask_open)  / 2.0         AS open,
        (hp.bid_high  + hp.ask_high)  / 2.0         AS high,
        (hp.bid_low   + hp.ask_low)   / 2.0         AS low,
        (hp.bid_close + hp.ask_close) / 2.0         AS close
    FROM market_data.hourly_prices hp
    JOIN market_data.symbols s ON s.id = hp.symbol_id
    -- Exclude the current incomplete hour.
    WHERE hp.date < date_trunc('hour', NOW() AT TIME ZONE 'UTC')
    {symbol_filter}
    {date_range_filter}
),

-- ===========================================================================
-- Issue 3: Log returns + rolling volatility + drift features
-- return_1h = LN(close / prev_close)  [log return]
-- All SD-normalised features use: close * GREATEST(stddev_20, 1e-8)
-- Features capped at [-10, +10].
-- ===========================================================================

with_returns AS (
    SELECT
        symbol,
        hour_ts,
        open,
        high,
        low,
        close,
        -- Log return: CASE guard required because LN errors on non-positive args
        CASE
            WHEN LAG(close) OVER w > 0 AND close > 0
            THEN LN(close / LAG(close) OVER w)
            ELSE NULL
        END AS return_1h
    FROM hourly_ohlc
    WINDOW w AS (PARTITION BY symbol ORDER BY hour_ts)
),

rolling_stats AS (
    SELECT
        *,
        STDDEV(return_1h) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS stddev_20,
        STDDEV(return_1h) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
        ) AS stddev_100,
        AVG(close) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS ma_20,
        MAX(high) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS range_high_20,
        MIN(low) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS range_low_20,
        LAG(close, 20) OVER (
            PARTITION BY symbol ORDER BY hour_ts
        ) AS close_20_ago
    FROM with_returns
),

-- ===========================================================================
-- Issue 4: Efficiency, flip count, shock features
-- ===========================================================================

with_path AS (
    SELECT
        *,
        SUM(ABS(return_1h)) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS total_abs_path_20
    FROM rolling_stats
),

with_flip AS (
    SELECT
        *,
        CASE
            WHEN return_1h * LAG(return_1h) OVER (PARTITION BY symbol ORDER BY hour_ts) < 0
            THEN 1
            ELSE 0
        END AS is_flip
    FROM with_path
),

-- 3-bar max absolute return pre-computed to keep final SELECT clean
with_3h_shock AS (
    SELECT
        *,
        MAX(ABS(return_1h)) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS max_abs_return_3h
    FROM with_flip
),

-- ===========================================================================
-- Issue 5: 20-hour breakout levels (trailing, lookahead-free)
-- ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING excludes the current bar.
-- ===========================================================================

with_breakout_levels AS (
    SELECT
        *,
        MAX(high) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS high_20,
        MIN(low) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS low_20
    FROM with_3h_shock
)

-- ===========================================================================
-- Final SELECT: assemble all features
-- Denominator pattern : close * GREATEST(stddev_20, 1e-8)
-- Cap pattern         : LEAST(GREATEST(raw, -10.0), 10.0)
-- ===========================================================================

SELECT
    symbol,
    hour_ts,
    open,
    high,
    low,
    close,

    -- Issue 3: volatility and drift
    return_1h,
    stddev_20,
    stddev_100,
    CASE
        WHEN stddev_100 IS NOT NULL
        THEN stddev_20 / GREATEST(stddev_100, 1e-8)
        ELSE NULL
    END AS vol_ratio_20_100,
    CASE
        WHEN stddev_20 IS NOT NULL AND close > 0
        THEN LEAST(GREATEST(
            (range_high_20 - range_low_20) / (close * GREATEST(stddev_20, 1e-8)),
            -10.0
        ), 10.0)
        ELSE NULL
    END AS range_20_sd_units,
    CASE
        WHEN stddev_20 IS NOT NULL AND close > 0 AND close_20_ago IS NOT NULL
        THEN LEAST(GREATEST(
            (close - close_20_ago) / (close * GREATEST(stddev_20, 1e-8)),
            -10.0
        ), 10.0)
        ELSE NULL
    END AS drift_20_sd_units,
    ma_20,
    CASE
        WHEN stddev_20 IS NOT NULL AND close > 0
        THEN LEAST(GREATEST(
            (close - ma_20) / (close * GREATEST(stddev_20, 1e-8)),
            -10.0
        ), 10.0)
        ELSE NULL
    END AS distance_from_ma20_sd,

    -- Issue 4: efficiency (naturally bounded; no cap), flip counts, shock
    CASE
        WHEN total_abs_path_20 > 0 AND close_20_ago > 0 AND close > 0
        THEN ABS(LN(close / close_20_ago)) / total_abs_path_20
        ELSE NULL
    END AS efficiency_20,
    CAST(
        SUM(is_flip) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ) AS INTEGER
    ) AS flip_count_10,
    CAST(
        SUM(is_flip) OVER (
            PARTITION BY symbol ORDER BY hour_ts
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS INTEGER
    ) AS flip_count_20,
    CASE
        WHEN stddev_20 IS NOT NULL
        THEN LEAST(GREATEST(
            return_1h / GREATEST(stddev_20, 1e-8),
            -10.0
        ), 10.0)
        ELSE NULL
    END AS shock_1h_sd,
    CASE
        WHEN stddev_20 IS NOT NULL
        THEN LEAST(GREATEST(
            max_abs_return_3h / GREATEST(stddev_20, 1e-8),
            -10.0
        ), 10.0)
        ELSE NULL
    END AS shock_3h_max_sd,

    -- Issue 5: breakout
    high_20,
    low_20,
    (close > high_20)  AS breakout_up_20h_flag,
    (close < low_20)   AS breakout_down_20h_flag,
    CASE
        WHEN close > high_20 AND stddev_20 IS NOT NULL AND close > 0
            THEN LEAST(GREATEST(
                (close - high_20) / (close * GREATEST(stddev_20, 1e-8)),
                -10.0
            ), 10.0)
        WHEN close < low_20  AND stddev_20 IS NOT NULL AND close > 0
            THEN LEAST(GREATEST(
                (low_20 - close)  / (close * GREATEST(stddev_20, 1e-8)),
                -10.0
            ), 10.0)
        ELSE NULL
    END AS breakout_strength_sd

FROM with_breakout_levels

ON CONFLICT (symbol, hour_ts) DO UPDATE SET
    open                    = EXCLUDED.open,
    high                    = EXCLUDED.high,
    low                     = EXCLUDED.low,
    close                   = EXCLUDED.close,
    return_1h               = EXCLUDED.return_1h,
    stddev_20               = EXCLUDED.stddev_20,
    stddev_100              = EXCLUDED.stddev_100,
    vol_ratio_20_100        = EXCLUDED.vol_ratio_20_100,
    range_20_sd_units       = EXCLUDED.range_20_sd_units,
    drift_20_sd_units       = EXCLUDED.drift_20_sd_units,
    ma_20                   = EXCLUDED.ma_20,
    distance_from_ma20_sd   = EXCLUDED.distance_from_ma20_sd,
    efficiency_20           = EXCLUDED.efficiency_20,
    flip_count_10           = EXCLUDED.flip_count_10,
    flip_count_20           = EXCLUDED.flip_count_20,
    shock_1h_sd             = EXCLUDED.shock_1h_sd,
    shock_3h_max_sd         = EXCLUDED.shock_3h_max_sd,
    high_20                 = EXCLUDED.high_20,
    low_20                  = EXCLUDED.low_20,
    breakout_up_20h_flag    = EXCLUDED.breakout_up_20h_flag,
    breakout_down_20h_flag  = EXCLUDED.breakout_down_20h_flag,
    breakout_strength_sd    = EXCLUDED.breakout_strength_sd,
    updated_at              = NOW()
"""


def _build_sql(
    symbol: Optional[str],
    from_ts: Optional[str],
    to_ts: Optional[str],
) -> tuple[str, list]:
    """
    Substitute filters into the SQL template and return (sql, params).

    Filters are injected as literal WHERE / AND clauses (not as parameters)
    because they appear inside WITH blocks that reference different table
    aliases. The symbol name and date strings are validated before use.
    """
    params: list = []

    # Symbol filter — injected into the hourly_ohlc CTE WHERE clause.
    if symbol is not None:
        if len(symbol) > 20 or not all(c.isalnum() or c in ("/_-. ") for c in symbol):
            raise ValueError(f"Symbol looks unsafe: {symbol!r}")
        symbol_filter = f"AND s.symbol = '{symbol}'"
    else:
        symbol_filter = ""

    # Date-range filter on hp.date inside the hourly_ohlc CTE.
    # Accepts YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS.
    date_parts: list[str] = []
    if from_ts is not None:
        if len(from_ts) < 10 or from_ts[4] != "-" or from_ts[7] != "-":
            raise ValueError(f"from_ts must start YYYY-MM-DD, got: {from_ts!r}")
        date_parts.append(f"AND hp.date >= '{from_ts}'::timestamptz")
    if to_ts is not None:
        if len(to_ts) < 10 or to_ts[4] != "-" or to_ts[7] != "-":
            raise ValueError(f"to_ts must start YYYY-MM-DD, got: {to_ts!r}")
        date_parts.append(f"AND hp.date < '{to_ts}'::timestamptz")

    date_range_filter = "\n    ".join(date_parts)

    sql = _BACKFILL_SQL.format(
        symbol_filter=symbol_filter,
        date_range_filter=date_range_filter,
    )
    return sql, params


# ---------------------------------------------------------------------------
# Validation query
# ---------------------------------------------------------------------------
_VALIDATION_SQL = """
SELECT
    symbol,
    COUNT(*)                            AS row_count,
    MIN(hour_ts)                        AS earliest,
    MAX(hour_ts)                        AS latest,
    COUNT(*) FILTER (WHERE stddev_20    IS NOT NULL) AS has_stddev_20,
    COUNT(*) FILTER (WHERE high_20      IS NOT NULL) AS has_high_20,
    COUNT(*) FILTER (WHERE breakout_up_20h_flag)     AS breakout_up_rows,
    COUNT(*) FILTER (WHERE breakout_down_20h_flag)   AS breakout_down_rows
FROM features.hourly_market_features
{where_clause}
GROUP BY symbol
ORDER BY symbol;
"""


def _run_validation(cur: psycopg2.extensions.cursor, symbol: Optional[str]) -> None:
    where = f"WHERE symbol = '{symbol}'" if symbol else ""
    cur.execute(_VALIDATION_SQL.format(where_clause=where))
    rows = cur.fetchall()
    log.info("Validation results:")
    log.info(
        "  %-20s %10s  %-24s  %-24s  %12s  %10s  %12s  %14s",
        "symbol", "rows", "earliest", "latest",
        "has_stddev20", "has_high20", "breakout_up", "breakout_down",
    )
    for r in rows:
        log.info(
            "  %-20s %10d  %-24s  %-24s  %12d  %10d  %12d  %14d",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill features.hourly_market_features from minute_prices."
    )
    parser.add_argument(
        "--symbol",
        metavar="SYM",
        default=None,
        help='Limit to one symbol, e.g. "USD/JPY". Default: all symbols.',
    )
    parser.add_argument(
        "--from",
        dest="from_ts",
        metavar="YYYY-MM-DD",
        default=None,
        help="Start date (inclusive). Default: full history.",
    )
    parser.add_argument(
        "--to",
        dest="to_ts",
        metavar="YYYY-MM-DD",
        default=None,
        help="End date (exclusive). Default: full history.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print generated SQL and exit without executing.",
    )
    args = parser.parse_args()

    sql, _ = _build_sql(args.symbol, args.from_ts, args.to_ts)

    if args.dry_run:
        print(sql)
        return

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info(
        "Backfill parameters  symbol=%s  from=%s  to=%s",
        args.symbol or "(all)", args.from_ts or "(full)", args.to_ts or "(full)",
    )

    conn = get_connection()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            log.info("Running backfill INSERT ... WITH ... SELECT ...")
            cur.execute(sql)
            inserted = cur.rowcount
            conn.commit()
        log.info("Backfill complete: %d rows inserted/updated.", inserted)

        with conn.cursor() as cur:
            _run_validation(cur, args.symbol)

    except Exception:
        conn.rollback()
        log.exception("Backfill failed; transaction rolled back.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

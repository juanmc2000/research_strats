"""
scripts/incremental_hourly_market_features.py

Incremental update for features.hourly_market_features.

Instead of reprocessing full history (backfill), this script detects the
current high-water mark per symbol in the feature table and recomputes from
(max_hour_ts - BUFFER_HOURS) to the latest available minute_prices data.

Why a buffer window?
    Rolling features (stddev_20, stddev_100, drift, efficiency) depend on up
    to 100 prior bars.  Computing only the newest hour produces incorrect values
    because the window function has no preceding context.  The buffer ensures
    that all window-based features are computed with a complete lookback.

Buffer size
    BUFFER_HOURS = 100  (covers the longest window: stddev_100)
    Rows inside the buffer period are overwritten via ON CONFLICT DO UPDATE.
    Rows outside the buffer are untouched.

Usage
-----
    # Update all symbols (auto-detect watermark per symbol)
    python incremental_hourly_market_features.py

    # Update a single symbol
    python incremental_hourly_market_features.py --symbol "USD/JPY"

    # Override buffer size (hours)
    python incremental_hourly_market_features.py --buffer-hours 150

    # Dry-run: print per-symbol SQL without executing
    python incremental_hourly_market_features.py --dry-run

Standards
---------
    - Imports SQL builder from backfill_hourly_market_features (single source of truth)
    - psycopg2 only (no SQLAlchemy, no pandas)
    - ON CONFLICT DO UPDATE: safe for reruns
    - Layer: Postgres / Background (no hot-path impact)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Import shared SQL builder from backfill script (single source of truth)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from backfill_hourly_market_features import _build_sql, get_connection  # noqa: E402

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
# Constants
# ---------------------------------------------------------------------------
# 100 hours covers the longest rolling window (stddev_100).
# Increase if longer windows are added in future.
DEFAULT_BUFFER_HOURS: int = 100


# ---------------------------------------------------------------------------
# Watermark queries
# ---------------------------------------------------------------------------

_WATERMARK_SQL = """
SELECT
    s.symbol,
    MAX(f.hour_ts) AS max_hour_ts
FROM market_data.symbols s
LEFT JOIN features.hourly_market_features f ON f.symbol = s.symbol
{where_clause}
GROUP BY s.symbol
ORDER BY s.symbol;
"""

_MIN_AVAILABLE_SQL = """
SELECT
    s.symbol,
    MIN(date_trunc('hour', mp.date)) AS min_hour_ts,
    MAX(date_trunc('hour', mp.date)) AS max_hour_ts
FROM market_data.symbols s
JOIN market_data.minute_prices mp ON mp.symbol_id = s.id
{where_clause}
GROUP BY s.symbol
ORDER BY s.symbol;
"""


def get_watermarks(
    conn: psycopg2.extensions.connection,
    symbol: Optional[str],
) -> dict[str, Optional[datetime]]:
    """
    Return {symbol: max_hour_ts} from features.hourly_market_features.
    max_hour_ts is None when the symbol has no rows yet (first-time run).
    """
    where = f"WHERE s.symbol = '{symbol}'" if symbol else ""
    with conn.cursor() as cur:
        cur.execute(_WATERMARK_SQL.format(where_clause=where))
        return {row[0]: row[1] for row in cur.fetchall()}


def get_minute_price_bounds(
    conn: psycopg2.extensions.connection,
    symbol: Optional[str],
) -> dict[str, tuple[datetime, datetime]]:
    """
    Return {symbol: (min_hour_ts, max_hour_ts)} from minute_prices.
    Used to detect symbols with new data beyond the current watermark.
    """
    where = f"WHERE s.symbol = '{symbol}'" if symbol else ""
    with conn.cursor() as cur:
        cur.execute(_MIN_AVAILABLE_SQL.format(where_clause=where))
        return {row[0]: (row[1], row[2]) for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Per-symbol incremental run
# ---------------------------------------------------------------------------

def run_symbol(
    conn: psycopg2.extensions.connection,
    symbol: str,
    watermark: Optional[datetime],
    source_max: datetime,
    buffer_hours: int,
    dry_run: bool,
) -> int:
    """
    Recompute features for one symbol over [buffer_start, source_max].

    Returns the number of rows inserted/updated (0 for dry-run).
    """
    if watermark is None:
        # Symbol has no feature rows yet: full history
        from_ts: Optional[str] = None
        log.info(
            "  %s: no existing rows — full history backfill", symbol
        )
    else:
        buffer_start = watermark - timedelta(hours=buffer_hours)
        from_ts = buffer_start.strftime("%Y-%m-%dT%H:%M:%S")
        log.info(
            "  %s: watermark=%s  buffer_start=%s  source_max=%s",
            symbol,
            watermark.strftime("%Y-%m-%dT%H:%M:%S"),
            from_ts,
            source_max.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    sql, _ = _build_sql(symbol=symbol, from_ts=from_ts, to_ts=None)

    if dry_run:
        print(f"\n-- DRY RUN: {symbol} (from_ts={from_ts}) --")
        print(sql)
        return 0

    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            n = cur.rowcount
        conn.commit()
        return n
    except Exception:
        conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Incremental update for features.hourly_market_features. "
            "Recomputes the last BUFFER_HOURS hours per symbol to ensure "
            "rolling window features are correct."
        )
    )
    parser.add_argument(
        "--symbol",
        metavar="SYM",
        default=None,
        help='Limit to one symbol, e.g. "USD/JPY". Default: all symbols.',
    )
    parser.add_argument(
        "--buffer-hours",
        type=int,
        default=DEFAULT_BUFFER_HOURS,
        metavar="N",
        help=(
            f"Hours of history to recompute behind the watermark. "
            f"Must be >= longest rolling window (default: {DEFAULT_BUFFER_HOURS})."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print generated SQL per symbol and exit without executing.",
    )
    args = parser.parse_args()

    if args.buffer_hours < 1:
        parser.error("--buffer-hours must be >= 1")

    log.info(
        "Incremental update  symbol=%s  buffer_hours=%d  dry_run=%s",
        args.symbol or "(all)",
        args.buffer_hours,
        args.dry_run,
    )

    conn = get_connection()
    try:
        # ---- Determine which symbols need updating --------------------------
        watermarks = get_watermarks(conn, args.symbol)
        source_bounds = get_minute_price_bounds(conn, args.symbol)

        if not source_bounds:
            log.warning("No symbols found in minute_prices. Nothing to do.")
            return

        # Identify symbols with new source data beyond their watermark
        symbols_to_run: list[tuple[str, Optional[datetime], datetime]] = []
        for sym, (src_min, src_max) in source_bounds.items():
            watermark = watermarks.get(sym)
            if watermark is None or src_max > watermark:
                symbols_to_run.append((sym, watermark, src_max))
            else:
                log.info(
                    "  %s: up-to-date (watermark=%s  source_max=%s) — skipping",
                    sym,
                    watermark.strftime("%Y-%m-%dT%H:%M:%S") if watermark else "None",
                    src_max.strftime("%Y-%m-%dT%H:%M:%S"),
                )

        if not symbols_to_run:
            log.info("All symbols are up-to-date. Nothing to do.")
            return

        log.info(
            "%d symbol(s) to update: %s",
            len(symbols_to_run),
            ", ".join(s for s, _, _ in symbols_to_run),
        )

        # ---- Run per-symbol incremental update -----------------------------
        total_rows = 0
        errors: list[str] = []

        for sym, watermark, src_max in symbols_to_run:
            try:
                n = run_symbol(
                    conn=conn,
                    symbol=sym,
                    watermark=watermark,
                    source_max=src_max,
                    buffer_hours=args.buffer_hours,
                    dry_run=args.dry_run,
                )
                if not args.dry_run:
                    log.info("  %s: %d rows inserted/updated", sym, n)
                    total_rows += n
            except Exception as exc:
                log.error("  %s: FAILED — %s", sym, exc)
                errors.append(sym)
                # Continue to next symbol; partial success is better than full abort

        # ---- Summary --------------------------------------------------------
        if not args.dry_run:
            log.info(
                "Incremental update complete: %d total rows, %d symbol(s) failed.",
                total_rows,
                len(errors),
            )
            if errors:
                log.error("Failed symbols: %s", ", ".join(errors))
                sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    main()

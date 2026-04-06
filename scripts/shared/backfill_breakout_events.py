"""
scripts/backfill_breakout_events.py

Historical backfill for features.breakout_events — Issue #75.

Extracts FX breakout signals from features.hourly_market_features,
freezes the feature snapshot at signal time, attaches the SD30d risk
unit from features.daily_market_metrics, and labels trade outcomes
using a 2 SD30d trailing stop with FX-continuous holding.

Source : features.hourly_market_features   (TimescaleDB)
         features.daily_market_metrics      (TimescaleDB)
Target : features.breakout_events          (TimescaleDB)

Trade mechanics
---------------
- Entry       : close of the breakout bar
- Risk unit   : entry_range_sd_30d — rolling 30-day stddev of daily range,
                frozen at signal time, never updated
- Trailing stop (long):  price <= running_max - 2 × SD30d
- Trailing stop (short): price >= running_min + 2 × SD30d
- Running max/min initialised at entry_close
- Positions held across weekends (FX continuous — no session gaps)
- No fixed profit target
- Exit reason : trailing_stop | end_of_data

Outcome columns
---------------
  entry_range_sd_30d        : SD30d at entry (risk unit)
  max_favorable_excursion_sd30d : peak favourable move / SD30d (>= 0)
  max_adverse_excursion_sd30d   : peak adverse move   / SD30d (>= 0)
  realized_return_sd30d     : (exit_price - entry) / SD30d  (long)
                              (entry - exit_price) / SD30d  (short)
  exit_price                : mid-close at exit bar
  exit_ts                   : timestamp of exit bar
  exit_reason               : trailing_stop | end_of_data
  bars_held                 : number of bars from entry+1 to exit inclusive
  outcome_label             : mirrors exit_reason for convenience

Sanity check
------------
  For trailing_stop exits:
    realized_return_sd30d = max_favorable_excursion_sd30d - 2.0  (exact)

Usage
-----
    python scripts/backfill_breakout_events.py
    python scripts/backfill_breakout_events.py --symbol "USD/JPY"
    python scripts/backfill_breakout_events.py --from 2020-01-01 --to 2023-01-01
    python scripts/backfill_breakout_events.py --max-bars 1000
    python scripts/backfill_breakout_events.py --dry-run
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

DEFAULT_MAX_HOLDING_BARS: int = 500   # ~10 weeks; trailing stop exits well before this


def get_connection() -> psycopg2.extensions.connection:
    if TIMESCALE_DSN:
        return psycopg2.connect(TIMESCALE_DSN)
    return psycopg2.connect(
        host=PGHOST, dbname=PGDATABASE,
        user=PGUSER, password=PGPASSWORD, port=PGPORT,
    )


# ---------------------------------------------------------------------------
# SQL pipeline
# ---------------------------------------------------------------------------
# Step 1  fx_symbols       : FX symbols from market_data.symbols
# Step 2  events           : breakout signals with SD30d attached via LATERAL
# Step 3  forward_path     : future bars with running max/min and stop flag
# Step 4  exit_bar         : first stop-trigger bar (or last bar = end_of_data)
# Step 5  exit_values      : close price at exit bar
# Step 6  outcomes         : excursions and realised return assembled
# Step 7  INSERT           : upsert into features.breakout_events

_BACKFILL_SQL = """
INSERT INTO features.breakout_events (
    symbol,
    event_hour_ts,
    breakout_direction,
    entry_close,
    high_20,
    low_20,
    breakout_strength_sd,
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
    -- SD30d outcome columns (issue #75)
    entry_range_sd_30d,
    max_favorable_excursion_sd30d,
    max_adverse_excursion_sd30d,
    realized_return_sd30d,
    exit_price,
    exit_ts,
    exit_reason,
    bars_held,
    outcome_label
)

WITH

-- ===========================================================================
-- Step 1: Target symbols (FX or non-FX depending on run mode)
-- ===========================================================================
target_symbols AS (
    SELECT id, symbol
    FROM market_data.symbols
    WHERE {symbol_type_filter}
),

-- ===========================================================================
-- Step 2: Breakout events with SD30d risk unit
--
-- SD30d is the most recent completed daily value before the event bar,
-- fetched via LATERAL to guarantee it is frozen at execution time.
-- Events without a valid SD30d (first ~30 days of history) are excluded.
-- ===========================================================================
events AS (
    SELECT
        h.symbol,
        h.hour_ts                                               AS event_hour_ts,
        CASE WHEN h.breakout_up_20h_flag THEN 'UP' ELSE 'DOWN' END
                                                                AS breakout_direction,
        h.close                                                 AS entry_close,
        h.high_20,
        h.low_20,
        h.breakout_strength_sd,
        h.stddev_20,
        h.stddev_100,
        h.vol_ratio_20_100,
        h.range_20_sd_units,
        h.drift_20_sd_units,
        h.ma_20,
        h.distance_from_ma20_sd,
        h.efficiency_20,
        h.flip_count_10,
        h.flip_count_20,
        h.shock_1h_sd,
        h.shock_3h_max_sd,
        dm.range_sd_30d                                         AS entry_range_sd_30d
    FROM features.hourly_market_features h
    JOIN target_symbols fs ON fs.symbol = h.symbol
    -- Freeze SD30d at signal time: most recent completed day before event bar
    JOIN LATERAL (
        SELECT range_sd_30d
        FROM features.daily_market_metrics
        WHERE symbol  = h.symbol
          AND day_ts  < DATE(h.hour_ts AT TIME ZONE 'UTC')
          AND range_sd_30d IS NOT NULL
          AND range_sd_30d > 0
        ORDER BY day_ts DESC
        LIMIT 1
    ) dm ON TRUE
    WHERE (h.breakout_up_20h_flag = TRUE OR h.breakout_down_20h_flag = TRUE)
      AND h.stddev_20  IS NOT NULL
      AND h.close      >  0
      AND h.hour_ts    <  date_trunc('hour', NOW() AT TIME ZONE 'UTC')
    {event_filter}
),

-- ===========================================================================
-- Step 3: Forward path — future bars with running max/min and stop flag
--
-- Running max/min is initialised at GREATEST/LEAST(future_close, entry_close)
-- so that an immediate reversal of 2 × SD30d on bar+1 is correctly detected.
--
-- Trailing stop (long):  close <= GREATEST(running_max, entry_close) - 2 × SD30d
-- Trailing stop (short): close >= LEAST(running_min,   entry_close) + 2 × SD30d
--
-- The forward window is bounded to {max_holding_bars} bars.  Most trades
-- exit via trailing stop well before this limit.
-- ===========================================================================
forward_path AS (
    SELECT
        e.symbol,
        e.event_hour_ts,
        e.breakout_direction,
        e.entry_close,
        e.entry_range_sd_30d,
        f.hour_ts                                               AS future_ts,
        f.close                                                 AS future_close,
        -- Running max/min including entry_close as floor/ceiling
        GREATEST(
            MAX(f.close) OVER (
                PARTITION BY e.symbol, e.event_hour_ts
                ORDER BY f.hour_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ),
            e.entry_close
        )                                                       AS running_max,
        LEAST(
            MIN(f.close) OVER (
                PARTITION BY e.symbol, e.event_hour_ts
                ORDER BY f.hour_ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ),
            e.entry_close
        )                                                       AS running_min
    FROM events e
    JOIN features.hourly_market_features f
        ON  f.symbol   = e.symbol
        AND f.hour_ts  > e.event_hour_ts
        AND f.hour_ts <= e.event_hour_ts + ({max_holding_bars} * INTERVAL '1 hour')
),

-- Add stop-trigger flag (evaluated after running_max/min are computed)
with_stop_flag AS (
    SELECT
        *,
        CASE
            WHEN breakout_direction = 'UP'
                THEN future_close <= running_max - 2.0 * entry_range_sd_30d
            WHEN breakout_direction = 'DOWN'
                THEN future_close >= running_min + 2.0 * entry_range_sd_30d
            ELSE FALSE
        END                                                     AS stop_triggered
    FROM forward_path
),

-- ===========================================================================
-- Step 4: Exit bar
-- First bar where stop triggers; otherwise last available bar (end_of_data).
-- ===========================================================================
exit_bar AS (
    SELECT
        symbol,
        event_hour_ts,
        COALESCE(
            MIN(future_ts) FILTER (WHERE stop_triggered),
            MAX(future_ts)
        )                                                       AS exit_ts,
        (MIN(future_ts) FILTER (WHERE stop_triggered)) IS NOT NULL
                                                                AS stop_was_triggered,
        COUNT(*)                                                AS bars_held
    FROM with_stop_flag
    GROUP BY symbol, event_hour_ts
),

-- ===========================================================================
-- Step 5: Exit close and running max/min at exit bar
-- ===========================================================================
exit_values AS (
    SELECT
        sf.symbol,
        sf.event_hour_ts,
        sf.future_close                                         AS exit_close,
        sf.running_max                                          AS final_running_max,
        sf.running_min                                          AS final_running_min
    FROM with_stop_flag sf
    JOIN exit_bar eb
        ON  eb.symbol        = sf.symbol
        AND eb.event_hour_ts = sf.event_hour_ts
        AND eb.exit_ts       = sf.future_ts
),

-- ===========================================================================
-- Step 6: Excursions and realised return
--
-- Excursions computed over all bars up to and including the exit bar.
-- Favourable = best move in breakout direction  (>= 0)
-- Adverse    = worst move against direction     (>= 0)
--
-- Realised return for trailing_stop exits:
--   = max_favorable_excursion_sd30d - 2.0  (exact by construction)
-- ===========================================================================
outcomes AS (
    SELECT
        e.symbol,
        e.event_hour_ts,
        e.breakout_direction,
        e.entry_close,
        e.entry_range_sd_30d,
        eb.exit_ts,
        ev.exit_close,
        eb.stop_was_triggered,
        eb.bars_held,
        CASE WHEN eb.stop_was_triggered THEN 'trailing_stop' ELSE 'end_of_data' END
                                                                AS exit_reason,
        -- Realised return in SD30d units
        CASE
            WHEN e.breakout_direction = 'UP'
                THEN (ev.exit_close - e.entry_close) / e.entry_range_sd_30d
            ELSE
                (e.entry_close - ev.exit_close) / e.entry_range_sd_30d
        END                                                     AS realized_return_sd30d,
        -- Peak favourable excursion (running high-water mark at exit)
        GREATEST(
            CASE
                WHEN e.breakout_direction = 'UP'
                    THEN (ev.final_running_max - e.entry_close) / e.entry_range_sd_30d
                ELSE
                    (e.entry_close - ev.final_running_min) / e.entry_range_sd_30d
            END,
            0.0
        )                                                       AS max_favorable_excursion_sd30d,
        -- Peak adverse excursion (worst point against direction up to exit)
        GREATEST(
            MAX(
                CASE
                    WHEN e.breakout_direction = 'UP'
                        THEN (e.entry_close - sf.future_close) / e.entry_range_sd_30d
                    ELSE
                        (sf.future_close - e.entry_close) / e.entry_range_sd_30d
                END
            ),
            0.0
        )                                                       AS max_adverse_excursion_sd30d
    FROM events e
    JOIN exit_bar   eb ON eb.symbol = e.symbol AND eb.event_hour_ts = e.event_hour_ts
    JOIN exit_values ev ON ev.symbol = e.symbol AND ev.event_hour_ts = e.event_hour_ts
    -- All bars up to and including exit bar
    JOIN with_stop_flag sf
        ON  sf.symbol        = e.symbol
        AND sf.event_hour_ts = e.event_hour_ts
        AND sf.future_ts    <= eb.exit_ts
    GROUP BY
        e.symbol, e.event_hour_ts, e.breakout_direction, e.entry_close,
        e.entry_range_sd_30d, eb.exit_ts, ev.exit_close, eb.stop_was_triggered,
        eb.bars_held, ev.final_running_max, ev.final_running_min
)

-- ===========================================================================
-- Step 7: Final assembly
-- ===========================================================================
SELECT
    e.symbol,
    e.event_hour_ts,
    e.breakout_direction,
    e.entry_close,
    e.high_20,
    e.low_20,
    e.breakout_strength_sd,
    e.stddev_20,
    e.stddev_100,
    e.vol_ratio_20_100,
    e.range_20_sd_units,
    e.drift_20_sd_units,
    e.ma_20,
    e.distance_from_ma20_sd,
    e.efficiency_20,
    e.flip_count_10,
    e.flip_count_20,
    e.shock_1h_sd,
    e.shock_3h_max_sd,
    o.entry_range_sd_30d,
    o.max_favorable_excursion_sd30d,
    o.max_adverse_excursion_sd30d,
    o.realized_return_sd30d,
    o.exit_close                                                AS exit_price,
    o.exit_ts,
    o.exit_reason,
    o.bars_held,
    o.exit_reason                                               AS outcome_label

FROM events e
JOIN outcomes o
    ON  o.symbol        = e.symbol
    AND o.event_hour_ts = e.event_hour_ts

ON CONFLICT (symbol, event_hour_ts, breakout_direction) DO UPDATE SET
    entry_close                    = EXCLUDED.entry_close,
    high_20                        = EXCLUDED.high_20,
    low_20                         = EXCLUDED.low_20,
    breakout_strength_sd           = EXCLUDED.breakout_strength_sd,
    stddev_20                      = EXCLUDED.stddev_20,
    stddev_100                     = EXCLUDED.stddev_100,
    vol_ratio_20_100               = EXCLUDED.vol_ratio_20_100,
    range_20_sd_units              = EXCLUDED.range_20_sd_units,
    drift_20_sd_units              = EXCLUDED.drift_20_sd_units,
    ma_20                          = EXCLUDED.ma_20,
    distance_from_ma20_sd          = EXCLUDED.distance_from_ma20_sd,
    efficiency_20                  = EXCLUDED.efficiency_20,
    flip_count_10                  = EXCLUDED.flip_count_10,
    flip_count_20                  = EXCLUDED.flip_count_20,
    shock_1h_sd                    = EXCLUDED.shock_1h_sd,
    shock_3h_max_sd                = EXCLUDED.shock_3h_max_sd,
    entry_range_sd_30d             = EXCLUDED.entry_range_sd_30d,
    max_favorable_excursion_sd30d  = EXCLUDED.max_favorable_excursion_sd30d,
    max_adverse_excursion_sd30d    = EXCLUDED.max_adverse_excursion_sd30d,
    realized_return_sd30d          = EXCLUDED.realized_return_sd30d,
    exit_price                     = EXCLUDED.exit_price,
    exit_ts                        = EXCLUDED.exit_ts,
    exit_reason                    = EXCLUDED.exit_reason,
    bars_held                      = EXCLUDED.bars_held,
    outcome_label                  = EXCLUDED.outcome_label,
    updated_at                     = NOW()
"""


def _build_sql(
    symbol: Optional[str],
    from_ts: Optional[str],
    to_ts: Optional[str],
    max_holding_bars: int,
    non_fx: bool = False,
) -> str:
    filters: list[str] = []

    if symbol is not None:
        if len(symbol) > 20 or not all(c.isalnum() or c in ("/_-. ") for c in symbol):
            raise ValueError(f"Symbol looks unsafe: {symbol!r}")
        filters.append(f"AND h.symbol = '{symbol}'")

    if from_ts is not None:
        if len(from_ts) < 10 or from_ts[4] != "-" or from_ts[7] != "-":
            raise ValueError(f"from_ts must start YYYY-MM-DD, got: {from_ts!r}")
        filters.append(f"AND h.hour_ts >= '{from_ts}'::timestamptz")

    if to_ts is not None:
        if len(to_ts) < 10 or to_ts[4] != "-" or to_ts[7] != "-":
            raise ValueError(f"to_ts must start YYYY-MM-DD, got: {to_ts!r}")
        filters.append(f"AND h.hour_ts < '{to_ts}'::timestamptz")

    symbol_type_filter = "type != 'forex'" if non_fx else "type = 'forex'"

    return _BACKFILL_SQL.format(
        event_filter="\n    ".join(filters),
        max_holding_bars=int(max_holding_bars),
        symbol_type_filter=symbol_type_filter,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_VALIDATION_SQL = """
SELECT
    symbol,
    breakout_direction,
    COUNT(*)                                                        AS events,
    COUNT(*) FILTER (WHERE exit_reason = 'trailing_stop')          AS trailing_stop,
    COUNT(*) FILTER (WHERE exit_reason = 'end_of_data')            AS end_of_data,
    ROUND(AVG(realized_return_sd30d)::numeric,   3)                AS avg_return_sd30d,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY realized_return_sd30d)::numeric, 3)
                                                                    AS median_return,
    ROUND(
        COUNT(*) FILTER (WHERE realized_return_sd30d > 0) * 100.0 / COUNT(*),
        1
    )                                                               AS win_pct,
    ROUND(AVG(bars_held)::numeric, 1)                              AS avg_bars_held,
    ROUND(AVG(max_favorable_excursion_sd30d)::numeric, 3)          AS avg_fav_sd30d,
    ROUND(AVG(max_adverse_excursion_sd30d)::numeric,   3)          AS avg_adv_sd30d
FROM features.breakout_events
{where_clause}
GROUP BY symbol, breakout_direction
ORDER BY symbol, breakout_direction;
"""


def _run_validation(cur: psycopg2.extensions.cursor, symbol: Optional[str]) -> None:
    where = f"WHERE symbol = '{symbol}'" if symbol else ""
    cur.execute(_VALIDATION_SQL.format(where_clause=where))
    rows = cur.fetchall()
    log.info("Validation results:")
    log.info(
        "  %-20s %-5s %8s %13s %12s %14s %12s %8s %10s %12s %12s",
        "symbol", "dir", "events", "trail_stop", "end_of_data",
        "avg_ret_sd30d", "median_ret", "win%", "avg_bars",
        "avg_fav_sd30d", "avg_adv_sd30d",
    )
    for r in rows:
        log.info(
            "  %-20s %-5s %8d %13d %12d %14s %12s %8s %10s %12s %12s",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill features.breakout_events with 2 SD30d trailing stop labeling."
    )
    parser.add_argument("--symbol",   default=None, metavar="SYM")
    parser.add_argument("--from",     dest="from_ts", default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--to",       dest="to_ts",   default=None, metavar="YYYY-MM-DD")
    parser.add_argument("--max-bars", type=int, default=DEFAULT_MAX_HOLDING_BARS,
                        help=f"Max forward bars per event (default {DEFAULT_MAX_HOLDING_BARS})")
    parser.add_argument("--non-fx",   action="store_true",
                        help="Process non-FX instruments instead of FX symbols")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if args.max_bars < 1:
        parser.error("--max-bars must be >= 1")

    sql = _build_sql(
        symbol=args.symbol,
        from_ts=args.from_ts,
        to_ts=args.to_ts,
        max_holding_bars=args.max_bars,
        non_fx=args.non_fx,
    )

    if args.dry_run:
        print(sql)
        return

    scope = args.symbol or ("all non-FX instruments" if args.non_fx else "all FX symbols")
    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info(
        "Backfill parameters  symbol=%s  from=%s  to=%s  max_bars=%d",
        scope, args.from_ts or "(full)", args.to_ts or "(full)",
        args.max_bars,
    )

    conn = get_connection()
    try:
        conn.autocommit = False
        with conn.cursor() as cur:
            log.info("Running trailing-stop event extraction ...")
            cur.execute(sql)
            inserted = cur.rowcount
            conn.commit()
        log.info("Backfill complete: %d rows inserted/updated.", inserted)

        with conn.cursor() as cur:
            _run_validation(cur, args.symbol)

    except Exception:
        conn.rollback()
        log.exception("Backfill failed — rolled back.")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/ctu/markov_ctu_universe_qualification.py
Issue #4: CTU Universe Qualification and Sleeve Construction

Promotes CTU from validated signal candidate to a properly examined sleeve
candidate by applying the same disciplined process used for ETD.

This script does NOT introduce new filters or optimise thresholds.
It applies pre-declared qualification rules to classify each FX symbol as:
  KEEP    : frozen CTU is a valid and portable strategy object for this symbol
  MONITOR : mixed evidence or thin forward sample; more data required
  EXCLUDE : frozen CTU is structurally unsuitable for this symbol

Pre-declared qualification rules (written before results are inspected)
-----------------------------------------------------------------------
KEEP:
  - full_sharpe >= KEEP_FULL_SHARPE_MIN  (0.0)
  - fwd_sharpe  >= KEEP_FWD_SHARPE_MIN   (0.0)
  - fwd_n       >= MIN_FWD_TRADES        (10)

MONITOR:
  - fwd_n < MIN_FWD_TRADES
  OR
  - full_sharpe >= 0 but fwd_sharpe < 0  (positive full, fails forward)
  OR
  - full_sharpe < 0 but fwd_sharpe >= 0  (negative full, rescued by forward — unstable)
  OR
  - full and forward disagree: abs(full_sharpe - fwd_sharpe) > DISAGREE_THRESHOLD

EXCLUDE:
  - full_sharpe < 0  AND  fwd_sharpe < 0
  OR
  - weak profile: full_win_rate < WEAK_WIN_RATE  AND  full_median < WEAK_MEDIAN
  OR
  - left-tail degradation: fwd_p_lt_neg1 > universe_fwd_p_lt_neg1 * LEFT_TAIL_MULTIPLIER

Symbol families (fixed, pre-declared)
--------------------------------------
  usd_majors       : EUR/USD, GBP/USD, AUD/USD, NZD/USD
  commodity_usd    : USD/CAD
  european_cross   : EUR/GBP
  safe_haven_usd   : USD/CHF

Universe candidates (pre-declared)
------------------------------------
  A_all_7          : all 7 symbols
  B_exclude_only   : KEEP + MONITOR (drop EXCLUDE)
  C_keep_only      : KEEP symbols only
  D_keep_plus_monitor : KEEP + MONITOR (alias, explicit for ranking)

Universes ranked primarily by forward metrics:
  1. forward Sharpe
  2. forward max drawdown
  3. forward p(<-1)
  4. forward trade count

Scope
------
  FX only. TimescaleDB features schema only.
  No new entry filters. No sizing. No overlays. No parquet. No prod DB writes.

Usage
------
  python scripts/ctu/markov_ctu_universe_qualification.py
  python scripts/ctu/markov_ctu_universe_qualification.py --no-write
  python scripts/ctu/markov_ctu_universe_qualification.py \\
      --output results/ctu_universe_qualification.csv \\
      --memo   results/ctu_universe_qualification_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Universe ───────────────────────────────────────────────────────────────────

FX_SYMBOLS: list[str] = [
    'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
]

# ── Symbol families (fixed, pre-declared — not chosen after results) ───────────

SYMBOL_FAMILIES: dict[str, list[str]] = {
    'usd_majors':      ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD'],
    'commodity_usd':   ['USD/CAD'],
    'european_cross':  ['EUR/GBP'],
    'safe_haven_usd':  ['USD/CHF'],
}

# ── Frozen CTU signal definition ───────────────────────────────────────────────
# Do not modify these without a new issue and versioned research note.

VOL_CONTRACTING_MAX: float = 0.80
EFF_TRENDING_MIN:    float = 0.60
MEDIUM_COST:         float = 0.07
TRAIN_END_YEAR:      int   = 2022
VAL_END_YEAR:        int   = 2024

CONFIG_NAME: str = 'frozen_ctu'

# ── Pre-declared qualification thresholds ─────────────────────────────────────
# Written before results are inspected.

MIN_FWD_TRADES:       int   = 10     # minimum forward trades to call KEEP
KEEP_FULL_SHARPE_MIN: float = 0.0    # full-period Sharpe must be >= this for KEEP
KEEP_FWD_SHARPE_MIN:  float = 0.0    # forward Sharpe must be >= this for KEEP
DISAGREE_THRESHOLD:   float = 0.30   # |full_sharpe - fwd_sharpe| above this → MONITOR
WEAK_WIN_RATE:        float = 0.35   # win_rate below this is weak (combined with median)
WEAK_MEDIAN:          float = -0.50  # median below this is weak (combined with win_rate)
LEFT_TAIL_MULTIPLIER: float = 2.0    # fwd p(<-1) > universe_baseline * this → left-tail flag

# ── Splits ─────────────────────────────────────────────────────────────────────

SPLITS: list[str] = ['full', 'train', 'val', 'fwd']


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_root / '.env')
    dsn = os.environ.get('TIMESCALE_DSN')
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.environ.get('PGHOST', 'localhost'),
        dbname=os.environ.get('PGDATABASE', 'market_data'),
        user=os.environ.get('PGUSER', 'backtesting'),
        password=os.environ.get('PGPASSWORD', 'backtesting_pass'),
        port=int(os.environ.get('PGPORT', '5434')),
    )


# ── DDL (idempotent) ───────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS features.ctu_symbol_qualification (
    symbol                  VARCHAR(20)       NOT NULL,
    config_name             VARCHAR(40)       NOT NULL,
    status                  VARCHAR(10)       NOT NULL,
    full_n                  INTEGER,
    train_n                 INTEGER,
    val_n                   INTEGER,
    fwd_n                   INTEGER,
    raw_full_sharpe         DOUBLE PRECISION,
    raw_fwd_sharpe          DOUBLE PRECISION,
    filtered_full_sharpe    DOUBLE PRECISION,
    filtered_fwd_sharpe     DOUBLE PRECISION,
    full_mean               DOUBLE PRECISION,
    fwd_mean                DOUBLE PRECISION,
    full_max_dd             DOUBLE PRECISION,
    fwd_max_dd              DOUBLE PRECISION,
    full_win_rate           DOUBLE PRECISION,
    fwd_win_rate            DOUBLE PRECISION,
    full_p_lt_neg1          DOUBLE PRECISION,
    fwd_p_lt_neg1           DOUBLE PRECISION,
    ruling_reason           TEXT,
    created_at              TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_symbol_qualification PRIMARY KEY (symbol, config_name)
);

CREATE TABLE IF NOT EXISTS features.ctu_symbol_cluster_analysis (
    cluster_name    VARCHAR(30)       NOT NULL,
    symbol          VARCHAR(20)       NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    p_lt_neg1       DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_symbol_cluster_analysis
        PRIMARY KEY (cluster_name, symbol, split_name)
);

CREATE TABLE IF NOT EXISTS features.ctu_universe_comparison (
    universe_name   VARCHAR(40)       NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    p_lt_neg1       DOUBLE PRECISION,
    p_gt_pos1       DOUBLE PRECISION,
    cum_pnl         DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_universe_comparison
        PRIMARY KEY (universe_name, split_name)
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ctu_trades(symbol: str) -> list[dict]:
    """Load all frozen CTU events for a symbol from TimescaleDB."""
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                be.symbol,
                be.event_hour_ts,
                be.realized_return_sd30d,
                be.vol_ratio_20_100,
                be.efficiency_20,
                be.max_adverse_excursion_sd30d,
                be.max_favorable_excursion_sd30d
            FROM features.breakout_events be
            WHERE be.symbol = %s
              AND be.exit_reason IS NOT NULL
              AND be.entry_range_sd_30d > 0
              AND be.vol_ratio_20_100 < %s
              AND be.efficiency_20 > %s
              AND be.breakout_direction = 'UP'
              AND be.realized_return_sd30d IS NOT NULL
            ORDER BY be.event_hour_ts
        """, (symbol, VOL_CONTRACTING_MAX, EFF_TRENDING_MIN))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    trades = []
    for r in rows:
        ts = r['event_hour_ts']
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
        trades.append(dict(
            symbol=symbol,
            event_hour_ts=ts,
            excess=float(r['realized_return_sd30d']) - MEDIUM_COST,
            year=year,
        ))
    return trades


# ── Split helpers ──────────────────────────────────────────────────────────────

def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> Optional[dict]:
    vals = [tr['excess'] for tr in trades]
    n = len(vals)
    if n < 2:
        return None
    mn = sum(vals) / n
    sd = statistics.stdev(vals)
    sharpe = mn / sd if sd > 0 else 0.0
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0.0 else 0.0
    sv = sorted(vals)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        cum += v
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    p_lt_neg1 = sum(1 for v in vals if v < -1.0) / n
    p_gt_pos1 = sum(1 for v in vals if v > 1.0) / n
    return dict(
        n=n, mean=mn, median=med, std=sd, sharpe=sharpe,
        win_rate=win_rate, payoff=payoff, max_dd=max_dd,
        p_lt_neg1=p_lt_neg1, p_gt_pos1=p_gt_pos1, cum_pnl=cum,
    )


# ── Qualification rules ────────────────────────────────────────────────────────

def qualify_symbol(
    full_stats: Optional[dict],
    fwd_stats: Optional[dict],
    universe_fwd_p_lt_neg1: float,
) -> tuple[str, str]:
    """
    Apply pre-declared qualification rules. Returns (status, reason).
    Rules are written before results are inspected.
    """
    full_sh  = full_stats['sharpe']   if full_stats else None
    fwd_sh   = fwd_stats['sharpe']    if fwd_stats  else None
    fwd_n    = fwd_stats['n']         if fwd_stats  else 0
    full_wr  = full_stats['win_rate'] if full_stats else None
    full_med = full_stats['median']   if full_stats else None
    fwd_ptl  = fwd_stats['p_lt_neg1'] if fwd_stats  else None

    # ── EXCLUDE conditions ─────────────────────────────────────────────────────
    if full_sh is not None and fwd_sh is not None and full_sh < 0.0 and fwd_sh < 0.0:
        return ('EXCLUDE',
                f'both full Sharpe ({full_sh:.3f}) and fwd Sharpe ({fwd_sh:.3f}) negative')

    if (full_wr is not None and full_med is not None
            and full_wr < WEAK_WIN_RATE and full_med < WEAK_MEDIAN):
        return ('EXCLUDE',
                f'weak profile: win_rate={full_wr:.3f} < {WEAK_WIN_RATE},'
                f' median={full_med:.3f} < {WEAK_MEDIAN}')

    if (fwd_ptl is not None and universe_fwd_p_lt_neg1 > 0.0
            and fwd_ptl > universe_fwd_p_lt_neg1 * LEFT_TAIL_MULTIPLIER):
        return ('EXCLUDE',
                f'left-tail degradation: fwd_p_lt_neg1={fwd_ptl:.3f}'
                f' > {LEFT_TAIL_MULTIPLIER}x universe baseline ({universe_fwd_p_lt_neg1:.3f})')

    # ── MONITOR conditions ─────────────────────────────────────────────────────
    if fwd_n < MIN_FWD_TRADES:
        return ('MONITOR',
                f'forward sample too thin: n={fwd_n} < {MIN_FWD_TRADES}')

    if (full_sh is not None and fwd_sh is not None
            and full_sh >= KEEP_FULL_SHARPE_MIN and fwd_sh < KEEP_FWD_SHARPE_MIN):
        return ('MONITOR',
                f'full Sharpe positive ({full_sh:.3f}) but fwd Sharpe negative ({fwd_sh:.3f})'
                f' — regime-dependent')

    if (full_sh is not None and fwd_sh is not None
            and full_sh < KEEP_FULL_SHARPE_MIN and fwd_sh >= KEEP_FWD_SHARPE_MIN):
        return ('MONITOR',
                f'full Sharpe negative ({full_sh:.3f}) but fwd Sharpe positive ({fwd_sh:.3f})'
                f' — forward-only rescue, unstable')

    if (full_sh is not None and fwd_sh is not None
            and abs(full_sh - fwd_sh) > DISAGREE_THRESHOLD):
        return ('MONITOR',
                f'full/fwd Sharpe disagree materially:'
                f' full={full_sh:.3f}, fwd={fwd_sh:.3f}'
                f' (diff={abs(full_sh - fwd_sh):.3f} > {DISAGREE_THRESHOLD})')

    # ── KEEP conditions ────────────────────────────────────────────────────────
    if (full_sh is not None and fwd_sh is not None
            and full_sh >= KEEP_FULL_SHARPE_MIN
            and fwd_sh >= KEEP_FWD_SHARPE_MIN
            and fwd_n >= MIN_FWD_TRADES):
        return ('KEEP',
                f'full Sharpe {full_sh:.3f}, fwd Sharpe {fwd_sh:.3f}, n_fwd={fwd_n}')

    return ('MONITOR', 'mixed evidence — requires further monitoring')


# ── TimescaleDB upserts ────────────────────────────────────────────────────────

def upsert_qualification(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.ctu_symbol_qualification (
                    symbol, config_name, status,
                    full_n, train_n, val_n, fwd_n,
                    raw_full_sharpe, raw_fwd_sharpe,
                    filtered_full_sharpe, filtered_fwd_sharpe,
                    full_mean, fwd_mean, full_max_dd, fwd_max_dd,
                    full_win_rate, fwd_win_rate,
                    full_p_lt_neg1, fwd_p_lt_neg1,
                    ruling_reason, updated_at
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, now()
                )
                ON CONFLICT (symbol, config_name) DO UPDATE SET
                    status               = EXCLUDED.status,
                    full_n               = EXCLUDED.full_n,
                    train_n              = EXCLUDED.train_n,
                    val_n                = EXCLUDED.val_n,
                    fwd_n                = EXCLUDED.fwd_n,
                    raw_full_sharpe      = EXCLUDED.raw_full_sharpe,
                    raw_fwd_sharpe       = EXCLUDED.raw_fwd_sharpe,
                    filtered_full_sharpe = EXCLUDED.filtered_full_sharpe,
                    filtered_fwd_sharpe  = EXCLUDED.filtered_fwd_sharpe,
                    full_mean            = EXCLUDED.full_mean,
                    fwd_mean             = EXCLUDED.fwd_mean,
                    full_max_dd          = EXCLUDED.full_max_dd,
                    fwd_max_dd           = EXCLUDED.fwd_max_dd,
                    full_win_rate        = EXCLUDED.full_win_rate,
                    fwd_win_rate         = EXCLUDED.fwd_win_rate,
                    full_p_lt_neg1       = EXCLUDED.full_p_lt_neg1,
                    fwd_p_lt_neg1        = EXCLUDED.fwd_p_lt_neg1,
                    ruling_reason        = EXCLUDED.ruling_reason,
                    updated_at           = now()
            """, (
                r['symbol'], r['config_name'], r['status'],
                r.get('full_n'), r.get('train_n'), r.get('val_n'), r.get('fwd_n'),
                r.get('raw_full_sharpe'), r.get('raw_fwd_sharpe'),
                r.get('filtered_full_sharpe'), r.get('filtered_fwd_sharpe'),
                r.get('full_mean'), r.get('fwd_mean'),
                r.get('full_max_dd'), r.get('fwd_max_dd'),
                r.get('full_win_rate'), r.get('fwd_win_rate'),
                r.get('full_p_lt_neg1'), r.get('fwd_p_lt_neg1'),
                r.get('ruling_reason'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def upsert_cluster(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.ctu_symbol_cluster_analysis (
                    cluster_name, symbol, split_name,
                    n_trades, sharpe, max_drawdown, win_rate, p_lt_neg1,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (cluster_name, symbol, split_name) DO UPDATE SET
                    n_trades     = EXCLUDED.n_trades,
                    sharpe       = EXCLUDED.sharpe,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate     = EXCLUDED.win_rate,
                    p_lt_neg1    = EXCLUDED.p_lt_neg1,
                    updated_at   = now()
            """, (
                r['cluster_name'], r['symbol'], r['split_name'],
                r.get('n_trades'), r.get('sharpe'), r.get('max_drawdown'),
                r.get('win_rate'), r.get('p_lt_neg1'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def upsert_universe_comparison(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.ctu_universe_comparison (
                    universe_name, split_name,
                    n_trades, sharpe, max_drawdown, win_rate,
                    p_lt_neg1, p_gt_pos1, cum_pnl,
                    updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (universe_name, split_name) DO UPDATE SET
                    n_trades     = EXCLUDED.n_trades,
                    sharpe       = EXCLUDED.sharpe,
                    max_drawdown = EXCLUDED.max_drawdown,
                    win_rate     = EXCLUDED.win_rate,
                    p_lt_neg1    = EXCLUDED.p_lt_neg1,
                    p_gt_pos1    = EXCLUDED.p_gt_pos1,
                    cum_pnl      = EXCLUDED.cum_pnl,
                    updated_at   = now()
            """, (
                r['universe_name'], r['split_name'],
                r.get('n_trades'), r.get('sharpe'), r.get('max_drawdown'),
                r.get('win_rate'), r.get('p_lt_neg1'), r.get('p_gt_pos1'),
                r.get('cum_pnl'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(val: Optional[float], d: int = 3) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{d}f}'


# ── Sleeve construction ────────────────────────────────────────────────────────

def sleeve_comparison(
    anchor: str,
    candidates: list[str],
    all_trades: dict[str, list[dict]],
) -> list[dict]:
    """
    Compare anchor-only vs anchor+one-symbol additions.
    Returns list of dicts with Sharpe delta, drawdown relief, etc.
    """
    results = []
    anchor_trades = split_trades(all_trades[anchor])

    for sp in ['full', 'fwd']:
        base = compute_stats(anchor_trades[sp])
        results.append(dict(
            comparison='anchor_only',
            added_symbol=anchor,
            split=sp,
            n=base['n'] if base else None,
            sharpe=base['sharpe'] if base else None,
            max_dd=base['max_dd'] if base else None,
            win_rate=base['win_rate'] if base else None,
            p_lt_neg1=base['p_lt_neg1'] if base else None,
            sharpe_delta=0.0,
            dd_relief=0.0,
        ))

    for cand in candidates:
        if cand == anchor:
            continue
        pooled_trades = all_trades[anchor] + all_trades[cand]
        splits = split_trades(pooled_trades)
        for sp in ['full', 'fwd']:
            base = compute_stats(split_trades(all_trades[anchor])[sp])
            combined = compute_stats(splits[sp])
            if combined is None:
                continue
            base_sh = base['sharpe'] if base else 0.0
            base_dd = base['max_dd'] if base else 0.0
            results.append(dict(
                comparison=f'anchor+{cand}',
                added_symbol=cand,
                split=sp,
                n=combined['n'],
                sharpe=combined['sharpe'],
                max_dd=combined['max_dd'],
                win_rate=combined['win_rate'],
                p_lt_neg1=combined['p_lt_neg1'],
                sharpe_delta=combined['sharpe'] - base_sh,
                dd_relief=base_dd - combined['max_dd'],
            ))
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',   default='results/ctu_universe_qualification.csv')
    parser.add_argument('--memo',     default='results/ctu_universe_qualification_memo.txt')
    parser.add_argument('--no-write', action='store_true',
                        help='Print only — do not write to TimescaleDB')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.memo) or '.', exist_ok=True)

    # ── Create tables ──────────────────────────────────────────────────────────
    if not args.no_write:
        conn = connect()
        create_tables(conn)
        conn.close()
        print('Tables created/verified.')

    # ── Load trades ────────────────────────────────────────────────────────────
    print(f'\nLoading frozen CTU trades for {len(FX_SYMBOLS)} FX symbols...')
    all_trades: dict[str, list[dict]] = {}
    for symbol in FX_SYMBOLS:
        trades = load_ctu_trades(symbol)
        all_trades[symbol] = trades
        splits = split_trades(trades)
        print(f'  {symbol:<10}  full={len(trades):4d}  '
              f'train={len(splits["train"]):4d}  '
              f'val={len(splits["val"]):4d}  '
              f'fwd={len(splits["fwd"]):4d}')

    # ── Compute per-symbol stats ───────────────────────────────────────────────
    sym_stats: dict[str, dict[str, Optional[dict]]] = {}
    for symbol in FX_SYMBOLS:
        sp = split_trades(all_trades[symbol])
        sym_stats[symbol] = {s: compute_stats(sp[s]) for s in SPLITS}

    # ── Universe baseline for left-tail rule ──────────────────────────────────
    all_pooled = [tr for trades in all_trades.values() for tr in trades]
    all_fwd = [tr for tr in all_pooled if tr['year'] >= VAL_END_YEAR]
    universe_fwd_stats = compute_stats(all_fwd)
    universe_fwd_p_lt_neg1 = universe_fwd_stats['p_lt_neg1'] if universe_fwd_stats else 0.0

    # ── Apply qualification rules ──────────────────────────────────────────────
    print(f'\nApplying qualification rules...')
    print(f'  (universe fwd p_lt_neg1 baseline: {universe_fwd_p_lt_neg1:.3f})')
    qual_status: dict[str, str] = {}
    qual_reason: dict[str, str] = {}
    db_qual_rows: list[dict] = []

    for symbol in FX_SYMBOLS:
        full_s = sym_stats[symbol]['full']
        fwd_s  = sym_stats[symbol]['fwd']
        status, reason = qualify_symbol(full_s, fwd_s, universe_fwd_p_lt_neg1)
        qual_status[symbol] = status
        qual_reason[symbol] = reason
        print(f'  {symbol:<10}  {status:<8}  {reason}')

        db_qual_rows.append(dict(
            symbol=symbol,
            config_name=CONFIG_NAME,
            status=status,
            full_n=full_s['n']         if full_s else None,
            train_n=sym_stats[symbol]['train']['n'] if sym_stats[symbol]['train'] else None,
            val_n=sym_stats[symbol]['val']['n']     if sym_stats[symbol]['val']   else None,
            fwd_n=fwd_s['n']           if fwd_s  else None,
            raw_full_sharpe=full_s['sharpe']    if full_s else None,
            raw_fwd_sharpe=fwd_s['sharpe']      if fwd_s  else None,
            filtered_full_sharpe=full_s['sharpe'] if full_s else None,  # no filter variant for CTU
            filtered_fwd_sharpe=fwd_s['sharpe']   if fwd_s  else None,
            full_mean=full_s['mean']    if full_s else None,
            fwd_mean=fwd_s['mean']      if fwd_s  else None,
            full_max_dd=full_s['max_dd']  if full_s else None,
            fwd_max_dd=fwd_s['max_dd']    if fwd_s  else None,
            full_win_rate=full_s['win_rate'] if full_s else None,
            fwd_win_rate=fwd_s['win_rate']   if fwd_s  else None,
            full_p_lt_neg1=full_s['p_lt_neg1'] if full_s else None,
            fwd_p_lt_neg1=fwd_s['p_lt_neg1']   if fwd_s  else None,
            ruling_reason=reason,
        ))

    if not args.no_write:
        n = upsert_qualification(db_qual_rows)
        print(f'  {n} rows written to features.ctu_symbol_qualification')

    # ── Cluster analysis ───────────────────────────────────────────────────────
    print('\nComputing cluster analysis...')
    db_cluster_rows: list[dict] = []
    for cluster_name, cluster_symbols in SYMBOL_FAMILIES.items():
        for symbol in cluster_symbols:
            for sp_name in SPLITS:
                s = sym_stats[symbol].get(sp_name)
                db_cluster_rows.append(dict(
                    cluster_name=cluster_name,
                    symbol=symbol,
                    split_name=sp_name,
                    n_trades=s['n']         if s else None,
                    sharpe=s['sharpe']      if s else None,
                    max_drawdown=s['max_dd'] if s else None,
                    win_rate=s['win_rate']  if s else None,
                    p_lt_neg1=s['p_lt_neg1'] if s else None,
                ))

    if not args.no_write:
        n = upsert_cluster(db_cluster_rows)
        print(f'  {n} rows written to features.ctu_symbol_cluster_analysis')

    # ── Build pre-declared universes ───────────────────────────────────────────
    keep_syms    = [s for s in FX_SYMBOLS if qual_status[s] == 'KEEP']
    monitor_syms = [s for s in FX_SYMBOLS if qual_status[s] == 'MONITOR']
    exclude_syms = [s for s in FX_SYMBOLS if qual_status[s] == 'EXCLUDE']

    universes: dict[str, list[str]] = {
        'A_all_7':           FX_SYMBOLS,
        'B_exclude_only':    [s for s in FX_SYMBOLS if qual_status[s] != 'EXCLUDE'],
        'C_keep_only':       keep_syms,
        'D_keep_plus_monitor': keep_syms + monitor_syms,
    }

    # ── Universe pooled stats and comparison ───────────────────────────────────
    print('\nComputing universe comparison...')
    universe_stats: dict[str, dict[str, Optional[dict]]] = {}
    db_universe_rows: list[dict] = []

    for uname, usymbols in universes.items():
        if not usymbols:
            print(f'  {uname:<25}  (empty — skipping)')
            universe_stats[uname] = {sp: None for sp in SPLITS}
            continue
        pooled = [tr for sym in usymbols for tr in all_trades[sym]]
        sp = split_trades(pooled)
        universe_stats[uname] = {s: compute_stats(sp[s]) for s in SPLITS}
        fwd_s = universe_stats[uname]['fwd']
        print(f'  {uname:<25}  symbols={len(usymbols)}  '
              f'fwd_n={fwd_s["n"] if fwd_s else 0:4d}  '
              f'fwd_sh={fmt(fwd_s["sharpe"] if fwd_s else None)}  '
              f'fwd_dd={fmt(fwd_s["max_dd"] if fwd_s else None)}')

        for sp_name in SPLITS:
            s = universe_stats[uname][sp_name]
            db_universe_rows.append(dict(
                universe_name=uname,
                split_name=sp_name,
                n_trades=s['n']       if s else None,
                sharpe=s['sharpe']    if s else None,
                max_drawdown=s['max_dd'] if s else None,
                win_rate=s['win_rate'] if s else None,
                p_lt_neg1=s['p_lt_neg1'] if s else None,
                p_gt_pos1=s['p_gt_pos1'] if s else None,
                cum_pnl=s['cum_pnl']  if s else None,
            ))

    if not args.no_write:
        n = upsert_universe_comparison(db_universe_rows)
        print(f'  {n} rows written to features.ctu_universe_comparison')

    # ── Forward-ranked universe table ──────────────────────────────────────────
    ranked: list[tuple[str, Optional[dict]]] = []
    for uname in universes:
        fwd_s = universe_stats[uname].get('fwd')
        ranked.append((uname, fwd_s))
    ranked.sort(
        key=lambda x: (
            -(x[1]['sharpe']    if x[1] and x[1]['sharpe']    is not None else -999),
             (x[1]['max_dd']    if x[1] and x[1]['max_dd']    is not None else  999),
             (x[1]['p_lt_neg1'] if x[1] and x[1]['p_lt_neg1'] is not None else  999),
            -(x[1]['n']         if x[1] and x[1]['n']         is not None else    0),
        )
    )

    # ── Sleeve construction ────────────────────────────────────────────────────
    print('\nSleeve construction...')
    anchor = None
    if keep_syms:
        # Anchor = KEEP symbol with best forward Sharpe
        anchor = max(
            keep_syms,
            key=lambda s: (sym_stats[s]['fwd']['sharpe']
                           if sym_stats[s]['fwd'] else -999)
        )
    elif monitor_syms:
        anchor = max(
            monitor_syms,
            key=lambda s: (sym_stats[s]['fwd']['sharpe']
                           if sym_stats[s]['fwd'] else -999)
        )

    sleeve_rows: list[dict] = []
    if anchor:
        sleeve_candidates = [s for s in (keep_syms + monitor_syms) if s != anchor]
        sleeve_rows = sleeve_comparison(anchor, sleeve_candidates, all_trades)
        print(f'  Anchor: {anchor}')
        for r in sleeve_rows:
            if r['split'] == 'fwd':
                print(f'    [{r["split"]}] {r["comparison"]:<25}  '
                      f'sharpe={fmt(r["sharpe"])}  '
                      f'sharpe_delta={r["sharpe_delta"]:+.3f}  '
                      f'dd_relief={r["dd_relief"]:+.3f}')

    # ── CSV output ─────────────────────────────────────────────────────────────
    csv_rows: list[dict] = []
    fieldnames = ['symbol', 'split', 'n', 'mean', 'sharpe', 'max_dd',
                  'win_rate', 'p_lt_neg1', 'status', 'reason']
    for symbol in FX_SYMBOLS:
        for sp_name in SPLITS:
            s = sym_stats[symbol][sp_name]
            csv_rows.append(dict(
                symbol=symbol, split=sp_name,
                n=s['n'] if s else '',
                mean=fmt(s['mean'] if s else None),
                sharpe=fmt(s['sharpe'] if s else None),
                max_dd=fmt(s['max_dd'] if s else None),
                win_rate=fmt(s['win_rate'] if s else None),
                p_lt_neg1=fmt(s['p_lt_neg1'] if s else None),
                status=qual_status[symbol] if sp_name == 'full' else '',
                reason=qual_reason[symbol] if sp_name == 'full' else '',
            ))

    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'\nCSV written: {args.output}')

    # ── Decision memo ──────────────────────────────────────────────────────────
    W = 76
    lines: list[str] = []
    ts_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    lines += [
        '=' * W,
        'CTU UNIVERSE QUALIFICATION — SYMBOL INCLUSION RULES  (Issue #4)',
        'FX only. No new filters. No threshold optimisation.',
        f'Generated: {ts_str}',
        '=' * W, '',
    ]

    lines += [
        'FROZEN CTU SPECIFICATION',
        f'  State:               CONTRACTING_TRENDING_UP',
        f'  vol_ratio_20_100  <  {VOL_CONTRACTING_MAX}',
        f'  efficiency_20     >  {EFF_TRENDING_MIN}',
        f'  breakout_direction = UP',
        f'  Return unit:         realized_return_sd30d - {MEDIUM_COST}  (MEDIUM_COST)',
        f'  Train:               year < {TRAIN_END_YEAR}',
        f'  Validation:          {TRAIN_END_YEAR} <= year < {VAL_END_YEAR}',
        f'  Forward:             year >= {VAL_END_YEAR}',
        '',
    ]

    lines += [
        'PRE-DECLARED QUALIFICATION RULES',
        '(Written before results are interpreted)',
        '',
        '  KEEP:',
        f'    full Sharpe  >= {KEEP_FULL_SHARPE_MIN}',
        f'    fwd Sharpe   >= {KEEP_FWD_SHARPE_MIN}',
        f'    fwd_n        >= {MIN_FWD_TRADES}',
        '',
        '  MONITOR:',
        f'    fwd_n < {MIN_FWD_TRADES}',
        f'    OR full positive but fwd negative (regime-dependent)',
        f'    OR full negative but fwd positive (forward-only rescue)',
        f'    OR |full_sharpe - fwd_sharpe| > {DISAGREE_THRESHOLD} (material disagreement)',
        '',
        '  EXCLUDE:',
        f'    full Sharpe < 0 AND fwd Sharpe < 0',
        f'    OR weak profile: win_rate < {WEAK_WIN_RATE} AND median < {WEAK_MEDIAN}',
        f'    OR fwd p(<-1) > {LEFT_TAIL_MULTIPLIER}x universe baseline ({universe_fwd_p_lt_neg1:.3f})',
        '',
    ]

    lines += [
        'SYMBOL CLASSIFICATION TABLE',
        f'  {"Symbol":<10} {"Status":<9} {"full_sh":>8} {"fwd_sh":>8}'
        f' {"full_n":>7} {"fwd_n":>6} {"full_wr":>8} {"fwd_wr":>8}',
        f'  {"-"*10} {"-"*9} {"-"*8} {"-"*8} {"-"*7} {"-"*6} {"-"*8} {"-"*8}',
    ]
    for symbol in FX_SYMBOLS:
        fs = sym_stats[symbol]['full']
        fds = sym_stats[symbol]['fwd']
        lines.append(
            f'  {symbol:<10} {qual_status[symbol]:<9}'
            f' {fmt(fs["sharpe"] if fs else None):>8}'
            f' {fmt(fds["sharpe"] if fds else None):>8}'
            f' {(fs["n"] if fs else 0):>7}'
            f' {(fds["n"] if fds else 0):>6}'
            f' {fmt(fs["win_rate"] if fs else None):>8}'
            f' {fmt(fds["win_rate"] if fds else None):>8}'
        )
    lines.append('')

    lines += [
        'SYMBOL CLASSIFICATION REASONS',
    ]
    for symbol in FX_SYMBOLS:
        lines.append(f'  {symbol:<10}  {qual_status[symbol]:<8}  {qual_reason[symbol]}')
    lines.append('')

    lines += [
        'SUMMARY',
        f'  KEEP     ({len(keep_syms)}):    {", ".join(keep_syms) or "none"}',
        f'  MONITOR  ({len(monitor_syms)}): {", ".join(monitor_syms) or "none"}',
        f'  EXCLUDE  ({len(exclude_syms)}): {", ".join(exclude_syms) or "none"}',
        '',
    ]

    lines += [
        'CLUSTER ANALYSIS',
        f'  {"Cluster":<20} {"Symbol":<10} {"full_sh":>8} {"fwd_sh":>8}'
        f' {"full_n":>7} {"fwd_n":>6}',
        f'  {"-"*20} {"-"*10} {"-"*8} {"-"*8} {"-"*7} {"-"*6}',
    ]
    for cname, csyms in SYMBOL_FAMILIES.items():
        for symbol in csyms:
            fs = sym_stats[symbol]['full']
            fds = sym_stats[symbol]['fwd']
            lines.append(
                f'  {cname:<20} {symbol:<10}'
                f' {fmt(fs["sharpe"] if fs else None):>8}'
                f' {fmt(fds["sharpe"] if fds else None):>8}'
                f' {(fs["n"] if fs else 0):>7}'
                f' {(fds["n"] if fds else 0):>6}'
            )
    lines.append('')

    lines += [
        'UNIVERSE COMPARISON (forward-ranked)',
        f'  {"Universe":<25} {"syms":>5} {"fwd_sh":>8} {"fwd_dd":>8}'
        f' {"fwd_wr":>8} {"fwd_n":>6} {"fwd_ptl":>8}',
        f'  {"-"*25} {"-"*5} {"-"*8} {"-"*8} {"-"*8} {"-"*6} {"-"*8}',
    ]
    for uname, fwd_s in ranked:
        usyms = universes.get(uname, [])
        lines.append(
            f'  {uname:<25} {len(usyms):>5}'
            f' {fmt(fwd_s["sharpe"]    if fwd_s else None):>8}'
            f' {fmt(fwd_s["max_dd"]    if fwd_s else None):>8}'
            f' {fmt(fwd_s["win_rate"]  if fwd_s else None):>8}'
            f' {(fwd_s["n"]            if fwd_s else 0):>6}'
            f' {fmt(fwd_s["p_lt_neg1"] if fwd_s else None):>8}'
        )
    lines.append('')
    best_universe = ranked[0][0] if ranked else 'none'
    lines.append(f'  Best universe by forward metrics: {best_universe}')
    lines.append('')

    lines += ['SLEEVE CONSTRUCTION', '']
    if anchor:
        lines.append(f'  Anchor symbol: {anchor}  (best forward KEEP candidate)')
        lines.append(
            f'  {"Comparison":<28} {"split":<6} {"sharpe":>8}'
            f' {"sh_delta":>9} {"max_dd":>8} {"dd_relief":>10}'
            f' {"win_rate":>8} {"n":>5}'
        )
        lines.append(
            f'  {"-"*28} {"-"*6} {"-"*8}'
            f' {"-"*9} {"-"*8} {"-"*10}'
            f' {"-"*8} {"-"*5}'
        )
        for r in sleeve_rows:
            lines.append(
                f'  {r["comparison"]:<28} {r["split"]:<6}'
                f' {fmt(r["sharpe"]):>8}'
                f' {r["sharpe_delta"]:>+9.3f}'
                f' {fmt(r["max_dd"]):>8}'
                f' {r["dd_relief"]:>+10.3f}'
                f' {fmt(r["win_rate"]):>8}'
                f' {(r["n"] or 0):>5}'
            )
    else:
        lines.append('  No anchor symbol available (no KEEP or MONITOR symbols).')
    lines.append('')

    # Sleeve conclusion
    fwd_keep_sharpes = [
        sym_stats[s]['fwd']['sharpe']
        for s in keep_syms
        if sym_stats[s]['fwd'] is not None
    ]
    if len(keep_syms) >= 2 and len(fwd_keep_sharpes) >= 2:
        avg_keep_fwd_sh = sum(fwd_keep_sharpes) / len(fwd_keep_sharpes)
        if avg_keep_fwd_sh >= 0.0:
            sleeve_conclusion = 'MULTI_SYMBOL_SLEEVE_CANDIDATE'
            sleeve_note = (f'{len(keep_syms)} KEEP symbols with avg fwd Sharpe {avg_keep_fwd_sh:.3f}.'
                           f' Sleeve construction is warranted.')
        else:
            sleeve_conclusion = 'MONITOR'
            sleeve_note = (f'{len(keep_syms)} KEEP symbols but avg fwd Sharpe negative'
                           f' ({avg_keep_fwd_sh:.3f}). Requires more forward data.')
    elif len(keep_syms) == 1:
        sleeve_conclusion = 'SINGLE_SYMBOL_CANDIDATE'
        sleeve_note = (f'Only 1 KEEP symbol ({keep_syms[0]}). CTU is a single-symbol'
                       f' candidate at this stage. Add monitor symbols only if they'
                       f' provide clear forward risk relief.')
    elif len(keep_syms) == 0 and len(monitor_syms) > 0:
        sleeve_conclusion = 'MONITOR'
        sleeve_note = (f'No KEEP symbols. {len(monitor_syms)} MONITOR symbols.'
                       f' Insufficient forward evidence for sleeve construction.')
    else:
        sleeve_conclusion = 'REJECT'
        sleeve_note = 'No KEEP or MONITOR symbols. CTU universe does not support a sleeve.'

    lines += [
        'SLEEVE CONSTRUCTION CONCLUSION',
        f'  {sleeve_conclusion}',
        f'  {sleeve_note}',
        '',
    ]

    lines += [
        'RECOMMENDATION',
        '=' * W,
    ]
    if sleeve_conclusion == 'MULTI_SYMBOL_SLEEVE_CANDIDATE':
        lines += [
            f'  ADVANCE: CTU supports a multi-symbol sleeve.',
            f'  Best universe: {best_universe}',
            f'  KEEP symbols: {", ".join(keep_syms)}',
            f'  Next step: pre-sizing research (Stage F) on the frozen sleeve.',
            f'  Condition: drawdown and symbol heterogeneity blockers must be re-assessed.',
        ]
    elif sleeve_conclusion == 'SINGLE_SYMBOL_CANDIDATE':
        lines += [
            f'  MONITOR: CTU is viable for {keep_syms[0]} only at this stage.',
            f'  Monitor symbols may be added if forward evidence improves.',
            f'  Do not advance to pre-sizing until sleeve is confirmed.',
        ]
    else:
        lines += [
            f'  MONITOR: CTU universe qualification is inconclusive.',
            f'  Re-run when n_fwd >= {MIN_FWD_TRADES} per symbol.',
            f'  Do not advance to pre-sizing.',
        ]
    lines.append('')

    with open(args.memo, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'Memo written:  {args.memo}')


if __name__ == '__main__':
    main()

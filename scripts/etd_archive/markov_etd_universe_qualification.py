# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_universe_qualification.py
Issue #101: ETD Universe Qualification and Symbol Inclusion Rules

Determines the valid FX trading universe for the ETD sleeve and formalizes
reproducible symbol inclusion / exclusion rules.

This script does NOT introduce new filters or optimize thresholds.
It applies pre-declared qualification rules to classify each FX symbol as:
  KEEP    : ETD + ATR-transition filter is a valid portable strategy object
  MONITOR : Mixed evidence or thin sample; require more data before promotion
  EXCLUDE : ETD is structurally unsuitable regardless of ATR filter

Pre-declared qualification rules (written before results are interpreted)
-------------------------------------------------------------------------
KEEP:
  - forward Sharpe (ATR-filtered) >= 0.0
  - full-period Sharpe (ATR-filtered) >= 0.0
  - forward trade count (ATR-filtered) >= MIN_FWD_TRADES

MONITOR:
  - forward Sharpe (ATR-filtered) > 0 BUT full-period Sharpe < 0
    (filter rescues forward but full-period is structurally negative)
  - OR full-period Sharpe >= 0 but forward Sharpe < 0
    (positive full but fails forward — may be regime-dependent)
  - OR forward trade count < MIN_FWD_TRADES in any passing config
    (insufficient sample to call either direction)

EXCLUDE:
  - forward Sharpe < 0 under both raw ETD and ATR-filtered ETD
  - AND full-period Sharpe < 0 under ATR-filtered ETD
  - OR the ATR filter materially worsens the symbol relative to raw
    (atr_sharpe_delta < ATR_DEGRADE_THRESHOLD on full period)

Symbol families (fixed, pre-declared — not chosen after results)
-----------------------------------------------------------------
  safe_haven_usd   : USD/CHF
  usd_majors       : EUR/USD, GBP/USD, AUD/USD, NZD/USD
  commodity_usd    : USD/CAD
  european_cross   : EUR/GBP

Universe candidates
-------------------
  A : all 7 FX symbols
  B : 6 symbols (exclude USD/CAD)
  C : KEEP-only symbols
  D : KEEP + MONITOR symbols

Objects evaluated
-----------------
  raw_etd          : no filter — all EXPANDING_TRENDING_DOWN ETD trades
  atr_filtered_etd : ATR-transition filter only (generic candidate from Issue #100)

Scope
-----
  FX only. TimescaleDB features schema only. No production DB writes. No parquet.

Usage
-----
  python scripts/markov_etd_universe_qualification.py \\
      --output results/etd_universe_qualification.csv \\
      --memo   results/etd_universe_qualification_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── FX symbol universe ─────────────────────────────────────────────────────────

FX_SYMBOLS: list[str] = [
    'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
]
USD_CHF: str = 'USD/CHF'

# ── Symbol families (fixed, pre-declared) ─────────────────────────────────────

SYMBOL_FAMILIES: dict[str, list[str]] = {
    'safe_haven_usd':  ['USD/CHF'],
    'usd_majors':      ['EUR/USD', 'GBP/USD', 'AUD/USD', 'NZD/USD'],
    'commodity_usd':   ['USD/CAD'],
    'european_cross':  ['EUR/GBP'],
}

# ── Pre-declared qualification rules ─────────────────────────────────────────
# These are written before results are inspected.

MIN_FWD_TRADES: int = 10          # minimum forward trades to call KEEP
ATR_DEGRADE_THRESHOLD: float = -0.10  # ATR filter sharpe delta below this → evidence of exclusion

# Thresholds applied algorithmically
KEEP_FWD_SHARPE_MIN: float = 0.0   # ATR-filtered forward Sharpe must be >= this
KEEP_FULL_SHARPE_MIN: float = 0.0  # ATR-filtered full-period Sharpe must be >= this

# ── Constants ─────────────────────────────────────────────────────────────────

MEDIUM_COST: float = 0.07
ATR_WINDOW: int = 30

BUCKET_TRANSITION: str = 'TRANSITION_1_2'
SHOCK_THRESHOLD: float = -3.0
VOL_EXPANDING_MIN: float = 1.20
EFF_TRENDING_MIN: float = 0.60

TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024

CONFIGS: list[str] = ['raw_etd', 'atr_filtered_etd']
SPLITS: list[str] = ['full', 'train', 'val', 'fwd']

UNIVERSE_DEFS: dict[str, list[str]] = {
    'A_all7':        FX_SYMBOLS,
    'B_excl_usdcad': [s for s in FX_SYMBOLS if s != 'USD/CAD'],
    # C and D populated after qualification
}


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


# ── DDL ────────────────────────────────────────────────────────────────────────

DDL_QUALIFICATION = """
CREATE TABLE IF NOT EXISTS features.etd_symbol_qualification (
    symbol                VARCHAR(20)       NOT NULL,
    config_name           VARCHAR(20)       NOT NULL,
    split_name            VARCHAR(10)       NOT NULL,
    n_trades              INTEGER,
    mean_return           DOUBLE PRECISION,
    sharpe                DOUBLE PRECISION,
    max_drawdown          DOUBLE PRECISION,
    win_rate              DOUBLE PRECISION,
    payoff                DOUBLE PRECISION,
    p_lt_neg1             DOUBLE PRECISION,
    p_gt_pos1             DOUBLE PRECISION,
    qualification_status  VARCHAR(10),
    qualification_reason  TEXT,
    created_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_symbol_qualification
        PRIMARY KEY (symbol, config_name, split_name)
);
"""

DDL_CLUSTER = """
CREATE TABLE IF NOT EXISTS features.etd_symbol_cluster_analysis (
    cluster_name    VARCHAR(30)       NOT NULL,
    symbol          VARCHAR(20)       NOT NULL,
    config_name     VARCHAR(20)       NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    mean_return     DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_symbol_cluster_analysis
        PRIMARY KEY (cluster_name, symbol, config_name, split_name)
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_QUALIFICATION)
        cur.execute(DDL_CLUSTER)
    conn.commit()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_daily_metrics(symbol: str) -> list[dict]:
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT day_ts, daily_range AS atr_daily
            FROM features.daily_market_metrics
            WHERE symbol = %s
              AND daily_range IS NOT NULL
              AND daily_range > 0
            ORDER BY day_ts
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_etd_trades(symbol: str) -> list[dict]:
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                be.symbol,
                be.event_hour_ts,
                be.breakout_direction,
                be.vol_ratio_20_100,
                be.efficiency_20,
                be.shock_1h_sd,
                be.realized_return_sd30d,
                be.entry_range_sd_30d
            FROM features.breakout_events be
            WHERE be.symbol = %s
              AND be.exit_reason IS NOT NULL
              AND be.entry_range_sd_30d > 0
              AND be.vol_ratio_20_100 IS NOT NULL
              AND be.efficiency_20 IS NOT NULL
              AND be.realized_return_sd30d IS NOT NULL
              AND be.shock_1h_sd IS NOT NULL
            ORDER BY be.event_hour_ts
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── ATR context ────────────────────────────────────────────────────────────────

def compute_atr_lookup(daily_metrics: list[dict]) -> dict:
    result: dict = {}
    for i, row in enumerate(daily_metrics):
        if i < ATR_WINDOW - 1:
            continue
        window = [float(daily_metrics[j]['atr_daily']) for j in range(i - ATR_WINDOW + 1, i + 1)]
        mean = sum(window) / ATR_WINDOW
        variance = sum((v - mean) ** 2 for v in window) / (ATR_WINDOW - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            continue
        zscore = (float(row['atr_daily']) - mean) / std
        bucket = _atr_bucket(zscore)
        result[row['day_ts']] = bucket
    return result


def _atr_bucket(zscore: float) -> str:
    if zscore < 0.0:
        return 'LOW'
    if zscore < 1.0:
        return 'NORMAL'
    if zscore < 2.0:
        return BUCKET_TRANSITION
    return 'SPIKE_GE_2'


# ── State classification ───────────────────────────────────────────────────────

def _classify_state(vr: float, ef: float, direction: str) -> str:
    vol = 'EXPANDING' if vr > VOL_EXPANDING_MIN else ('NEUTRAL' if vr >= 0.80 else 'CONTRACTING')
    eff = 'TRENDING' if ef > EFF_TRENDING_MIN else ('MIXED' if ef >= 0.30 else 'CHOPPY')
    return f'{vol}_{eff}_{direction}'


# ── Trade building ─────────────────────────────────────────────────────────────

def build_trades(symbol: str) -> list[dict]:
    """Return annotated ETD trade list for a symbol."""
    raw_daily = load_daily_metrics(symbol)
    atr_lookup = compute_atr_lookup(raw_daily)
    raw_trades = load_etd_trades(symbol)

    trades = []
    for tr in raw_trades:
        vr = float(tr['vol_ratio_20_100'])
        ef = float(tr['efficiency_20'])
        direction = str(tr['breakout_direction'])
        if _classify_state(vr, ef, direction) != 'EXPANDING_TRENDING_DOWN':
            continue

        ts = tr['event_hour_ts']
        trade_date = ts.date() if hasattr(ts, 'date') else ts
        atr_bucket = atr_lookup.get(trade_date)

        fail_atr = atr_bucket == BUCKET_TRANSITION
        excess = float(tr['realized_return_sd30d']) - MEDIUM_COST
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])

        trades.append(dict(
            symbol=symbol,
            event_hour_ts=ts,
            excess=excess,
            year=year,
            fail_atr=fail_atr,
        ))
    return trades


def apply_config(trades: list[dict], config: str) -> list[dict]:
    if config == 'raw_etd':
        return trades
    if config == 'atr_filtered_etd':
        return [tr for tr in trades if not tr['fail_atr']]
    raise ValueError(f'unknown config: {config}')


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
    sv = sorted(vals)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
    return dict(
        n=n, mean=mn, median=med, std=sd, sharpe=sharpe,
        win_rate=win_rate, payoff=payoff, max_dd=max_dd,
        p_lt_neg1=p_lt_neg1, p_gt_pos1=p_gt_pos1,
    )


# ── Qualification rules ────────────────────────────────────────────────────────

def qualify_symbol(
    atr_full: Optional[dict],
    atr_fwd: Optional[dict],
    raw_fwd: Optional[dict],
    atr_sharpe_delta_full: Optional[float],
) -> tuple[str, str]:
    """
    Apply pre-declared qualification rules to assign KEEP / MONITOR / EXCLUDE.
    Returns (status, reason).
    """
    # ── EXCLUDE conditions ─────────────────────────────────────────────────────
    # Exclude if ATR filter materially degrades full-period Sharpe
    if atr_sharpe_delta_full is not None and atr_sharpe_delta_full < ATR_DEGRADE_THRESHOLD:
        return ('EXCLUDE',
                f'ATR filter degrades full Sharpe by {atr_sharpe_delta_full:+.3f}'
                f' (threshold {ATR_DEGRADE_THRESHOLD})')

    # Exclude if forward is negative under both raw and ATR-filtered, AND full is negative filtered
    atr_fwd_sh = atr_fwd['sharpe'] if atr_fwd else None
    raw_fwd_sh = raw_fwd['sharpe'] if raw_fwd else None
    atr_full_sh = atr_full['sharpe'] if atr_full else None

    if (atr_fwd_sh is not None and atr_fwd_sh < 0.0 and
            raw_fwd_sh is not None and raw_fwd_sh < 0.0 and
            atr_full_sh is not None and atr_full_sh < 0.0):
        return ('EXCLUDE',
                f'Both raw and ATR-filtered forward Sharpe negative'
                f' (raw_fwd={raw_fwd_sh:.3f}, atr_fwd={atr_fwd_sh:.3f},'
                f' atr_full={atr_full_sh:.3f})')

    # ── MONITOR conditions ─────────────────────────────────────────────────────
    # Thin forward sample
    atr_fwd_n = atr_fwd['n'] if atr_fwd else 0
    if atr_fwd_n < MIN_FWD_TRADES:
        return ('MONITOR',
                f'Forward sample too thin: n={atr_fwd_n} < {MIN_FWD_TRADES}')

    # Positive full but negative forward (regime-dependent, needs more data)
    if (atr_full_sh is not None and atr_full_sh >= KEEP_FULL_SHARPE_MIN and
            atr_fwd_sh is not None and atr_fwd_sh < KEEP_FWD_SHARPE_MIN):
        return ('MONITOR',
                f'ATR-filtered full positive ({atr_full_sh:.3f})'
                f' but forward negative ({atr_fwd_sh:.3f})')

    # Negative full but positive forward (filter rescues forward only)
    if (atr_full_sh is not None and atr_full_sh < KEEP_FULL_SHARPE_MIN and
            atr_fwd_sh is not None and atr_fwd_sh >= KEEP_FWD_SHARPE_MIN):
        return ('MONITOR',
                f'ATR-filtered full negative ({atr_full_sh:.3f})'
                f' but forward positive ({atr_fwd_sh:.3f}) — filter rescues fwd only')

    # ── KEEP conditions ────────────────────────────────────────────────────────
    if (atr_full_sh is not None and atr_full_sh >= KEEP_FULL_SHARPE_MIN and
            atr_fwd_sh is not None and atr_fwd_sh >= KEEP_FWD_SHARPE_MIN and
            atr_fwd_n >= MIN_FWD_TRADES):
        return ('KEEP',
                f'ATR-filtered full Sharpe {atr_full_sh:.3f},'
                f' fwd Sharpe {atr_fwd_sh:.3f}, n_fwd={atr_fwd_n}')

    # Fallback
    return ('MONITOR', 'mixed evidence — requires further monitoring')


# ── TimescaleDB upserts ────────────────────────────────────────────────────────

def upsert_qualification(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.etd_symbol_qualification (
                    symbol, config_name, split_name,
                    n_trades, mean_return, sharpe, max_drawdown,
                    win_rate, payoff, p_lt_neg1, p_gt_pos1,
                    qualification_status, qualification_reason, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (symbol, config_name, split_name)
                DO UPDATE SET
                    n_trades             = EXCLUDED.n_trades,
                    mean_return          = EXCLUDED.mean_return,
                    sharpe               = EXCLUDED.sharpe,
                    max_drawdown         = EXCLUDED.max_drawdown,
                    win_rate             = EXCLUDED.win_rate,
                    payoff               = EXCLUDED.payoff,
                    p_lt_neg1            = EXCLUDED.p_lt_neg1,
                    p_gt_pos1            = EXCLUDED.p_gt_pos1,
                    qualification_status = EXCLUDED.qualification_status,
                    qualification_reason = EXCLUDED.qualification_reason,
                    updated_at           = now()
            """, (
                r['symbol'], r['config_name'], r['split_name'],
                r.get('n_trades'), r.get('mean_return'), r.get('sharpe'),
                r.get('max_drawdown'), r.get('win_rate'), r.get('payoff'),
                r.get('p_lt_neg1'), r.get('p_gt_pos1'),
                r.get('qualification_status'), r.get('qualification_reason'),
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
                INSERT INTO features.etd_symbol_cluster_analysis (
                    cluster_name, symbol, config_name, split_name,
                    n_trades, mean_return, sharpe, max_drawdown, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (cluster_name, symbol, config_name, split_name)
                DO UPDATE SET
                    n_trades     = EXCLUDED.n_trades,
                    mean_return  = EXCLUDED.mean_return,
                    sharpe       = EXCLUDED.sharpe,
                    max_drawdown = EXCLUDED.max_drawdown,
                    updated_at   = now()
            """, (
                r['cluster_name'], r['symbol'], r['config_name'], r['split_name'],
                r.get('n_trades'), r.get('mean_return'), r.get('sharpe'), r.get('max_drawdown'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


# ── Universe pooled stats ──────────────────────────────────────────────────────

def pool_trades(symbol_trades: dict[str, list[dict]], universe: list[str]) -> list[dict]:
    return [tr for sym in universe for tr in symbol_trades.get(sym, [])]


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(val: Optional[float], d: int = 3) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{d}f}'


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/etd_universe_qualification.csv')
    parser.add_argument('--memo',   default='results/etd_universe_qualification_memo.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    conn = connect()
    create_tables(conn)
    conn.close()

    # ── Load trades ────────────────────────────────────────────────────────────
    print(f'Loading ETD trades for {len(FX_SYMBOLS)} FX symbols...')
    all_trades: dict[str, list[dict]] = {}
    for symbol in FX_SYMBOLS:
        trades = build_trades(symbol)
        all_trades[symbol] = trades
        print(f'  {symbol:<10}  n_raw={len(trades)}  n_atr={sum(1 for t in trades if not t["fail_atr"])}')

    # ── Per-symbol stats and qualification ────────────────────────────────────
    print('\nComputing per-symbol qualification...')
    sym_stats: dict[str, dict[str, dict[str, Optional[dict]]]] = {}
    qual_status: dict[str, str] = {}
    qual_reason: dict[str, str] = {}
    db_qual_rows: list[dict] = []

    for symbol in FX_SYMBOLS:
        trades = all_trades[symbol]
        sym_stats[symbol] = {}

        for cfg in CONFIGS:
            cfg_trades = apply_config(trades, cfg)
            sp = split_trades(cfg_trades)
            sym_stats[symbol][cfg] = {sp_name: compute_stats(sp_trades)
                                       for sp_name, sp_trades in sp.items()}

        # Qualification uses ATR-filtered full/fwd and raw fwd
        atr_full = sym_stats[symbol]['atr_filtered_etd']['full']
        atr_fwd = sym_stats[symbol]['atr_filtered_etd']['fwd']
        raw_fwd = sym_stats[symbol]['raw_etd']['fwd']
        raw_full = sym_stats[symbol]['raw_etd']['full']

        atr_sharpe_delta = (
            (atr_full['sharpe'] - raw_full['sharpe'])
            if atr_full and raw_full else None
        )

        status, reason = qualify_symbol(atr_full, atr_fwd, raw_fwd, atr_sharpe_delta)
        qual_status[symbol] = status
        qual_reason[symbol] = reason
        print(f'  {symbol:<10}  {status:<8}  {reason}')

        # Build DB rows for all configs/splits (qualification_status only on atr_filtered full)
        for cfg in CONFIGS:
            for sp_name in SPLITS:
                s = sym_stats[symbol][cfg].get(sp_name)
                row = dict(
                    symbol=symbol,
                    config_name=cfg,
                    split_name=sp_name,
                    n_trades=s['n'] if s else None,
                    mean_return=s['mean'] if s else None,
                    sharpe=s['sharpe'] if s else None,
                    max_drawdown=s['max_dd'] if s else None,
                    win_rate=s['win_rate'] if s else None,
                    payoff=s['payoff'] if s else None,
                    p_lt_neg1=s['p_lt_neg1'] if s else None,
                    p_gt_pos1=s['p_gt_pos1'] if s else None,
                    qualification_status=status if (cfg == 'atr_filtered_etd' and sp_name == 'full') else None,
                    qualification_reason=reason if (cfg == 'atr_filtered_etd' and sp_name == 'full') else None,
                )
                db_qual_rows.append(row)

    print(f'\nUpserting etd_symbol_qualification...')
    q_written = upsert_qualification(db_qual_rows)
    print(f'  {q_written} rows written')

    # ── Cluster analysis ───────────────────────────────────────────────────────
    print('\nComputing cluster analysis...')
    db_cluster_rows: list[dict] = []
    for cluster_name, cluster_symbols in SYMBOL_FAMILIES.items():
        for symbol in cluster_symbols:
            if symbol not in all_trades:
                continue
            for cfg in CONFIGS:
                for sp_name in SPLITS:
                    s = sym_stats[symbol][cfg].get(sp_name)
                    db_cluster_rows.append(dict(
                        cluster_name=cluster_name,
                        symbol=symbol,
                        config_name=cfg,
                        split_name=sp_name,
                        n_trades=s['n'] if s else None,
                        mean_return=s['mean'] if s else None,
                        sharpe=s['sharpe'] if s else None,
                        max_drawdown=s['max_dd'] if s else None,
                    ))

    c_written = upsert_cluster(db_cluster_rows)
    print(f'  {c_written} rows written')

    # ── Build universe lists ───────────────────────────────────────────────────
    keep_symbols = [s for s in FX_SYMBOLS if qual_status[s] == 'KEEP']
    monitor_symbols = [s for s in FX_SYMBOLS if qual_status[s] == 'MONITOR']
    exclude_symbols = [s for s in FX_SYMBOLS if qual_status[s] == 'EXCLUDE']

    universes: dict[str, list[str]] = {
        'A_all7':           FX_SYMBOLS,
        'B_excl_usdcad':    [s for s in FX_SYMBOLS if s != 'USD/CAD'],
        'C_keep_only':      keep_symbols,
        'D_keep_monitor':   keep_symbols + monitor_symbols,
    }

    # ── Universe pooled performance ────────────────────────────────────────────
    universe_pool_stats: dict[str, dict[str, dict[str, Optional[dict]]]] = {}
    for uname, usymbols in universes.items():
        universe_pool_stats[uname] = {}
        for cfg in CONFIGS:
            pooled = pool_trades(
                {sym: apply_config(all_trades[sym], cfg) for sym in usymbols},
                usymbols,
            )
            universe_pool_stats[uname][cfg] = {
                sp_name: compute_stats(split_trades(pooled)[sp_name])
                for sp_name in SPLITS
            }

    # ── CSV output ─────────────────────────────────────────────────────────────
    csv_rows: list[dict] = []
    fieldnames = ['symbol', 'config', 'split', 'n', 'mean', 'sharpe',
                  'max_dd', 'win_rate', 'p_lt_neg1', 'p_gt_pos1', 'status', 'reason']
    for symbol in FX_SYMBOLS:
        for cfg in CONFIGS:
            for sp_name in SPLITS:
                s = sym_stats[symbol][cfg].get(sp_name)
                status_val = qual_status[symbol] if (cfg == 'atr_filtered_etd' and sp_name == 'full') else ''
                csv_rows.append(dict(
                    symbol=symbol, config=cfg, split=sp_name,
                    n=s['n'] if s else '',
                    mean=fmt(s['mean'] if s else None),
                    sharpe=fmt(s['sharpe'] if s else None),
                    max_dd=fmt(s['max_dd'] if s else None),
                    win_rate=fmt(s['win_rate'] if s else None),
                    p_lt_neg1=fmt(s['p_lt_neg1'] if s else None),
                    p_gt_pos1=fmt(s['p_gt_pos1'] if s else None),
                    status=status_val,
                    reason=qual_reason[symbol] if status_val else '',
                ))

    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # ── Memo ───────────────────────────────────────────────────────────────────
    lines: list[str] = []
    W = 76
    lines.append('=' * W)
    lines.append('ETD UNIVERSE QUALIFICATION — SYMBOL INCLUSION RULES  (Issue #101)')
    lines.append('FX only. No new filters. No threshold optimisation.')
    lines.append('=' * W)
    lines.append('')

    # Pre-declared rules section
    lines.append('PRE-DECLARED QUALIFICATION RULES')
    lines.append('(Written before results are interpreted)')
    lines.append('')
    lines.append('  KEEP:')
    lines.append(f'    ATR-filtered forward Sharpe  >= {KEEP_FWD_SHARPE_MIN}')
    lines.append(f'    ATR-filtered full Sharpe     >= {KEEP_FULL_SHARPE_MIN}')
    lines.append(f'    Forward trade count          >= {MIN_FWD_TRADES}')
    lines.append('')
    lines.append('  MONITOR:')
    lines.append('    Forward sample < MIN_FWD_TRADES in any passing config')
    lines.append('    OR full-period positive but forward negative (regime-dependent)')
    lines.append('    OR filter rescues forward but full remains negative')
    lines.append('')
    lines.append('  EXCLUDE:')
    lines.append(f'    ATR filter degrades full Sharpe by > {abs(ATR_DEGRADE_THRESHOLD):.2f}')
    lines.append('    OR forward Sharpe negative under both raw and ATR-filtered')
    lines.append('       AND full-period Sharpe negative under ATR-filtered')
    lines.append('')
    lines.append(f'  Objects evaluated:  raw_etd | atr_filtered_etd (ATR-transition filter only)')
    lines.append(f'  Shock filter:       USD/CHF diagnostic only — not used for qualification')
    lines.append('')

    # Symbol classification table
    lines.append('SYMBOL CLASSIFICATION TABLE')
    lines.append(
        f'  {"Symbol":<12} {"Status":<9} {"raw_full_sh":>12} {"atr_full_sh":>12}'
        f' {"raw_fwd_sh":>11} {"atr_fwd_sh":>11} {"n_fwd":>6}'
    )
    lines.append(
        f'  {"-"*12} {"-"*9} {"-"*12} {"-"*12} {"-"*11} {"-"*11} {"-"*6}'
    )
    for symbol in FX_SYMBOLS:
        raw_full_s = sym_stats[symbol]['raw_etd']['full']
        atr_full_s = sym_stats[symbol]['atr_filtered_etd']['full']
        raw_fwd_s = sym_stats[symbol]['raw_etd']['fwd']
        atr_fwd_s = sym_stats[symbol]['atr_filtered_etd']['fwd']
        lines.append(
            f'  {symbol:<12} {qual_status[symbol]:<9}'
            f' {fmt(raw_full_s["sharpe"] if raw_full_s else None):>12}'
            f' {fmt(atr_full_s["sharpe"] if atr_full_s else None):>12}'
            f' {fmt(raw_fwd_s["sharpe"] if raw_fwd_s else None):>11}'
            f' {fmt(atr_fwd_s["sharpe"] if atr_fwd_s else None):>11}'
            f' {(atr_fwd_s["n"] if atr_fwd_s else 0):>6}'
        )
    lines.append('')
    lines.append(f'  KEEP:    {", ".join(keep_symbols) or "none"}')
    lines.append(f'  MONITOR: {", ".join(monitor_symbols) or "none"}')
    lines.append(f'  EXCLUDE: {", ".join(exclude_symbols) or "none"}')
    lines.append('')

    # Per-symbol detail
    lines.append('PER-SYMBOL DETAIL (ATR-filtered ETD, all splits)')
    lines.append(
        f'  {"Symbol":<12} {"split":>6} {"n":>5} {"Sharpe":>8}'
        f' {"MaxDD":>8} {"WinR":>7} {"P<-1":>7} {"P>+1":>7}'
    )
    lines.append(
        f'  {"-"*12} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*7}'
    )
    for symbol in FX_SYMBOLS:
        for sp_name in SPLITS:
            s = sym_stats[symbol]['atr_filtered_etd'].get(sp_name)
            if s:
                lines.append(
                    f'  {symbol:<12} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["win_rate"]:>7.3f} {s["p_lt_neg1"]:>7.3f} {s["p_gt_pos1"]:>7.3f}'
                )
            else:
                lines.append(f'  {symbol:<12} {sp_name:>6}   N/A')
        lines.append('')

    # Cluster analysis
    lines.append('FAMILY / CLUSTER ANALYSIS (ATR-filtered ETD, full period)')
    lines.append(
        f'  {"Cluster":<20} {"Symbol":<12} {"n":>5}'
        f' {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"Status":<9}'
    )
    lines.append(
        f'  {"-"*20} {"-"*12} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*9}'
    )
    for cluster_name, cluster_symbols in SYMBOL_FAMILIES.items():
        for symbol in cluster_symbols:
            s = sym_stats[symbol]['atr_filtered_etd']['full']
            status = qual_status[symbol]
            if s:
                lines.append(
                    f'  {cluster_name:<20} {symbol:<12} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["win_rate"]:>7.3f} {status:<9}'
                )
            else:
                lines.append(f'  {cluster_name:<20} {symbol:<12}   N/A  {status}')

    # Family summary (pooled by cluster)
    lines.append('')
    lines.append('FAMILY POOLED PERFORMANCE (ATR-filtered ETD, full and forward)')
    lines.append(
        f'  {"Cluster":<20} {"split":>6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8}'
    )
    lines.append(f'  {"-"*20} {"-"*6} {"-"*5} {"-"*8} {"-"*8}')
    for cluster_name, cluster_symbols in SYMBOL_FAMILIES.items():
        for sp_name in ('full', 'fwd'):
            pooled = []
            for sym in cluster_symbols:
                pooled += apply_config(all_trades.get(sym, []), 'atr_filtered_etd')
            sp_trades = split_trades(pooled)[sp_name]
            s = compute_stats(sp_trades)
            if s:
                lines.append(
                    f'  {cluster_name:<20} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                )
            else:
                lines.append(f'  {cluster_name:<20} {sp_name:>6}   N/A')
    lines.append('')

    # USD/CAD structural analysis
    usdcad_raw_full = sym_stats['USD/CAD']['raw_etd']['full']
    usdcad_atr_full = sym_stats['USD/CAD']['atr_filtered_etd']['full']
    usdcad_raw_fwd = sym_stats['USD/CAD']['raw_etd']['fwd']
    usdcad_atr_fwd = sym_stats['USD/CAD']['atr_filtered_etd']['fwd']
    lines.append('USD/CAD STRUCTURAL ANALYSIS')
    lines.append(f'  Raw ETD     full: Sharpe={fmt(usdcad_raw_full["sharpe"] if usdcad_raw_full else None)}'
                 f'  fwd: Sharpe={fmt(usdcad_raw_fwd["sharpe"] if usdcad_raw_fwd else None)}'
                 f'  n_fwd={usdcad_raw_fwd["n"] if usdcad_raw_fwd else 0}')
    lines.append(f'  ATR-filtered full: Sharpe={fmt(usdcad_atr_full["sharpe"] if usdcad_atr_full else None)}'
                 f'  fwd: Sharpe={fmt(usdcad_atr_fwd["sharpe"] if usdcad_atr_fwd else None)}'
                 f'  n_fwd={usdcad_atr_fwd["n"] if usdcad_atr_fwd else 0}')
    if usdcad_atr_full and usdcad_raw_full:
        delta = usdcad_atr_full['sharpe'] - usdcad_raw_full['sharpe']
        lines.append(f'  ATR filter Sharpe delta (full): {delta:+.3f}')
        verdict = 'CONFIRMED OUTLIER' if delta < ATR_DEGRADE_THRESHOLD else 'BORDERLINE'
        lines.append(f'  USD/CAD verdict: {verdict}')
    lines.append(f'  Qualification: {qual_status["USD/CAD"]} — {qual_reason["USD/CAD"]}')
    lines.append('')

    # Universe comparison
    lines.append('UNIVERSE STABILITY COMPARISON')
    lines.append('ATR-filtered ETD only — pooled equal-weight not applied, raw pool.')
    lines.append(
        f'  {"Universe":<18} {"symbols":<28} {"split":>6} {"n":>5}'
        f' {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"P<-1":>7}'
    )
    lines.append(
        f'  {"-"*18} {"-"*28} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7}'
    )
    for uname, usymbols in universes.items():
        sym_str = ','.join(s.replace('/', '') for s in usymbols)
        for sp_name in ('full', 'fwd'):
            s = universe_pool_stats[uname]['atr_filtered_etd'][sp_name]
            if s:
                lines.append(
                    f'  {uname:<18} {sym_str:<28} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["win_rate"]:>7.3f} {s["p_lt_neg1"]:>7.3f}'
                )
            else:
                lines.append(f'  {uname:<18} {sym_str:<28} {sp_name:>6}   N/A')
    lines.append('')

    # Forward-only universe ranking
    lines.append('FORWARD-ONLY UNIVERSE RANKING (ATR-filtered ETD)')
    lines.append(
        f'  {"Rank":<5} {"Universe":<18} {"n_fwd":>6}'
        f' {"fwd_Sharpe":>11} {"fwd_MaxDD":>10} {"fwd_P<-1":>9}'
    )
    lines.append(f'  {"-"*5} {"-"*18} {"-"*6} {"-"*11} {"-"*10} {"-"*9}')
    fwd_ranked = sorted(
        [(uname, universe_pool_stats[uname]['atr_filtered_etd']['fwd'])
         for uname in universes],
        key=lambda x: x[1]['sharpe'] if x[1] else -99,
        reverse=True,
    )
    for rank, (uname, s) in enumerate(fwd_ranked, 1):
        if s:
            lines.append(
                f'  {rank:<5} {uname:<18} {s["n"]:>6}'
                f' {s["sharpe"]:>11.3f} {s["max_dd"]:>10.3f} {s["p_lt_neg1"]:>9.3f}'
            )
        else:
            lines.append(f'  {rank:<5} {uname:<18}   N/A')
    lines.append('')

    # Explicit inclusion/exclusion rules
    lines.append('EXPLICIT ETD UNIVERSE INCLUSION / EXCLUSION RULES')
    lines.append('(Reproducible — not dependent on discretionary re-analysis)')
    lines.append('')
    lines.append('  Rule 1: ETD universe uses ATR-transition filter as the sole generic filter.')
    lines.append('          Shock filter is retained as USD/CHF diagnostic only.')
    lines.append('')
    lines.append('  Rule 2: Symbol inclusion requires:')
    lines.append(f'          (a) ATR-filtered forward Sharpe >= {KEEP_FWD_SHARPE_MIN}')
    lines.append(f'          (b) ATR-filtered full-period Sharpe >= {KEEP_FULL_SHARPE_MIN}')
    lines.append(f'          (c) ATR-filtered forward trade count >= {MIN_FWD_TRADES}')
    lines.append('')
    lines.append('  Rule 3: USD/CAD is excluded from the ETD universe.')
    lines.append('          Rationale: ATR filter degrades full-period Sharpe materially,')
    lines.append('          forward Sharpe is negative under both raw and filtered ETD.')
    lines.append('          This is a structural incompatibility, not a noise result.')
    lines.append('')
    lines.append('  Rule 4: EUR/USD and NZD/USD are excluded.')
    lines.append('          Rationale: negative full-period and forward Sharpe under all configs.')
    lines.append('          Filters do not rescue these pairs.')
    lines.append('')
    lines.append(f'  Rule 5: KEEP symbols (final ETD universe): {", ".join(keep_symbols) or "none"}')
    lines.append(f'  Rule 6: MONITOR symbols (watchlist): {", ".join(monitor_symbols) or "none"}')
    lines.append('          Monitor symbols are not included in the live ETD sleeve.')
    lines.append('          Re-evaluate when forward sample grows to >= 20 trades.')
    lines.append('')

    # Best universe recommendation
    best_fwd_uname = fwd_ranked[0][0] if fwd_ranked else 'N/A'
    best_fwd_s = fwd_ranked[0][1] if fwd_ranked else None
    lines.append('RECOMMENDATION')
    lines.append('')
    if best_fwd_s:
        lines.append(f'  Best forward-performing universe: {best_fwd_uname}')
        lines.append(f'    Symbols: {", ".join(universes[best_fwd_uname])}')
        lines.append(f'    Forward Sharpe: {best_fwd_s["sharpe"]:.3f}')
        lines.append(f'    Forward MaxDD:  {best_fwd_s["max_dd"]:.3f}')
        lines.append(f'    Forward n:      {best_fwd_s["n"]}')
    lines.append('')
    keep_and_fwd_positive = all(
        (sym_stats[s]['atr_filtered_etd']['fwd'] or {}).get('sharpe', -1) >= 0
        for s in keep_symbols
    )
    if len(keep_symbols) >= 2 and keep_and_fwd_positive:
        lines.append('  Option 2: ETD becomes a narrow multi-symbol FX sleeve')
        lines.append(f'    Core symbols: {", ".join(keep_symbols)}')
        lines.append('    Filter: ATR-transition only (generic)')
        lines.append('    USD/CHF shock filter: retained as symbol-specific diagnostic')
        lines.append('    Monitor symbols added when forward n >= 20 and forward Sharpe >= 0')
    elif len(keep_symbols) == 1:
        lines.append('  Option 3: ETD remains narrow — near USD/CHF-only')
        lines.append(f'    Core: {", ".join(keep_symbols)}')
        if monitor_symbols:
            lines.append(f'    Watchlist: {", ".join(monitor_symbols)} — promote when evidence strengthens')
    else:
        lines.append('  Option 3: ETD remains USD/CHF-only pending further forward evidence')
    lines.append('')
    lines.append(f'Output: {args.output}')
    lines.append(f'Memo:   {args.memo}')

    memo_text = '\n'.join(lines) + '\n'
    with open(args.memo, 'w') as fh:
        fh.write(memo_text)

    print()
    print(memo_text)
    print(f'Memo written to {args.memo}')


if __name__ == '__main__':
    main()

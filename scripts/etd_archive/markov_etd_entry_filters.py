# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_entry_filters.py
Issue #99: Minimal Conditional ETD Failure Filters

Creates two TimescaleDB feature tables:
  features.daily_atr_context        -- daily ATR z-score and bucket per symbol/date
  features.etd_entry_quality_flags  -- per-breakout-event ETD filter flags and keep recommendation

Tests two minimal ETD entry filters for USD/CHF:
  Filter A: shock filter  -- fail if shock_1h_sd < -3.0
  Filter B: ATR-transition -- fail if atr_bucket == 'TRANSITION_1_2' (z-score in [1, 2))

Four configurations evaluated:
  baseline     -- all ETD trades
  shock_only   -- Filter A applied
  atr_only     -- Filter B applied
  combined     -- Filter A + Filter B (recommended_keep_flag)

Acceptance criteria (all must pass for PASS verdict):
  1. Sharpe improves >= +0.05 vs baseline (combined)
  2. Max drawdown reduces >= 15% vs baseline (combined)
  3. P(< -1 SD) materially reduces vs baseline
  4. Improvement survives in forward split
  5. Trade count remains economically meaningful

Usage
-----
  python scripts/markov_etd_entry_filters.py \\
      --output results/etd_entry_filters.csv \\
      --memo   results/etd_entry_filters_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Constants ──────────────────────────────────────────────────────────────────

MEDIUM_COST: float = 0.07
USD_CHF: str = 'USD/CHF'

# ATR context
ATR_WINDOW: int = 30          # rolling window days for mean/std

# ATR z-score buckets
BUCKET_LOW: str = 'LOW'                    # z < 0
BUCKET_NORMAL: str = 'NORMAL'             # 0 <= z < 1
BUCKET_TRANSITION: str = 'TRANSITION_1_2' # 1 <= z < 2
BUCKET_SPIKE: str = 'SPIKE_GE_2'          # z >= 2

# Filter thresholds
SHOCK_THRESHOLD: float = -3.0             # Filter A: shock_1h_sd < this
# Filter B: bucket == TRANSITION_1_2

# State classification thresholds
VOL_EXPANDING_MIN: float = 1.20
EFF_TRENDING_MIN: float = 0.60

# Time splits
TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024

# Acceptance criteria
SHARPE_IMPROVE_MIN: float = 0.05
DD_REDUCE_MIN_FRAC: float = 0.15
MIN_TRADE_COUNT: int = 20

# Metals chaos diagnostic (informational only)
METALS_CHAOS_THRESHOLD: float = 20.0


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


# ── Table creation ─────────────────────────────────────────────────────────────

DDL_DAILY_ATR_CONTEXT = """
CREATE TABLE IF NOT EXISTS features.daily_atr_context (
    symbol      VARCHAR(20)              NOT NULL,
    trade_date  DATE                     NOT NULL,
    atr_daily   DOUBLE PRECISION,
    atr_30d_mean DOUBLE PRECISION,
    atr_30d_std  DOUBLE PRECISION,
    atr_zscore  DOUBLE PRECISION,
    atr_bucket  VARCHAR(20),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, trade_date)
);
"""

DDL_ETD_ENTRY_QUALITY_FLAGS = """
CREATE TABLE IF NOT EXISTS features.etd_entry_quality_flags (
    symbol                   VARCHAR(20)              NOT NULL,
    event_hour_ts            TIMESTAMPTZ              NOT NULL,
    breakout_direction       VARCHAR(4)               NOT NULL,
    state                    VARCHAR(40),
    shock_1h_sd              DOUBLE PRECISION,
    efficiency_20            DOUBLE PRECISION,
    atr_zscore               DOUBLE PRECISION,
    atr_bucket               VARCHAR(20),
    fail_shock_flag          BOOLEAN,
    fail_atr_transition_flag BOOLEAN,
    fail_efficiency_flag     BOOLEAN,
    fail_metals_chaos_flag   BOOLEAN,
    recommended_keep_flag    BOOLEAN,
    fail_reason              TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, event_hour_ts, breakout_direction)
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_DAILY_ATR_CONTEXT)
        cur.execute(DDL_ETD_ENTRY_QUALITY_FLAGS)
    conn.commit()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_daily_metrics(symbol: str) -> list[dict]:
    """Load daily_range from features.daily_market_metrics (already = high - low)."""
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                day_ts,
                daily_range AS atr_daily
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
    """Load ETD breakout events for a symbol with all required fields."""
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
                be.max_favorable_excursion_sd30d,
                be.max_adverse_excursion_sd30d,
                be.bars_held,
                be.exit_reason,
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


def load_metal_monthly(symbol: str) -> dict:
    """Load monthly close prices for a metal (XAU/USD or XAG/USD)."""
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                date_trunc('month', mp.date) AS month,
                last(mp.bid_close, mp.date)  AS close_price
            FROM market_data.hourly_prices mp
            JOIN market_data.symbols s ON s.id = mp.symbol_id
            WHERE s.symbol = %s
              AND mp.date >= '2014-01-01'
            GROUP BY 1
            ORDER BY 1
        """, (symbol,))
        result = {r['month']: float(r['close_price']) for r in cur.fetchall()}
    conn.close()
    return result


# ── ATR context computation ────────────────────────────────────────────────────

def compute_atr_context(daily_metrics: list[dict]) -> list[dict]:
    """
    Compute rolling 30-day ATR mean, std, z-score, and bucket for each day.
    Requires ATR_WINDOW days of history before emitting a row.
    """
    result = []
    for i, row in enumerate(daily_metrics):
        if i < ATR_WINDOW - 1:
            result.append(dict(
                trade_date=row['day_ts'],
                atr_daily=float(row['atr_daily']),
                atr_30d_mean=None,
                atr_30d_std=None,
                atr_zscore=None,
                atr_bucket=None,
            ))
            continue
        window = [float(daily_metrics[j]['atr_daily']) for j in range(i - ATR_WINDOW + 1, i + 1)]
        mean = sum(window) / ATR_WINDOW
        if ATR_WINDOW < 2:
            std = 0.0
        else:
            variance = sum((v - mean) ** 2 for v in window) / (ATR_WINDOW - 1)
            std = math.sqrt(variance)
        if std == 0.0:
            zscore = None
            bucket = None
        else:
            zscore = (float(row['atr_daily']) - mean) / std
            bucket = _atr_bucket(zscore)
        result.append(dict(
            trade_date=row['day_ts'],
            atr_daily=float(row['atr_daily']),
            atr_30d_mean=mean,
            atr_30d_std=std,
            atr_zscore=zscore,
            atr_bucket=bucket,
        ))
    return result


def _atr_bucket(zscore: float) -> str:
    if zscore < 0.0:
        return BUCKET_LOW
    if zscore < 1.0:
        return BUCKET_NORMAL
    if zscore < 2.0:
        return BUCKET_TRANSITION
    return BUCKET_SPIKE


# ── State classification ───────────────────────────────────────────────────────

def classify_state(vr: float, ef: float, direction: str) -> str:
    vol = 'EXPANDING' if vr > VOL_EXPANDING_MIN else ('NEUTRAL' if vr >= 0.80 else 'CONTRACTING')
    eff = 'TRENDING' if ef > EFF_TRENDING_MIN else ('MIXED' if ef >= 0.30 else 'CHOPPY')
    return f'{vol}_{eff}_{direction}'


# ── Metals chaos helper ────────────────────────────────────────────────────────

def _trail_12m(prices: dict, months_sorted: list, event_ts) -> Optional[float]:
    m = event_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    m_key = next((gm for gm in reversed(months_sorted) if gm <= m), None)
    if m_key is None:
        return None
    yr_target = m_key.replace(year=m_key.year - 1)
    yr_key = next((gm for gm in reversed(months_sorted) if gm <= yr_target), None)
    if yr_key is None:
        return None
    return (prices[m_key] - prices[yr_key]) / prices[yr_key] * 100.0


# ── TimescaleDB writes ─────────────────────────────────────────────────────────

def upsert_daily_atr_context(
    symbol: str,
    rows: list[dict],
) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            if r['atr_30d_mean'] is None:
                continue
            cur.execute("""
                INSERT INTO features.daily_atr_context
                    (symbol, trade_date, atr_daily, atr_30d_mean, atr_30d_std,
                     atr_zscore, atr_bucket, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (symbol, trade_date)
                DO UPDATE SET
                    atr_daily    = EXCLUDED.atr_daily,
                    atr_30d_mean = EXCLUDED.atr_30d_mean,
                    atr_30d_std  = EXCLUDED.atr_30d_std,
                    atr_zscore   = EXCLUDED.atr_zscore,
                    atr_bucket   = EXCLUDED.atr_bucket,
                    updated_at   = now()
            """, (
                symbol,
                r['trade_date'],
                r['atr_daily'],
                r['atr_30d_mean'],
                r['atr_30d_std'],
                r['atr_zscore'],
                r['atr_bucket'],
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def upsert_etd_quality_flags(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.etd_entry_quality_flags (
                    symbol, event_hour_ts, breakout_direction, state,
                    shock_1h_sd, efficiency_20, atr_zscore, atr_bucket,
                    fail_shock_flag, fail_atr_transition_flag,
                    fail_efficiency_flag, fail_metals_chaos_flag,
                    recommended_keep_flag, fail_reason, updated_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, now()
                )
                ON CONFLICT (symbol, event_hour_ts, breakout_direction)
                DO UPDATE SET
                    state                    = EXCLUDED.state,
                    shock_1h_sd              = EXCLUDED.shock_1h_sd,
                    efficiency_20            = EXCLUDED.efficiency_20,
                    atr_zscore               = EXCLUDED.atr_zscore,
                    atr_bucket               = EXCLUDED.atr_bucket,
                    fail_shock_flag          = EXCLUDED.fail_shock_flag,
                    fail_atr_transition_flag = EXCLUDED.fail_atr_transition_flag,
                    fail_efficiency_flag     = EXCLUDED.fail_efficiency_flag,
                    fail_metals_chaos_flag   = EXCLUDED.fail_metals_chaos_flag,
                    recommended_keep_flag    = EXCLUDED.recommended_keep_flag,
                    fail_reason              = EXCLUDED.fail_reason,
                    updated_at               = now()
            """, (
                r['symbol'],
                r['event_hour_ts'],
                r['breakout_direction'],
                r['state'],
                r['shock_1h_sd'],
                r['efficiency_20'],
                r['atr_zscore'],
                r['atr_bucket'],
                r['fail_shock_flag'],
                r['fail_atr_transition_flag'],
                r['fail_efficiency_flag'],
                r['fail_metals_chaos_flag'],
                r['recommended_keep_flag'],
                r['fail_reason'],
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> Optional[dict]:
    vals = [float(tr['excess']) for tr in trades]
    n = len(vals)
    if n < 2:
        return None
    mn = sum(vals) / n
    sv = sorted(vals)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
    sd = statistics.stdev(vals)
    sharpe = mn / sd if sd > 0 else 0.0
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss else 0.0
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
        n=n,
        mean=mn,
        median=med,
        std=sd,
        sharpe=sharpe,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        payoff=payoff,
        cum_pnl=cum,
        max_dd=max_dd,
        p_lt_neg1=p_lt_neg1,
        p_gt_pos1=p_gt_pos1,
    )


def fmt(val: Optional[float], decimals: int = 3) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{decimals}f}'


# ── Main analysis ──────────────────────────────────────────────────────────────

def run_analysis(
    trades: list[dict],
    config: str,
) -> dict[str, Optional[dict]]:
    """Return per-split stats for a given trade list."""
    result = {}
    for label in ('full', 'train', 'val', 'fwd'):
        if label == 'full':
            subset = trades
        elif label == 'train':
            subset = [tr for tr in trades if tr['year'] < TRAIN_END_YEAR]
        elif label == 'val':
            subset = [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR]
        else:
            subset = [tr for tr in trades if tr['year'] >= VAL_END_YEAR]
        result[label] = compute_stats(subset)
    return result


def build_config_trades(
    all_etd: list[dict],
    config: str,
) -> list[dict]:
    if config == 'baseline':
        return all_etd
    if config == 'shock_only':
        return [tr for tr in all_etd if not tr['fail_shock_flag']]
    if config == 'atr_only':
        return [tr for tr in all_etd if not tr['fail_atr_transition_flag']]
    if config == 'combined':
        return [tr for tr in all_etd if tr['recommended_keep_flag']]
    raise ValueError(f'unknown config: {config}')


# ── Drawdown anatomy ───────────────────────────────────────────────────────────

def drawdown_anatomy(trades: list[dict], label: str) -> dict:
    """Identify left-tail clusters and shock/transition composition."""
    tail = [tr for tr in trades if tr['excess'] < -1.0]
    shock_in_tail = sum(1 for tr in tail if tr['fail_shock_flag'])
    atr_in_tail = sum(1 for tr in tail if tr['fail_atr_transition_flag'])
    mfe_near_zero = sum(1 for tr in tail
                        if tr.get('max_favorable_excursion_sd30d') is not None
                        and float(tr['max_favorable_excursion_sd30d']) < 0.10)
    return dict(
        label=label,
        n_tail=len(tail),
        shock_in_tail=shock_in_tail,
        atr_transition_in_tail=atr_in_tail,
        mfe_near_zero_in_tail=mfe_near_zero,
        pct_shock=shock_in_tail / len(tail) if tail else 0.0,
        pct_atr_trans=atr_in_tail / len(tail) if tail else 0.0,
    )


# ── Output helpers ─────────────────────────────────────────────────────────────

def stats_row(
    config: str,
    split: str,
    s: Optional[dict],
) -> dict:
    if s is None:
        return dict(config=config, split=split,
                    n='', mean='', median='', std='', sharpe='',
                    win_rate='', payoff='', cum_pnl='', max_dd='',
                    p_lt_neg1='', p_gt_pos1='')
    return dict(
        config=config,
        split=split,
        n=s['n'],
        mean=fmt(s['mean']),
        median=fmt(s['median']),
        std=fmt(s['std']),
        sharpe=fmt(s['sharpe']),
        win_rate=fmt(s['win_rate']),
        payoff=fmt(s['payoff']),
        cum_pnl=fmt(s['cum_pnl']),
        max_dd=fmt(s['max_dd']),
        p_lt_neg1=fmt(s['p_lt_neg1']),
        p_gt_pos1=fmt(s['p_gt_pos1']),
    )


# ── Verdict logic ──────────────────────────────────────────────────────────────

def evaluate_verdict(
    baseline_full: Optional[dict],
    combined_full: Optional[dict],
    combined_fwd: Optional[dict],
    shock_full: Optional[dict],
    atr_full: Optional[dict],
    baseline_n: int,
    combined_n: int,
) -> tuple[str, list[str]]:
    checks = []
    passed = 0

    def chk(label: str, ok: bool) -> None:
        nonlocal passed
        checks.append(f'  {"PASS" if ok else "FAIL"}  {label}')
        if ok:
            passed += 1

    if baseline_full and combined_full:
        sharpe_delta = combined_full['sharpe'] - baseline_full['sharpe']
        dd_delta_frac = (baseline_full['max_dd'] - combined_full['max_dd']) / baseline_full['max_dd'] \
            if baseline_full['max_dd'] > 0 else 0.0
        tail_improved = combined_full['p_lt_neg1'] < baseline_full['p_lt_neg1']
        chk(f'Sharpe +{SHARPE_IMPROVE_MIN} (got {sharpe_delta:+.3f})', sharpe_delta >= SHARPE_IMPROVE_MIN)
        chk(f'Max DD -15% (got {dd_delta_frac:+.1%})', dd_delta_frac >= DD_REDUCE_MIN_FRAC)
        chk(f'P(<-1) improves (base={baseline_full["p_lt_neg1"]:.3f} comb={combined_full["p_lt_neg1"]:.3f})', tail_improved)
    else:
        for lbl in ('Sharpe +0.05', 'Max DD -15%', 'P(<-1) improves'):
            chk(lbl, False)

    if combined_fwd and baseline_full:
        fwd_better = combined_fwd['sharpe'] >= (baseline_full['sharpe'] - 0.05)
        chk(f'Fwd Sharpe meaningful (got {combined_fwd["sharpe"]:.3f})', fwd_better)
    else:
        chk('Fwd Sharpe meaningful', False)

    trade_ok = combined_n >= MIN_TRADE_COUNT
    chk(f'Trade count >= {MIN_TRADE_COUNT} (got {combined_n})', trade_ok)

    verdict = 'PASS' if passed >= 4 else 'FAIL'
    # Partial verdict annotation
    if passed == 3:
        verdict = 'MARGINAL'
    return verdict, checks


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/etd_entry_filters.csv')
    parser.add_argument('--memo',   default='results/etd_entry_filters_memo.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    print('[1/6] Creating tables...')
    conn = connect()
    create_tables(conn)
    conn.close()

    print('[2/6] Building daily ATR context for USD/CHF...')
    raw_daily = load_daily_metrics(USD_CHF)
    atr_context = compute_atr_context(raw_daily)
    written = upsert_daily_atr_context(USD_CHF, atr_context)
    print(f'      Upserted {written} rows into features.daily_atr_context')

    # Build lookup: date -> atr context row
    atr_lookup: dict[date, dict] = {
        r['trade_date']: r for r in atr_context if r['atr_bucket'] is not None
    }

    print('[3/6] Loading ETD breakout events...')
    raw_trades = load_etd_trades(USD_CHF)

    # Load metals data for diagnostics
    print('      Loading metals monthly prices for diagnostic flags...')
    gold_prices = load_metal_monthly('XAU/USD')
    silver_prices = load_metal_monthly('XAG/USD')
    gold_sorted = sorted(gold_prices)
    silver_sorted = sorted(silver_prices)

    print('[4/6] Building ETD entry quality flags...')
    flag_rows = []
    etd_trades: list[dict] = []

    for tr in raw_trades:
        vr = float(tr['vol_ratio_20_100'])
        ef = float(tr['efficiency_20'])
        direction = str(tr['breakout_direction'])
        state = classify_state(vr, ef, direction)

        # ETD only: EXPANDING_TRENDING_DOWN
        if state != 'EXPANDING_TRENDING_DOWN':
            continue

        ts = tr['event_hour_ts']
        trade_date = ts.date() if hasattr(ts, 'date') else ts
        atr_ctx = atr_lookup.get(trade_date)

        shock_val = float(tr['shock_1h_sd'])
        eff_val = float(tr['efficiency_20'])
        atr_zscore = atr_ctx['atr_zscore'] if atr_ctx else None
        atr_bucket = atr_ctx['atr_bucket'] if atr_ctx else None

        fail_shock = shock_val < SHOCK_THRESHOLD
        fail_atr = atr_bucket == BUCKET_TRANSITION
        fail_eff = eff_val > 0.90   # diagnostic only
        g = _trail_12m(gold_prices, gold_sorted, ts)
        s = _trail_12m(silver_prices, silver_sorted, ts)
        fail_metals = (g is not None and s is not None
                       and g > METALS_CHAOS_THRESHOLD
                       and s > METALS_CHAOS_THRESHOLD)

        keep = not (fail_shock or fail_atr)
        reasons = []
        if fail_shock:
            reasons.append('shock')
        if fail_atr:
            reasons.append('atr_transition')

        flag_rows.append(dict(
            symbol=USD_CHF,
            event_hour_ts=ts,
            breakout_direction=direction,
            state=state,
            shock_1h_sd=shock_val,
            efficiency_20=eff_val,
            atr_zscore=atr_zscore,
            atr_bucket=atr_bucket,
            fail_shock_flag=fail_shock,
            fail_atr_transition_flag=fail_atr,
            fail_efficiency_flag=fail_eff,
            fail_metals_chaos_flag=fail_metals,
            recommended_keep_flag=keep,
            fail_reason=','.join(reasons) if reasons else None,
        ))

        # Prepare trade dict for analysis (exclude trades with no ATR context)
        excess = float(tr['realized_return_sd30d']) - MEDIUM_COST
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
        etd_trades.append(dict(
            **tr,
            state=state,
            atr_zscore=atr_zscore,
            atr_bucket=atr_bucket,
            fail_shock_flag=fail_shock,
            fail_atr_transition_flag=fail_atr,
            fail_efficiency_flag=fail_eff,
            fail_metals_chaos_flag=fail_metals,
            recommended_keep_flag=keep,
            excess=excess,
            year=year,
        ))

    written_flags = upsert_etd_quality_flags(flag_rows)
    print(f'      Upserted {written_flags} rows into features.etd_entry_quality_flags')
    print(f'      Total ETD trades: {len(etd_trades)}')

    # ── Bucket distribution ────────────────────────────────────────────────────
    bucket_counts: dict[str, int] = defaultdict(int)
    for tr in etd_trades:
        if tr['atr_bucket']:
            bucket_counts[tr['atr_bucket']] += 1

    n_shock = sum(1 for tr in etd_trades if tr['fail_shock_flag'])
    n_atr_trans = sum(1 for tr in etd_trades if tr['fail_atr_transition_flag'])
    n_combined = sum(1 for tr in etd_trades if not tr['recommended_keep_flag'])

    print(f'      Shock filter removes:       {n_shock} trades')
    print(f'      ATR-transition filter removes: {n_atr_trans} trades')
    print(f'      Combined filter removes:    {n_combined} trades')

    print('[5/6] Running ETD analysis across 4 configurations...')
    configs = ['baseline', 'shock_only', 'atr_only', 'combined']
    all_results: dict[str, dict] = {}
    for cfg in configs:
        subset = build_config_trades(etd_trades, cfg)
        all_results[cfg] = run_analysis(subset, cfg)

    print('[6/6] Computing drawdown anatomy...')
    anatomy_rows = []
    for cfg in configs:
        subset = build_config_trades(etd_trades, cfg)
        anatomy_rows.append(drawdown_anatomy(subset, cfg))

    # ── Verdict ────────────────────────────────────────────────────────────────
    base_full = all_results['baseline']['full']
    comb_full = all_results['combined']['full']
    comb_fwd  = all_results['combined']['fwd']
    shock_full = all_results['shock_only']['full']
    atr_full   = all_results['atr_only']['full']
    combined_n = len(build_config_trades(etd_trades, 'combined'))

    verdict, checks = evaluate_verdict(
        base_full, comb_full, comb_fwd,
        shock_full, atr_full,
        len(etd_trades), combined_n,
    )

    # ── CSV output ─────────────────────────────────────────────────────────────
    csv_rows = []
    for cfg in configs:
        for split in ('full', 'train', 'val', 'fwd'):
            csv_rows.append(stats_row(cfg, split, all_results[cfg][split]))

    fieldnames = ['config', 'split', 'n', 'mean', 'median', 'std',
                  'sharpe', 'win_rate', 'payoff', 'cum_pnl', 'max_dd',
                  'p_lt_neg1', 'p_gt_pos1']
    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'      Results written to {args.output}')

    # ── Memo output ────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append('=' * 72)
    lines.append('ETD ENTRY FILTER ANALYSIS — Issue #99')
    lines.append('=' * 72)
    lines.append('')
    lines.append('FILTER DEFINITIONS')
    lines.append(f'  Filter A (shock):          shock_1h_sd < {SHOCK_THRESHOLD}')
    lines.append(f'  Filter B (ATR-transition): atr_bucket == TRANSITION_1_2 (z in [1,2))')
    lines.append(f'  Combined keep:             NOT (A OR B)')
    lines.append('')
    lines.append('FILTER COVERAGE')
    lines.append(f'  Total ETD trades:          {len(etd_trades)}')
    lines.append(f'  Removed by shock:          {n_shock} ({n_shock/len(etd_trades):.1%})')
    lines.append(f'  Removed by ATR-transition: {n_atr_trans} ({n_atr_trans/len(etd_trades):.1%})')
    lines.append(f'  Removed by combined:       {n_combined} ({n_combined/len(etd_trades):.1%})')
    lines.append(f'  Remaining (combined):      {combined_n} ({combined_n/len(etd_trades):.1%})')
    lines.append('')
    lines.append('ATR BUCKET DISTRIBUTION (ETD trades)')
    for bk in (BUCKET_LOW, BUCKET_NORMAL, BUCKET_TRANSITION, BUCKET_SPIKE):
        cnt = bucket_counts.get(bk, 0)
        pct = cnt / len(etd_trades) * 100 if etd_trades else 0.0
        lines.append(f'  {bk:<22} {cnt:>5}  ({pct:.1f}%)')
    lines.append('')

    lines.append('PERFORMANCE TABLE (full period)')
    lines.append(f'  {"Config":<15} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"P<-1":>8} {"CumPnL":>8}')
    lines.append(f'  {"-"*15} {"-"*5} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for cfg in configs:
        s = all_results[cfg]['full']
        if s:
            lines.append(f'  {cfg:<15} {s["n"]:>5} {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f} {s["p_lt_neg1"]:>8.3f} {s["cum_pnl"]:>8.3f}')
    lines.append('')

    lines.append('TIME-SPLIT TABLE (Sharpe)')
    lines.append(f'  {"Config":<15} {"train":>8} {"val":>8} {"fwd":>8}')
    lines.append(f'  {"-"*15} {"-"*8} {"-"*8} {"-"*8}')
    for cfg in configs:
        parts = []
        for sp in ('train', 'val', 'fwd'):
            s = all_results[cfg][sp]
            parts.append(f'{s["sharpe"]:>8.3f}' if s else f'{"N/A":>8}')
        lines.append(f'  {cfg:<15} {"".join(parts)}')
    lines.append('')

    lines.append('DRAWDOWN ANATOMY (left tail = excess < -1 SD)')
    lines.append(f'  {"Config":<15} {"tail_n":>7} {"shock%":>8} {"atr_tr%":>8} {"mfe~0":>8}')
    lines.append(f'  {"-"*15} {"-"*7} {"-"*8} {"-"*8} {"-"*8}')
    for row in anatomy_rows:
        lines.append(
            f'  {row["label"]:<15} {row["n_tail"]:>7}'
            f' {row["pct_shock"]:>8.1%} {row["pct_atr_trans"]:>8.1%} {row["mfe_near_zero_in_tail"]:>8}'
        )
    lines.append('')

    lines.append('DIAGNOSTIC FILTERS (informational, not production candidates)')
    n_eff = sum(1 for tr in etd_trades if tr['fail_efficiency_flag'])
    n_metals = sum(1 for tr in etd_trades if tr['fail_metals_chaos_flag'])
    lines.append(f'  Efficiency > 0.90 flag:    {n_eff} trades ({n_eff/len(etd_trades):.1%})')
    lines.append(f'  Metals chaos flag:         {n_metals} trades ({n_metals/len(etd_trades):.1%})')
    lines.append('')

    lines.append('FILTER CONTRIBUTION (which filter is doing the work?)')
    if base_full and shock_full and atr_full and comb_full:
        shock_sharpe_delta = shock_full['sharpe'] - base_full['sharpe']
        atr_sharpe_delta = atr_full['sharpe'] - base_full['sharpe']
        comb_sharpe_delta = comb_full['sharpe'] - base_full['sharpe']
        lines.append(f'  Shock-only Sharpe delta:       {shock_sharpe_delta:+.3f}')
        lines.append(f'  ATR-transition Sharpe delta:   {atr_sharpe_delta:+.3f}')
        lines.append(f'  Combined Sharpe delta:         {comb_sharpe_delta:+.3f}')
        dominant = 'shock' if abs(shock_sharpe_delta) >= abs(atr_sharpe_delta) else 'atr_transition'
        lines.append(f'  Dominant filter:               {dominant}')
    lines.append('')

    lines.append('ACCEPTANCE CRITERIA')
    for chk in checks:
        lines.append(chk)
    lines.append('')
    lines.append(f'VERDICT: {verdict}')
    lines.append('')

    # Recommendation
    lines.append('RECOMMENDATION')
    if verdict == 'PASS':
        if base_full and shock_full and atr_full and comb_full:
            s_d = shock_full['sharpe'] - base_full['sharpe']
            a_d = atr_full['sharpe'] - base_full['sharpe']
            if abs(s_d - a_d) < 0.03:
                lines.append('  Keep combined ETD filter (both filters contribute meaningfully).')
                lines.append('  Combined provides best coverage of identified failure modes.')
            elif s_d > a_d:
                lines.append('  Shock filter (A) drives the improvement.')
                lines.append('  Consider keeping shock-only as minimal production candidate.')
                lines.append('  ATR-transition is additive but secondary.')
            else:
                lines.append('  ATR-transition filter (B) drives the improvement.')
                lines.append('  Consider keeping ATR-only as minimal production candidate.')
                lines.append('  Shock filter is additive but secondary.')
    elif verdict == 'MARGINAL':
        lines.append('  Marginal improvement — do not promote ETD filters to production.')
        lines.append('  Review whether improvement is driven by a single historical cluster.')
    else:
        lines.append('  Reject ETD entry filtering.')
        lines.append('  Neither shock nor ATR-transition filter achieves acceptance criteria.')
        lines.append('  Leave ETD unfiltered.')
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

# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_entry_filters_nonfx.py
Issue #99 (non-FX extension): ETD Entry Filters — Indices, Commodities, Crypto, Bonds

Applies the same two minimal ETD filters tested on USD/CHF to all non-FX symbols:
  Filter A: shock_1h_sd < -3.0
  Filter B: atr_bucket == TRANSITION_1_2  (ATR z-score in [1, 2))
  Combined: recommended_keep_flag = NOT (A OR B)

Reports:
  - per-symbol full-period stats across 4 configurations
  - pooled non-FX stats across 4 configurations + train/val/fwd splits
  - filter coverage breakdown per symbol
  - drawdown anatomy pooled

Same thresholds as USD/CHF. No re-optimisation.

Usage
-----
  python scripts/markov_etd_entry_filters_nonfx.py \\
      --output results/etd_entry_filters_nonfx.csv \\
      --memo   results/etd_entry_filters_nonfx_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Non-FX symbol list ─────────────────────────────────────────────────────────

NON_FX_SYMBOLS: list[str] = [
    'AUS200', 'BTC/USD', 'Bund', 'CHN50', 'CORNF', 'Copper',
    'ESP35', 'ETH/USD', 'EUSTX50', 'FRA40', 'GER30', 'HKG33',
    'JPN225', 'NAS100', 'NGAS', 'SOYF', 'SPX500', 'UK100',
    'UKOil', 'US2000', 'US30', 'USOil', 'WHEATF',
    'XAG/USD', 'XAU/USD', 'XRP/USD',
]

# ── Constants (identical to USD/CHF test — no re-optimisation) ─────────────────

MEDIUM_COST: float = 0.07
ATR_WINDOW: int = 30

BUCKET_LOW: str = 'LOW'
BUCKET_NORMAL: str = 'NORMAL'
BUCKET_TRANSITION: str = 'TRANSITION_1_2'
BUCKET_SPIKE: str = 'SPIKE_GE_2'

SHOCK_THRESHOLD: float = -3.0
VOL_EXPANDING_MIN: float = 1.20
EFF_TRENDING_MIN: float = 0.60

TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024

SHARPE_IMPROVE_MIN: float = 0.05
DD_REDUCE_MIN_FRAC: float = 0.15
MIN_TRADE_COUNT: int = 10


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


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


# ── ATR context ────────────────────────────────────────────────────────────────

def compute_atr_context(daily_metrics: list[dict]) -> list[dict]:
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


# ── TimescaleDB upsert ─────────────────────────────────────────────────────────

def upsert_daily_atr_context(symbol: str, rows: list[dict]) -> int:
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
                symbol, r['trade_date'], r['atr_daily'],
                r['atr_30d_mean'], r['atr_30d_std'],
                r['atr_zscore'], r['atr_bucket'],
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def upsert_etd_quality_flags(rows: list[dict]) -> int:
    if not rows:
        return 0
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
                    %s, %s, %s, %s,
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
                r['symbol'], r['event_hour_ts'], r['breakout_direction'], r['state'],
                r['shock_1h_sd'], r['efficiency_20'], r['atr_zscore'], r['atr_bucket'],
                r['fail_shock_flag'], r['fail_atr_transition_flag'],
                r['fail_efficiency_flag'], False,
                r['recommended_keep_flag'], r['fail_reason'],
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
        n=n, mean=mn, median=med, std=sd, sharpe=sharpe,
        win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        payoff=payoff, cum_pnl=cum, max_dd=max_dd,
        p_lt_neg1=p_lt_neg1, p_gt_pos1=p_gt_pos1,
    )


def fmt(val: Optional[float], decimals: int = 3) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{decimals}f}'


# ── Config filtering ───────────────────────────────────────────────────────────

def build_config_trades(trades: list[dict], config: str) -> list[dict]:
    if config == 'baseline':
        return trades
    if config == 'shock_only':
        return [tr for tr in trades if not tr['fail_shock_flag']]
    if config == 'atr_only':
        return [tr for tr in trades if not tr['fail_atr_transition_flag']]
    if config == 'combined':
        return [tr for tr in trades if tr['recommended_keep_flag']]
    raise ValueError(f'unknown config: {config}')


# ── Per-symbol processing ──────────────────────────────────────────────────────

def process_symbol(symbol: str) -> dict:
    """
    Returns a dict with:
      etd_trades:    list of annotated ETD trade dicts
      flag_rows:     list of flag rows for DB upsert
      atr_written:   int (rows upserted to daily_atr_context)
      n_raw:         int (total breakout events loaded)
    """
    raw_daily = load_daily_metrics(symbol)
    atr_context = compute_atr_context(raw_daily)
    atr_written = upsert_daily_atr_context(symbol, atr_context)
    atr_lookup = {
        r['trade_date']: r for r in atr_context if r['atr_bucket'] is not None
    }

    raw_trades = load_etd_trades(symbol)
    flag_rows = []
    etd_trades = []

    for tr in raw_trades:
        vr = float(tr['vol_ratio_20_100'])
        ef = float(tr['efficiency_20'])
        direction = str(tr['breakout_direction'])
        state = classify_state(vr, ef, direction)

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
        fail_eff = eff_val > 0.90
        keep = not (fail_shock or fail_atr)
        reasons = []
        if fail_shock:
            reasons.append('shock')
        if fail_atr:
            reasons.append('atr_transition')

        flag_rows.append(dict(
            symbol=symbol,
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
            recommended_keep_flag=keep,
            fail_reason=','.join(reasons) if reasons else None,
        ))

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
            recommended_keep_flag=keep,
            excess=excess,
            year=year,
        ))

    return dict(
        etd_trades=etd_trades,
        flag_rows=flag_rows,
        atr_written=atr_written,
        n_raw=len(raw_trades),
    )


# ── Output helpers ─────────────────────────────────────────────────────────────

def stats_row(symbol: str, config: str, split: str, s: Optional[dict]) -> dict:
    if s is None:
        return dict(symbol=symbol, config=config, split=split,
                    n='', mean='', median='', std='', sharpe='',
                    win_rate='', payoff='', cum_pnl='', max_dd='',
                    p_lt_neg1='', p_gt_pos1='')
    return dict(
        symbol=symbol, config=config, split=split,
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


def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/etd_entry_filters_nonfx.csv')
    parser.add_argument('--memo',   default='results/etd_entry_filters_nonfx_memo.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    configs = ['baseline', 'shock_only', 'atr_only', 'combined']
    per_symbol: dict[str, list[dict]] = {}        # symbol -> etd_trades
    coverage: dict[str, dict] = {}

    print(f'Processing {len(NON_FX_SYMBOLS)} non-FX symbols...')
    for symbol in NON_FX_SYMBOLS:
        result = process_symbol(symbol)
        flag_rows = result['flag_rows']
        etd_trades = result['etd_trades']
        upsert_etd_quality_flags(flag_rows)

        n_shock = sum(1 for tr in etd_trades if tr['fail_shock_flag'])
        n_atr = sum(1 for tr in etd_trades if tr['fail_atr_transition_flag'])
        n_combined = sum(1 for tr in etd_trades if not tr['recommended_keep_flag'])
        per_symbol[symbol] = etd_trades
        coverage[symbol] = dict(
            n_total=len(etd_trades),
            n_shock=n_shock,
            n_atr=n_atr,
            n_combined=n_combined,
            n_keep=len(etd_trades) - n_combined,
        )
        tag = f'  {symbol:<12} etd={len(etd_trades):>4}  shock={n_shock}  atr={n_atr}  combined={n_combined}'
        print(tag)

    # ── Pool all non-FX trades ─────────────────────────────────────────────────
    all_nonfx: list[dict] = [tr for trades in per_symbol.values() for tr in trades]
    print(f'\nTotal non-FX ETD trades: {len(all_nonfx)}')

    # ── CSV rows ───────────────────────────────────────────────────────────────
    csv_rows = []
    fieldnames = ['symbol', 'config', 'split', 'n', 'mean', 'median', 'std',
                  'sharpe', 'win_rate', 'payoff', 'cum_pnl', 'max_dd',
                  'p_lt_neg1', 'p_gt_pos1']

    # Per-symbol full-period rows
    for symbol in NON_FX_SYMBOLS:
        trades = per_symbol[symbol]
        for cfg in configs:
            subset = build_config_trades(trades, cfg)
            s = compute_stats(subset)
            csv_rows.append(stats_row(symbol, cfg, 'full', s))

    # Pooled rows (all splits)
    for cfg in configs:
        subset = build_config_trades(all_nonfx, cfg)
        splits = split_trades(subset)
        for sp, sp_trades in splits.items():
            s = compute_stats(sp_trades)
            csv_rows.append(stats_row('_POOLED', cfg, sp, s))

    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # ── Memo ───────────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append('=' * 72)
    lines.append('ETD ENTRY FILTERS — NON-FX SYMBOLS  (Issue #99 extension)')
    lines.append('Same thresholds as USD/CHF. No re-optimisation.')
    lines.append('=' * 72)
    lines.append('')
    lines.append('FILTER DEFINITIONS')
    lines.append(f'  Filter A (shock):          shock_1h_sd < {SHOCK_THRESHOLD}')
    lines.append(f'  Filter B (ATR-transition): atr_bucket == TRANSITION_1_2 (z in [1,2))')
    lines.append(f'  Combined keep:             NOT (A OR B)')
    lines.append('')

    # Coverage table
    total_etd = sum(v['n_total'] for v in coverage.values())
    total_shock = sum(v['n_shock'] for v in coverage.values())
    total_atr = sum(v['n_atr'] for v in coverage.values())
    total_comb = sum(v['n_combined'] for v in coverage.values())
    total_keep = sum(v['n_keep'] for v in coverage.values())

    lines.append('FILTER COVERAGE PER SYMBOL')
    lines.append(f'  {"Symbol":<12} {"etd_n":>6} {"shock":>6} {"atr_tr":>7} {"comb":>6} {"keep":>6} {"keep%":>6}')
    lines.append(f'  {"-"*12} {"-"*6} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*6}')
    for sym in NON_FX_SYMBOLS:
        c = coverage[sym]
        pct = c['n_keep'] / c['n_total'] * 100 if c['n_total'] else 0.0
        lines.append(
            f'  {sym:<12} {c["n_total"]:>6} {c["n_shock"]:>6} {c["n_atr"]:>7}'
            f' {c["n_combined"]:>6} {c["n_keep"]:>6} {pct:>5.0f}%'
        )
    lines.append(f'  {"TOTAL":<12} {total_etd:>6} {total_shock:>6} {total_atr:>7}'
                 f' {total_comb:>6} {total_keep:>6} {total_keep/total_etd*100:>5.0f}%')
    lines.append('')

    # Per-symbol performance table (full period)
    lines.append('PER-SYMBOL PERFORMANCE (full period, Sharpe)')
    lines.append(f'  {"Symbol":<12} {"base_n":>6} {"base_sh":>8} {"shock_sh":>9} {"atr_sh":>8} {"comb_sh":>8} {"comb_n":>7} {"dd_red%":>8}')
    lines.append(f'  {"-"*12} {"-"*6} {"-"*8} {"-"*9} {"-"*8} {"-"*8} {"-"*7} {"-"*8}')

    for sym in NON_FX_SYMBOLS:
        trades = per_symbol[sym]
        stats: dict[str, Optional[dict]] = {}
        for cfg in configs:
            stats[cfg] = compute_stats(build_config_trades(trades, cfg))

        base = stats['baseline']
        comb = stats['combined']
        shock = stats['shock_only']
        atr = stats['atr_only']
        base_sh = fmt(base['sharpe'] if base else None)
        shock_sh = fmt(shock['sharpe'] if shock else None)
        atr_sh = fmt(atr['sharpe'] if atr else None)
        comb_sh = fmt(comb['sharpe'] if comb else None)
        base_n = base['n'] if base else 0
        comb_n = comb['n'] if comb else 0

        if base and comb and base['max_dd'] > 0:
            dd_red = (base['max_dd'] - comb['max_dd']) / base['max_dd'] * 100
            dd_str = f'{dd_red:>+7.1f}%'
        else:
            dd_str = '    N/A'

        lines.append(
            f'  {sym:<12} {base_n:>6} {base_sh:>8} {shock_sh:>9}'
            f' {atr_sh:>8} {comb_sh:>8} {comb_n:>7} {dd_str:>8}'
        )
    lines.append('')

    # Pooled performance (all splits)
    lines.append('POOLED NON-FX PERFORMANCE (all symbols combined)')
    lines.append(f'  {"Config":<12} {"split":>6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"P<-1":>7} {"WinR":>7} {"Payoff":>8} {"CumPnL":>9}')
    lines.append(f'  {"-"*12} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*8} {"-"*9}')
    for cfg in configs:
        subset = build_config_trades(all_nonfx, cfg)
        for sp in ('full', 'train', 'val', 'fwd'):
            if sp == 'full':
                sp_trades = subset
            elif sp == 'train':
                sp_trades = [tr for tr in subset if tr['year'] < TRAIN_END_YEAR]
            elif sp == 'val':
                sp_trades = [tr for tr in subset if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR]
            else:
                sp_trades = [tr for tr in subset if tr['year'] >= VAL_END_YEAR]
            s = compute_stats(sp_trades)
            if s:
                lines.append(
                    f'  {cfg:<12} {sp:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["p_lt_neg1"]:>7.3f} {s["win_rate"]:>7.3f}'
                    f' {s["payoff"]:>8.3f} {s["cum_pnl"]:>9.3f}'
                )
            else:
                lines.append(f'  {cfg:<12} {sp:>6}   N/A')
    lines.append('')

    # Pooled verdict
    base_full = compute_stats(build_config_trades(all_nonfx, 'baseline'))
    comb_full = compute_stats(build_config_trades(all_nonfx, 'combined'))
    comb_fwd = compute_stats([tr for tr in build_config_trades(all_nonfx, 'combined') if tr['year'] >= VAL_END_YEAR])
    base_fwd = compute_stats([tr for tr in all_nonfx if tr['year'] >= VAL_END_YEAR])

    lines.append('POOLED ACCEPTANCE CRITERIA')
    checks = []
    passed = 0

    def chk(label: str, ok: bool) -> None:
        nonlocal passed
        checks.append(f'  {"PASS" if ok else "FAIL"}  {label}')
        if ok:
            passed += 1

    if base_full and comb_full:
        sharpe_d = comb_full['sharpe'] - base_full['sharpe']
        dd_d = (base_full['max_dd'] - comb_full['max_dd']) / base_full['max_dd'] if base_full['max_dd'] > 0 else 0.0
        chk(f'Sharpe +{SHARPE_IMPROVE_MIN} (got {sharpe_d:+.3f})', sharpe_d >= SHARPE_IMPROVE_MIN)
        chk(f'Max DD -15% (got {dd_d:+.1%})', dd_d >= DD_REDUCE_MIN_FRAC)
        chk(f'P(<-1) improves ({base_full["p_lt_neg1"]:.3f} -> {comb_full["p_lt_neg1"]:.3f})',
            comb_full['p_lt_neg1'] < base_full['p_lt_neg1'])
    else:
        for l in ('Sharpe', 'DD', 'P<-1'):
            chk(l, False)

    if comb_fwd and base_fwd:
        chk(f'Fwd Sharpe meaningful (comb={comb_fwd["sharpe"]:.3f} base={base_fwd["sharpe"]:.3f})',
            comb_fwd['sharpe'] >= base_fwd['sharpe'] - 0.05)
    else:
        chk('Fwd Sharpe meaningful', False)

    chk(f'Keep count >= {MIN_TRADE_COUNT} per symbol',
        all(coverage[s]['n_keep'] >= MIN_TRADE_COUNT for s in NON_FX_SYMBOLS if coverage[s]['n_total'] >= MIN_TRADE_COUNT))

    for c in checks:
        lines.append(c)

    verdict = 'PASS' if passed >= 4 else ('MARGINAL' if passed == 3 else 'FAIL')
    lines.append('')
    lines.append(f'POOLED VERDICT: {verdict}  ({passed}/5 criteria)')
    lines.append('')

    # Filter contribution
    if base_full and comb_full:
        shock_full = compute_stats(build_config_trades(all_nonfx, 'shock_only'))
        atr_full = compute_stats(build_config_trades(all_nonfx, 'atr_only'))
        lines.append('FILTER CONTRIBUTION (pooled)')
        if shock_full:
            lines.append(f'  Shock-only Sharpe delta:       {shock_full["sharpe"] - base_full["sharpe"]:+.3f}')
        if atr_full:
            lines.append(f'  ATR-transition Sharpe delta:   {atr_full["sharpe"] - base_full["sharpe"]:+.3f}')
        lines.append(f'  Combined Sharpe delta:         {comb_full["sharpe"] - base_full["sharpe"]:+.3f}')
    lines.append('')

    # Comparison with USD/CHF
    lines.append('COMPARISON vs USD/CHF')
    lines.append('  USD/CHF combined Sharpe delta: +0.126  (full period)')
    if base_full and comb_full:
        nonfx_d = comb_full['sharpe'] - base_full['sharpe']
        lines.append(f'  Non-FX combined Sharpe delta:  {nonfx_d:+.3f}  (pooled full period)')
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

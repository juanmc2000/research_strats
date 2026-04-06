# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_pre_sizing.py
Issue #103: ETD Sleeve Pre-Sizing Research — Fixed-Weight and Capped-Weight Allocation

Tests the frozen ETD sleeve (USD/CHF + GBP/USD, ATR-transition filter only)
under simple, reproducible, low-complexity allocation rules.

This script does NOT:
  - optimize weights
  - introduce new filters
  - alter thresholds
  - apply shock filter generically
  - include Kelly, pyramiding, or dynamic sizing

Frozen strategy object (Issue #101/#102):
  state  = EXPANDING_TRENDING_DOWN
  filter = ATR-transition filter only (atr_bucket == TRANSITION_1_2 excluded)
  symbols = USD/CHF, GBP/USD

Weighting schemes (pre-declared, not optimized):
  equal_weight       : 50% USD/CHF, 50% GBP/USD
  capped_conviction  : 60% USD/CHF, 40% GBP/USD  (anchor symbol gets mild overweight)
  conservative_cap   : max 50% per symbol (same as equal_weight; included for clarity)

Pre-declared decision rules:
  - if capped_conviction improves full-period metrics but worsens forward materially: reject it
  - if equal_weight and capped_conviction are similar: default to equal_weight
  - if sleeve is dominated (>70%) by one symbol's PnL contribution: flag, do not progress
  - if forward Sharpe < 0 under all schemes: stop, do not proceed to advanced sizing

Usage
-----
  python scripts/markov_etd_pre_sizing.py \\
      --output results/etd_pre_sizing.csv \\
      --memo   results/etd_pre_sizing_memo.txt
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

# ── Frozen strategy object ────────────────────────────────────────────────────

SLEEVE_SYMBOLS: list[str] = ['USD/CHF', 'GBP/USD']
SLEEVE_NAME: str = 'etd_atr_filtered'

# Pre-declared weighting schemes: symbol -> weight
WEIGHTING_SCHEMES: dict[str, dict[str, float]] = {
    'equal_weight':      {'USD/CHF': 0.50, 'GBP/USD': 0.50},
    'capped_conviction': {'USD/CHF': 0.60, 'GBP/USD': 0.40},
    'conservative_cap':  {'USD/CHF': 0.50, 'GBP/USD': 0.50},
}

# Pre-declared decision rule thresholds
CONCENTRATION_FLAG_THRESHOLD: float = 0.70   # PnL concentration > 70% from one symbol
CAPPED_FWD_DEGRADE_THRESHOLD: float = -0.05  # capped hurts forward vs equal by more than this

# ── Constants ─────────────────────────────────────────────────────────────────

MEDIUM_COST: float = 0.07
ATR_WINDOW: int = 30
BUCKET_TRANSITION: str = 'TRANSITION_1_2'
VOL_EXPANDING_MIN: float = 1.20
EFF_TRENDING_MIN: float = 0.60
TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024
SPLITS: list[str] = ['full', 'train', 'val', 'fwd']


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


# ── DDL ────────────────────────────────────────────────────────────────────────

DDL_PRE_SIZING = """
CREATE TABLE IF NOT EXISTS features.etd_sleeve_pre_sizing_results (
    sleeve_name          VARCHAR(30)       NOT NULL,
    weighting_scheme     VARCHAR(24)       NOT NULL,
    split_name           VARCHAR(12)       NOT NULL,
    n_trades             INTEGER,
    mean_return          DOUBLE PRECISION,
    sharpe               DOUBLE PRECISION,
    max_drawdown         DOUBLE PRECISION,
    win_rate             DOUBLE PRECISION,
    payoff               DOUBLE PRECISION,
    p_lt_neg1            DOUBLE PRECISION,
    p_gt_pos1            DOUBLE PRECISION,
    symbol_concentration DOUBLE PRECISION,
    created_at           TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_sleeve_pre_sizing_results
        PRIMARY KEY (sleeve_name, weighting_scheme, split_name)
);
"""

DDL_CONTRIBUTIONS = """
CREATE TABLE IF NOT EXISTS features.etd_sleeve_symbol_contributions (
    sleeve_name           VARCHAR(30)       NOT NULL,
    weighting_scheme      VARCHAR(24)       NOT NULL,
    symbol                VARCHAR(20)       NOT NULL,
    split_name            VARCHAR(12)       NOT NULL,
    n_trades              INTEGER,
    pnl_contribution      DOUBLE PRECISION,
    variance_contribution DOUBLE PRECISION,
    drawdown_contribution DOUBLE PRECISION,
    created_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_sleeve_symbol_contributions
        PRIMARY KEY (sleeve_name, weighting_scheme, symbol, split_name)
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_PRE_SIZING)
        cur.execute(DDL_CONTRIBUTIONS)
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
        if zscore < 0.0:
            bucket = 'LOW'
        elif zscore < 1.0:
            bucket = 'NORMAL'
        elif zscore < 2.0:
            bucket = BUCKET_TRANSITION
        else:
            bucket = 'SPIKE_GE_2'
        result[row['day_ts']] = bucket
    return result


# ── Trade building ─────────────────────────────────────────────────────────────

def build_filtered_trades(symbol: str) -> list[dict]:
    """Return ATR-transition-filtered EXPANDING_TRENDING_DOWN trades."""
    raw_daily = load_daily_metrics(symbol)
    atr_lookup = compute_atr_lookup(raw_daily)
    raw_trades = load_etd_trades(symbol)

    trades = []
    for tr in raw_trades:
        vr = float(tr['vol_ratio_20_100'])
        ef = float(tr['efficiency_20'])
        direction = str(tr['breakout_direction'])

        vol = 'EXPANDING' if vr > VOL_EXPANDING_MIN else ('NEUTRAL' if vr >= 0.80 else 'CONTRACTING')
        eff = 'TRENDING' if ef > EFF_TRENDING_MIN else ('MIXED' if ef >= 0.30 else 'CHOPPY')
        state = f'{vol}_{eff}_{direction}'
        if state != 'EXPANDING_TRENDING_DOWN':
            continue

        ts = tr['event_hour_ts']
        trade_date = ts.date() if hasattr(ts, 'date') else ts
        atr_bucket = atr_lookup.get(trade_date)
        if atr_bucket == BUCKET_TRANSITION:
            continue  # ATR-transition filter applied

        excess = float(tr['realized_return_sd30d']) - MEDIUM_COST
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
        quarter = (ts.month - 1) // 3 + 1 if hasattr(ts, 'month') else 1

        trades.append(dict(
            symbol=symbol,
            event_hour_ts=ts,
            excess=excess,
            year=year,
            quarter=quarter,
            quarter_label=f'Q{year}Q{quarter}',
        ))
    return trades


# ── Sleeve construction ────────────────────────────────────────────────────────

def build_sleeve_trades(
    symbol_trades: dict[str, list[dict]],
    weights: dict[str, float],
) -> list[dict]:
    """
    Combine per-symbol trade lists into sleeve-level trades with weighted returns.
    Each trade is scaled by its symbol's weight.
    Trades from different symbols are kept as individual rows (not aggregated per day)
    to preserve individual trade-level statistics.
    """
    sleeve = []
    for symbol, w in weights.items():
        for tr in symbol_trades.get(symbol, []):
            entry = dict(tr)
            entry['weighted_excess'] = tr['excess'] * w
            entry['weight'] = w
            sleeve.append(entry)
    sleeve.sort(key=lambda x: x['event_hour_ts'])
    return sleeve


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict], key: str = 'weighted_excess') -> Optional[dict]:
    vals = [tr[key] for tr in trades]
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
    return dict(
        n=n, mean=mn, std=sd, sharpe=sharpe,
        win_rate=win_rate, payoff=payoff, max_dd=max_dd,
        p_lt_neg1=p_lt_neg1, p_gt_pos1=p_gt_pos1,
        cum_pnl=cum,
    )


def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


def quarterly_splits(trades: list[dict]) -> dict[str, list[dict]]:
    quarters: dict[str, list[dict]] = defaultdict(list)
    for tr in trades:
        quarters[tr['quarter_label']].append(tr)
    return dict(quarters)


# ── Symbol contribution analysis ───────────────────────────────────────────────

def compute_symbol_contributions(
    symbol_trades: dict[str, list[dict]],
    weights: dict[str, float],
    split_trades_fn,
) -> dict[str, dict[str, dict]]:
    """
    For each split, compute each symbol's share of sleeve PnL, variance,
    and drawdown. Returns dict[split_name][symbol] -> contribution metrics.
    """
    results: dict[str, dict] = {}

    for sp_name in SPLITS:
        sp_result: dict[str, dict] = {}
        symbol_pnls: dict[str, float] = {}
        symbol_variances: dict[str, float] = {}

        for symbol, w in weights.items():
            sym_trades = split_trades_fn(
                [tr for tr in symbol_trades.get(symbol, [])]
            )[sp_name]
            if not sym_trades:
                symbol_pnls[symbol] = 0.0
                symbol_variances[symbol] = 0.0
                continue
            vals = [tr['excess'] * w for tr in sym_trades]
            symbol_pnls[symbol] = sum(vals)
            symbol_variances[symbol] = sum((v - sum(vals) / len(vals)) ** 2
                                           for v in vals) if len(vals) > 1 else 0.0

        total_pnl = sum(abs(v) for v in symbol_pnls.values())
        total_var = sum(symbol_variances.values())

        # Drawdown contribution: per-symbol max drawdown share (weighted)
        sleeve = build_sleeve_trades(symbol_trades, weights)
        sp_sleeve = split_trades(sleeve)[sp_name]
        sleeve_s = compute_stats(sp_sleeve)
        sleeve_max_dd = sleeve_s['max_dd'] if sleeve_s else 0.0

        for symbol, w in weights.items():
            sym_trades = split_trades(
                [tr for tr in symbol_trades.get(symbol, [])]
            )[sp_name]
            sym_vals = [tr['excess'] * w for tr in sym_trades]
            sym_cum = 0.0
            sym_peak = 0.0
            sym_max_dd = 0.0
            for v in sym_vals:
                sym_cum += v
                if sym_cum > sym_peak:
                    sym_peak = sym_cum
                dd = sym_peak - sym_cum
                if dd > sym_max_dd:
                    sym_max_dd = dd

            pnl_share = symbol_pnls[symbol] / total_pnl if total_pnl > 0 else 0.0
            var_share = symbol_variances[symbol] / total_var if total_var > 0 else 0.0
            dd_share = sym_max_dd / sleeve_max_dd if sleeve_max_dd > 0 else 0.0

            sp_result[symbol] = dict(
                n_trades=len(sym_trades),
                pnl_contribution=pnl_share,
                variance_contribution=var_share,
                drawdown_contribution=dd_share,
            )
        results[sp_name] = sp_result

    return results


# ── TimescaleDB upserts ────────────────────────────────────────────────────────

def upsert_pre_sizing(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.etd_sleeve_pre_sizing_results (
                    sleeve_name, weighting_scheme, split_name,
                    n_trades, mean_return, sharpe, max_drawdown,
                    win_rate, payoff, p_lt_neg1, p_gt_pos1,
                    symbol_concentration, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sleeve_name, weighting_scheme, split_name)
                DO UPDATE SET
                    n_trades             = EXCLUDED.n_trades,
                    mean_return          = EXCLUDED.mean_return,
                    sharpe               = EXCLUDED.sharpe,
                    max_drawdown         = EXCLUDED.max_drawdown,
                    win_rate             = EXCLUDED.win_rate,
                    payoff               = EXCLUDED.payoff,
                    p_lt_neg1            = EXCLUDED.p_lt_neg1,
                    p_gt_pos1            = EXCLUDED.p_gt_pos1,
                    symbol_concentration = EXCLUDED.symbol_concentration,
                    updated_at           = now()
            """, (
                r['sleeve_name'], r['weighting_scheme'], r['split_name'],
                r.get('n_trades'), r.get('mean_return'), r.get('sharpe'),
                r.get('max_drawdown'), r.get('win_rate'), r.get('payoff'),
                r.get('p_lt_neg1'), r.get('p_gt_pos1'),
                r.get('symbol_concentration'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def upsert_contributions(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.etd_sleeve_symbol_contributions (
                    sleeve_name, weighting_scheme, symbol, split_name,
                    n_trades, pnl_contribution, variance_contribution,
                    drawdown_contribution, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (sleeve_name, weighting_scheme, symbol, split_name)
                DO UPDATE SET
                    n_trades              = EXCLUDED.n_trades,
                    pnl_contribution      = EXCLUDED.pnl_contribution,
                    variance_contribution = EXCLUDED.variance_contribution,
                    drawdown_contribution = EXCLUDED.drawdown_contribution,
                    updated_at            = now()
            """, (
                r['sleeve_name'], r['weighting_scheme'], r['symbol'], r['split_name'],
                r.get('n_trades'), r.get('pnl_contribution'),
                r.get('variance_contribution'), r.get('drawdown_contribution'),
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


def pct(val: Optional[float]) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.1%}'


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/etd_pre_sizing.csv')
    parser.add_argument('--memo',   default='results/etd_pre_sizing_memo.txt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    conn = connect()
    create_tables(conn)
    conn.close()

    # ── Load filtered trades ───────────────────────────────────────────────────
    print(f'Loading ATR-filtered ETD trades for frozen sleeve: {SLEEVE_SYMBOLS}')
    symbol_trades: dict[str, list[dict]] = {}
    for symbol in SLEEVE_SYMBOLS:
        trades = build_filtered_trades(symbol)
        symbol_trades[symbol] = trades
        print(f'  {symbol:<10}  n={len(trades)}')

    # ── Build sleeve trades per scheme ────────────────────────────────────────
    scheme_sleeves: dict[str, list[dict]] = {}
    for scheme_name, weights in WEIGHTING_SCHEMES.items():
        scheme_sleeves[scheme_name] = build_sleeve_trades(symbol_trades, weights)

    # ── Compute sleeve-level stats ────────────────────────────────────────────
    print('\nComputing sleeve-level statistics...')
    db_presizing: list[dict] = []
    csv_rows: list[dict] = []
    fieldnames = ['scheme', 'split', 'n', 'mean', 'sharpe', 'max_dd',
                  'win_rate', 'payoff', 'p_lt_neg1', 'p_gt_pos1', 'concentration']

    scheme_split_stats: dict[str, dict[str, Optional[dict]]] = {}
    for scheme_name, sleeve in scheme_sleeves.items():
        sp = split_trades(sleeve)
        scheme_split_stats[scheme_name] = {}
        weights = WEIGHTING_SCHEMES[scheme_name]

        for sp_name, sp_trades in sp.items():
            s = compute_stats(sp_trades)

            # Concentration: max symbol PnL share in this split
            sym_pnls = {}
            for symbol, w in weights.items():
                sym_sp = [tr for tr in symbol_trades.get(symbol, [])
                          if tr in sp_trades or True]  # refilter by year
                sym_sp_filtered = [tr for tr in symbol_trades.get(symbol, [])
                                   if _in_split(tr, sp_name)]
                sym_pnls[symbol] = sum(tr['excess'] * w for tr in sym_sp_filtered)
            total_abs_pnl = sum(abs(v) for v in sym_pnls.values())
            concentration = (max(abs(v) for v in sym_pnls.values()) / total_abs_pnl
                             if total_abs_pnl > 0 else 0.0)

            scheme_split_stats[scheme_name][sp_name] = s
            db_presizing.append(dict(
                sleeve_name=SLEEVE_NAME,
                weighting_scheme=scheme_name,
                split_name=sp_name,
                n_trades=s['n'] if s else None,
                mean_return=s['mean'] if s else None,
                sharpe=s['sharpe'] if s else None,
                max_drawdown=s['max_dd'] if s else None,
                win_rate=s['win_rate'] if s else None,
                payoff=s['payoff'] if s else None,
                p_lt_neg1=s['p_lt_neg1'] if s else None,
                p_gt_pos1=s['p_gt_pos1'] if s else None,
                symbol_concentration=concentration,
            ))
            csv_rows.append(dict(
                scheme=scheme_name, split=sp_name,
                n=s['n'] if s else '',
                mean=fmt(s['mean'] if s else None),
                sharpe=fmt(s['sharpe'] if s else None),
                max_dd=fmt(s['max_dd'] if s else None),
                win_rate=fmt(s['win_rate'] if s else None),
                payoff=fmt(s['payoff'] if s else None),
                p_lt_neg1=fmt(s['p_lt_neg1'] if s else None),
                p_gt_pos1=fmt(s['p_gt_pos1'] if s else None),
                concentration=fmt(concentration),
            ))

    # ── Rolling quarterly stats ────────────────────────────────────────────────
    print('Computing rolling quarterly stats...')
    for scheme_name, sleeve in scheme_sleeves.items():
        quarters = quarterly_splits(sleeve)
        for q_label, q_trades in sorted(quarters.items()):
            if len(q_trades) < 2:
                continue
            s = compute_stats(q_trades)
            if not s:
                continue
            db_presizing.append(dict(
                sleeve_name=SLEEVE_NAME,
                weighting_scheme=scheme_name,
                split_name=q_label,
                n_trades=s['n'],
                mean_return=s['mean'],
                sharpe=s['sharpe'],
                max_drawdown=s['max_dd'],
                win_rate=s['win_rate'],
                payoff=s['payoff'],
                p_lt_neg1=s['p_lt_neg1'],
                p_gt_pos1=s['p_gt_pos1'],
                symbol_concentration=None,
            ))

    # ── Symbol contributions ───────────────────────────────────────────────────
    print('Computing symbol contributions...')
    db_contrib: list[dict] = []
    scheme_contrib_stats: dict[str, dict[str, dict[str, dict]]] = {}

    for scheme_name, weights in WEIGHTING_SCHEMES.items():
        contrib = compute_symbol_contributions(symbol_trades, weights, split_trades)
        scheme_contrib_stats[scheme_name] = contrib
        for sp_name, sym_dict in contrib.items():
            for symbol, metrics in sym_dict.items():
                db_contrib.append(dict(
                    sleeve_name=SLEEVE_NAME,
                    weighting_scheme=scheme_name,
                    symbol=symbol,
                    split_name=sp_name,
                    **metrics,
                ))

    # ── Upsert to DB ──────────────────────────────────────────────────────────
    print('\nUpserting etd_sleeve_pre_sizing_results...')
    r1 = upsert_pre_sizing(db_presizing)
    print(f'  {r1} rows written')

    print('Upserting etd_sleeve_symbol_contributions...')
    r2 = upsert_contributions(db_contrib)
    print(f'  {r2} rows written')

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # ── Determine recommendation ───────────────────────────────────────────────
    eq_fwd = scheme_split_stats['equal_weight'].get('fwd')
    cap_fwd = scheme_split_stats['capped_conviction'].get('fwd')
    eq_full = scheme_split_stats['equal_weight'].get('full')

    fwd_sharpe_eq = eq_fwd['sharpe'] if eq_fwd else None
    fwd_sharpe_cap = cap_fwd['sharpe'] if cap_fwd else None

    # Check concentration in forward
    contrib_eq_fwd = scheme_contrib_stats['equal_weight'].get('fwd', {})
    max_pnl_conc = max(
        (v.get('pnl_contribution', 0) for v in contrib_eq_fwd.values()),
        default=0.0,
    )

    # Decision rules
    all_fwd_negative = all(
        (scheme_split_stats[s].get('fwd') or {}).get('sharpe', -1) < 0
        for s in WEIGHTING_SCHEMES
    )
    dominated = max_pnl_conc > CONCENTRATION_FLAG_THRESHOLD
    capped_hurts_fwd = (
        fwd_sharpe_eq is not None and fwd_sharpe_cap is not None and
        (fwd_sharpe_cap - fwd_sharpe_eq) < CAPPED_FWD_DEGRADE_THRESHOLD
    )
    schemes_similar = (
        fwd_sharpe_eq is not None and fwd_sharpe_cap is not None and
        abs(fwd_sharpe_cap - fwd_sharpe_eq) < 0.05
    )

    if all_fwd_negative:
        recommendation = 'STOP'
        rec_text = 'All weighting schemes show negative forward Sharpe. Do not proceed to advanced sizing.'
    elif dominated:
        recommendation = 'MONITOR'
        rec_text = (f'Sleeve PnL is dominated by one symbol ({pct(max_pnl_conc)} concentration). '
                    f'Do not proceed to advanced sizing until concentration reduces.')
    elif capped_hurts_fwd:
        recommendation = 'EQUAL_WEIGHT'
        rec_text = ('Capped-conviction worsens forward materially. Proceed with equal-weight sleeve only.')
    elif schemes_similar:
        recommendation = 'EQUAL_WEIGHT'
        rec_text = ('Equal-weight and capped-conviction schemes are similar in forward. '
                    'Default to equal-weight by pre-declared rule.')
    else:
        recommendation = 'EQUAL_WEIGHT'
        rec_text = 'Equal-weight sleeve is the robust default. Proceed to advanced sizing research.'

    # ── Memo ───────────────────────────────────────────────────────────────────
    W = 76
    lines: list[str] = []
    lines.append('=' * W)
    lines.append('ETD SLEEVE PRE-SIZING RESEARCH  (Issue #103)')
    lines.append('Frozen sleeve: USD/CHF + GBP/USD, ATR-transition filter only.')
    lines.append('Simple fixed weights only. No optimization. No Kelly. No pyramiding.')
    lines.append('=' * W)
    lines.append('')
    lines.append('FROZEN STRATEGY OBJECT')
    lines.append('  State  : EXPANDING_TRENDING_DOWN')
    lines.append('  Filter : ATR-transition filter only (atr_bucket == TRANSITION_1_2 excluded)')
    lines.append('  Symbols: USD/CHF (anchor), GBP/USD (validated secondary)')
    lines.append('')
    lines.append('PRE-DECLARED WEIGHTING SCHEMES')
    for sname, weights in WEIGHTING_SCHEMES.items():
        wstr = '  '.join(f'{sym}={w:.0%}' for sym, w in weights.items())
        lines.append(f'  {sname:<22} {wstr}')
    lines.append('')
    lines.append('PRE-DECLARED DECISION RULES')
    lines.append(f'  1. If capped_conviction worsens forward vs equal_weight by > {abs(CAPPED_FWD_DEGRADE_THRESHOLD):.2f}: reject')
    lines.append(f'  2. If schemes within 0.05 fwd Sharpe of each other: default to equal_weight')
    lines.append(f'  3. If one symbol contributes > {CONCENTRATION_FLAG_THRESHOLD:.0%} of PnL: flag, do not progress')
    lines.append(f'  4. If all forward Sharpe < 0: STOP, do not proceed to advanced sizing')
    lines.append('')

    # Sleeve performance table
    lines.append('SLEEVE PERFORMANCE BY WEIGHTING SCHEME')
    lines.append(
        f'  {"Scheme":<22} {"split":>6} {"n":>5} {"Sharpe":>8}'
        f' {"MaxDD":>8} {"WinR":>7} {"P<-1":>7} {"P>+1":>7} {"Conc":>7}'
    )
    lines.append(
        f'  {"-"*22} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*7} {"-"*7}'
    )
    for scheme_name in WEIGHTING_SCHEMES:
        for sp_name in SPLITS:
            s = scheme_split_stats[scheme_name].get(sp_name)
            # Get concentration for this split
            conc_row = next(
                (r for r in db_presizing
                 if r['weighting_scheme'] == scheme_name
                 and r['split_name'] == sp_name
                 and r.get('symbol_concentration') is not None),
                None,
            )
            conc_val = conc_row['symbol_concentration'] if conc_row else None
            if s:
                lines.append(
                    f'  {scheme_name:<22} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["win_rate"]:>7.3f} {s["p_lt_neg1"]:>7.3f}'
                    f' {s["p_gt_pos1"]:>7.3f} {fmt(conc_val):>7}'
                )
            else:
                lines.append(f'  {scheme_name:<22} {sp_name:>6}   N/A')
        lines.append('')

    # Symbol contributions
    lines.append('SYMBOL CONTRIBUTION ANALYSIS (% of sleeve PnL / Variance / Drawdown)')
    lines.append(
        f'  {"Scheme":<22} {"split":>6} {"Symbol":<10}'
        f' {"PnL%":>8} {"Var%":>8} {"DD%":>8} {"n":>5}'
    )
    lines.append(
        f'  {"-"*22} {"-"*6} {"-"*10} {"-"*8} {"-"*8} {"-"*8} {"-"*5}'
    )
    for scheme_name in WEIGHTING_SCHEMES:
        for sp_name in SPLITS:
            contrib = scheme_contrib_stats[scheme_name].get(sp_name, {})
            for symbol in SLEEVE_SYMBOLS:
                m = contrib.get(symbol, {})
                lines.append(
                    f'  {scheme_name:<22} {sp_name:>6} {symbol:<10}'
                    f' {pct(m.get("pnl_contribution")):>8}'
                    f' {pct(m.get("variance_contribution")):>8}'
                    f' {pct(m.get("drawdown_contribution")):>8}'
                    f' {m.get("n_trades", 0):>5}'
                )
        lines.append('')

    # Rolling quarterly (equal_weight only)
    lines.append('ROLLING QUARTERLY PERFORMANCE (equal_weight and capped_conviction)')
    lines.append(
        f'  {"Quarter":<10} {"Scheme":<22} {"n":>5}'
        f' {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"P<-1":>7}'
    )
    lines.append(f'  {"-"*10} {"-"*22} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7}')

    quarterly_rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in db_presizing:
        if r['split_name'].startswith('Q') and r['weighting_scheme'] in ('equal_weight', 'capped_conviction'):
            quarterly_rows[r['split_name']][r['weighting_scheme']] = r

    for q_label in sorted(quarterly_rows.keys()):
        for scheme_name in ('equal_weight', 'capped_conviction'):
            r = quarterly_rows[q_label].get(scheme_name)
            if r and r.get('n_trades'):
                lines.append(
                    f'  {q_label:<10} {scheme_name:<22} {r["n_trades"]:>5}'
                    f' {fmt(r["sharpe"]):>8} {fmt(r["max_drawdown"]):>8}'
                    f' {fmt(r["win_rate"]):>7} {fmt(r["p_lt_neg1"]):>7}'
                )
    lines.append('')

    # Forward-only pre-sizing gate
    lines.append('FORWARD-ONLY PRE-SIZING GATE')
    lines.append(
        f'  {"Scheme":<22} {"fwd_n":>6} {"fwd_Sharpe":>11}'
        f' {"fwd_MaxDD":>10} {"fwd_WinR":>9} {"fwd_P<-1":>9}'
    )
    lines.append(f'  {"-"*22} {"-"*6} {"-"*11} {"-"*10} {"-"*9} {"-"*9}')
    all_fwd_pass = True
    for scheme_name in WEIGHTING_SCHEMES:
        s = scheme_split_stats[scheme_name].get('fwd')
        if s:
            gate = 'PASS' if s['sharpe'] >= 0 else 'FAIL'
            if s['sharpe'] < 0:
                all_fwd_pass = False
            lines.append(
                f'  {scheme_name:<22} {s["n"]:>6} {s["sharpe"]:>11.3f}'
                f' {s["max_dd"]:>10.3f} {s["win_rate"]:>9.3f} {s["p_lt_neg1"]:>9.3f}'
                f'  [{gate}]'
            )
        else:
            lines.append(f'  {scheme_name:<22}   N/A')
            all_fwd_pass = False
    lines.append('')

    # Decision rules evaluation
    lines.append('DECISION RULE EVALUATION')
    lines.append(f'  All forward Sharpe < 0:          {"YES — STOP" if all_fwd_negative else "No"}')
    lines.append(f'  Concentration flag (>{CONCENTRATION_FLAG_THRESHOLD:.0%}):    {"YES — " + pct(max_pnl_conc) if dominated else "No — " + pct(max_pnl_conc)}')
    lines.append(f'  Capped hurts forward (>{abs(CAPPED_FWD_DEGRADE_THRESHOLD):.2f}): {"YES" if capped_hurts_fwd else "No"}')
    lines.append(f'  Schemes similar (<0.05 fwd):     {"Yes" if schemes_similar else "No"}')
    lines.append('')
    lines.append('RECOMMENDATION')
    lines.append(f'  {recommendation}: {rec_text}')
    lines.append('')
    if recommendation in ('EQUAL_WEIGHT',):
        lines.append('  Acceptance criteria check:')
        checks = []
        eq_fwd_s = scheme_split_stats['equal_weight'].get('fwd')
        eq_full_s = scheme_split_stats['equal_weight'].get('full')
        checks.append(('Forward Sharpe >= 0',
                        eq_fwd_s is not None and eq_fwd_s['sharpe'] >= 0,
                        fmt(eq_fwd_s['sharpe'] if eq_fwd_s else None)))
        checks.append(('Forward MaxDD <= full MaxDD',
                        (eq_fwd_s is not None and eq_full_s is not None and
                         eq_fwd_s['max_dd'] <= eq_full_s['max_dd']),
                        f'fwd={fmt(eq_fwd_s["max_dd"] if eq_fwd_s else None)}'
                        f' full={fmt(eq_full_s["max_dd"] if eq_full_s else None)}'))
        checks.append(('No extreme concentration (<=70%)',
                        not dominated,
                        pct(max_pnl_conc)))
        checks.append(('At least one scheme positive fwd',
                        not all_fwd_negative,
                        ''))
        checks.append(('Simple logic supportable',
                        True, 'equal_weight is unconditionally simple'))
        passed = sum(1 for _, ok, _ in checks if ok)
        for label, ok, note in checks:
            marker = 'PASS' if ok else 'FAIL'
            lines.append(f'    {marker}  {label}{"  (" + note + ")" if note else ""}')
        verdict = 'PASS' if passed >= 4 else ('MARGINAL' if passed == 3 else 'FAIL')
        lines.append(f'  Overall: {verdict} ({passed}/5)')
        lines.append('')
        lines.append('  Next steps:')
        if verdict in ('PASS', 'MARGINAL'):
            lines.append('    - Proceed with equal-weight ETD sleeve (USD/CHF 50%, GBP/USD 50%)')
            lines.append('    - Advanced sizing research (Kelly criterion) is now unblocked')
            lines.append('    - CTU sleeve integration to follow in a separate issue')
        else:
            lines.append('    - Do not proceed to advanced sizing')
            lines.append('    - Monitor sleeve forward performance for 2 more quarters')

    lines.append('')
    lines.append(f'Output: {args.output}')
    lines.append(f'Memo:   {args.memo}')

    memo_text = '\n'.join(lines) + '\n'
    with open(args.memo, 'w') as fh:
        fh.write(memo_text)

    print()
    print(memo_text)
    print(f'Memo written to {args.memo}')


def _in_split(tr: dict, sp_name: str) -> bool:
    if sp_name == 'full':
        return True
    if sp_name == 'train':
        return tr['year'] < TRAIN_END_YEAR
    if sp_name == 'val':
        return TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR
    if sp_name == 'fwd':
        return tr['year'] >= VAL_END_YEAR
    return False


if __name__ == '__main__':
    main()

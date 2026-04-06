# ETD ARCHIVE — reference only. Do not modify without a new issue.
"""
scripts/markov_etd_robustness.py

ETD robustness validation -- Issue #96.

Four gating tests before ETD is promoted to sizing research:

  Test 1 — Tail dependency:
    Remove top-1 and top-3 trades by excess return.
    PASS if Sharpe and mean remain materially positive.

  Test 2 — Forward-split pruning stability:
    Split into train (pre-2022), validation (2022-2023), forward (2024+).
    For each symbol-state check mean per period.
    PASS if pruned symbols are negative/unstable in all periods,
    and no removed symbol is positive in forward.

  Test 3 — Symbol contribution concentration:
    Compute % PnL and % variance per symbol.
    PASS if no single symbol > 40% of total PnL.

  Test 4 — Sensitivity (pruning robustness):
    Reintroduce each removed symbol one at a time.
    PASS if performance does not collapse (Sharpe drops < 0.05).

Usage
-----
    python scripts/markov_etd_robustness.py
    python scripts/markov_etd_robustness.py --output results/etd_robustness.csv
    python scripts/markov_etd_robustness.py --output results/etd_robustness.csv \
        --memo results/etd_robustness_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
try:
    _project_root = Path(__file__).resolve().parents[1]
    _env_path = _project_root / '.env'
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path)
except Exception:
    pass

TIMESCALE_DSN = os.getenv('TIMESCALE_DSN')
PGHOST        = os.getenv('PGHOST',     'localhost')
PGDATABASE    = os.getenv('PGDATABASE', 'market_data')
PGUSER        = os.getenv('PGUSER',     'backtesting')
PGPASSWORD    = os.getenv('PGPASSWORD', 'backtesting_pass')
PGPORT        = int(os.getenv('PGPORT', '5434'))


def get_connection() -> psycopg2.extensions.connection:
    if TIMESCALE_DSN:
        return psycopg2.connect(TIMESCALE_DSN)
    return psycopg2.connect(
        host=PGHOST, dbname=PGDATABASE,
        user=PGUSER, password=PGPASSWORD, port=PGPORT,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETD_STATE:       str       = 'EXPANDING_TRENDING_DOWN'
COHORT_SYMBOLS:  list[str] = [
    'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
]
ETD_KEPT_SYMS:    list[str] = ['EUR/GBP', 'GBP/USD', 'USD/CAD', 'USD/CHF']
ETD_REMOVED_SYMS: list[str] = ['AUD/USD', 'NZD/USD', 'EUR/USD']

MEDIUM_COST:    float = 0.07
DATA_SPAN_YEARS: float = 11.0

# Split boundaries
TRAIN_END:   str = '2022-01-01'
VAL_END:     str = '2024-01-01'

# Thresholds
TAIL_SHARPE_FLOOR:    float = 0.05   # after removing top trades
TAIL_MEAN_FLOOR:      float = 0.0
CONCENTRATION_CEIL:   float = 0.40   # max single-symbol PnL share
SENSITIVITY_SHARPE_DROP: float = 0.05  # collapse if drop exceeds this

# Portfolio caps (matching issue #95 baseline)
MAX_PER_SYMBOL: int = 2
MAX_PER_STATE:  int = 5
MAX_TOTAL:      int = 10


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _median(vals: list[float]) -> float:
    sv  = sorted(vals)
    n   = len(sv)
    mid = n // 2
    return sv[mid] if n % 2 else (sv[mid - 1] + sv[mid]) / 2.0


def _std(vals: list[float]) -> Optional[float]:
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)), 4)


def _ts_date_str(ts: object) -> str:
    return ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)[:10]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_SQL = """
SELECT
    be.symbol,
    be.event_hour_ts,
    be.breakout_direction,
    be.vol_ratio_20_100,
    be.efficiency_20,
    be.realized_return_sd30d,
    be.bars_held,
    be.exit_reason
FROM features.breakout_events be
JOIN market_data.symbols s ON s.symbol = be.symbol
WHERE s.type = 'forex'
  AND be.exit_reason IS NOT NULL
  AND be.entry_range_sd_30d > 0
  AND be.vol_ratio_20_100   IS NOT NULL
  AND be.efficiency_20      IS NOT NULL
  AND be.realized_return_sd30d IS NOT NULL
ORDER BY be.symbol, be.event_hour_ts
"""


def _vol_bucket(v: float) -> str:
    if v < 0.80:
        return 'CONTRACTING'
    if v <= 1.20:
        return 'NEUTRAL'
    return 'EXPANDING'


def _eff_bucket(e: float) -> str:
    if e < 0.30:
        return 'CHOPPY'
    if e <= 0.60:
        return 'MIXED'
    return 'TRENDING'


def load_events() -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(_SQL)
            raw = cur.fetchall()
    finally:
        conn.close()

    events: list[dict] = []
    for row in raw:
        ev  = dict(row)
        vr  = ev.get('vol_ratio_20_100')
        eff = ev.get('efficiency_20')
        d   = ev.get('breakout_direction', '')
        if vr is None or eff is None or d not in ('UP', 'DOWN'):
            continue
        ev['vol_ratio_20_100'] = float(vr)
        ev['efficiency_20']    = float(eff)
        ev['state'] = (
            f'{_vol_bucket(ev["vol_ratio_20_100"])}_'
            f'{_eff_bucket(ev["efficiency_20"])}_'
            f'{d}'
        )
        events.append(ev)
    return events


def compute_baselines(events: list[dict]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        key = (ev['symbol'], ev['breakout_direction'])
        groups.setdefault(key, []).append(float(ev['realized_return_sd30d']))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def apply_excess_return(
    events: list[dict],
    baselines: dict[tuple[str, str], float],
) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        b = baselines.get((ev['symbol'], ev['breakout_direction']))
        if b is None:
            continue
        ne = dict(ev)
        ne['excess_return'] = float(ev['realized_return_sd30d']) - b
        ne['cost_adj']      = ne['excess_return'] - MEDIUM_COST
        out.append(ne)
    return out


# ---------------------------------------------------------------------------
# Portfolio simulator (ETD only, baseline caps)
# ---------------------------------------------------------------------------

def simulate_portfolio(
    events: list[dict],
    eligible_symbols: Optional[list[str]],
) -> list[dict]:
    """Simulate ETD-only portfolio with baseline caps."""
    sym_set = set(eligible_symbols) if eligible_symbols is not None else None
    candidates = sorted(
        [ev for ev in events
         if ev.get('state') == ETD_STATE
         and (sym_set is None or ev['symbol'] in sym_set)
         and ev.get('excess_return') is not None],
        key=lambda ev: ev['event_hour_ts'],
    )

    active:   list[dict] = []
    executed: list[dict] = []

    for ev in candidates:
        entry_ts = ev['event_hour_ts']
        bars     = int(ev.get('bars_held') or 365)
        exit_ts  = entry_ts + timedelta(hours=bars)

        active = [a for a in active if a['exit_ts'] > entry_ts]

        sym_count = sum(1 for a in active if a['symbol'] == ev['symbol'])
        if sym_count   >= MAX_PER_SYMBOL:
            continue
        if len(active) >= MAX_TOTAL:
            continue

        active.append({'entry_ts': entry_ts, 'exit_ts': exit_ts,
                        'symbol': ev['symbol']})
        executed.append({
            'entry_ts':      entry_ts,
            'exit_ts':       exit_ts,
            'symbol':        ev['symbol'],
            'excess_return': float(ev['excess_return']),
            'cost_adj':      float(ev['cost_adj']),
        })

    executed.sort(key=lambda t: t['exit_ts'])
    return executed


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def compute_stats(trades: list[dict], label: str = '') -> dict:
    if not trades:
        return {'label': label, 'n': 0}
    returns = [t['cost_adj'] for t in trades]
    n       = len(returns)
    mean_r  = sum(returns) / n
    std_r   = _std(returns)
    sharpe  = round(mean_r / std_r, 4) if std_r and std_r > 0 else None

    cum    = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in returns:
        cum  += r
        peak  = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    wins   = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    return {
        'label':        label,
        'n':            n,
        'mean':         round(mean_r, 4),
        'median':       round(_median(returns), 4),
        'std':          std_r,
        'sharpe':       sharpe,
        'win_rate':     round(len(wins) / n, 4),
        'avg_win':      round(sum(wins) / len(wins), 4)   if wins   else None,
        'avg_loss':     round(sum(losses) / len(losses), 4) if losses else None,
        'max_drawdown': round(max_dd, 4),
        'cum_pnl':      round(cum, 2),
        'annual_rate':  round(n / DATA_SPAN_YEARS, 1),
    }


# ---------------------------------------------------------------------------
# Test 1 — Tail dependency
# ---------------------------------------------------------------------------

def test_tail_dependency(trades: list[dict]) -> dict:
    """
    Remove top-N trades by excess_return (not cost_adj, to identify extreme events).
    Recompute stats on remainder.

    PASS: mean > TAIL_MEAN_FLOOR AND Sharpe >= TAIL_SHARPE_FLOOR after removing top-3.
    """
    sorted_by_ret = sorted(trades, key=lambda t: t['excess_return'], reverse=True)
    baseline_stats = compute_stats(trades, label='baseline')

    rows: list[dict] = [{'n_removed': 0, **baseline_stats}]
    for n_remove in (1, 3):
        removed     = sorted_by_ret[:n_remove]
        removed_ids = {id(t) for t in removed}
        remainder   = [t for t in trades if id(t) not in removed_ids]
        s           = compute_stats(remainder, label=f'minus_top_{n_remove}')
        rows.append({'n_removed': n_remove, **s})

    # Log the top trades being removed
    top3 = sorted_by_ret[:3]

    pass_result = (
        rows[-1].get('mean') is not None and rows[-1]['mean'] > TAIL_MEAN_FLOOR
        and rows[-1].get('sharpe') is not None and rows[-1]['sharpe'] >= TAIL_SHARPE_FLOOR
    )

    return {
        'pass': pass_result,
        'rows': rows,
        'top3_trades': [
            {'symbol': t['symbol'], 'excess_return': t['excess_return'],
             'entry_ts': _ts_date_str(t['entry_ts'])}
            for t in top3
        ],
    }


# ---------------------------------------------------------------------------
# Test 2 — Forward-split pruning stability
# ---------------------------------------------------------------------------

def _period_label(ts: object) -> str:
    d = _ts_date_str(ts)
    if d < TRAIN_END:
        return 'train'
    if d < VAL_END:
        return 'val'
    return 'forward'


def _classify(train_m: Optional[float], val_m: Optional[float], fwd_m: Optional[float]) -> str:
    vals = [v for v in (train_m, val_m, fwd_m) if v is not None]
    if not vals:
        return 'no_data'
    n_pos = sum(1 for v in vals if v > 0)
    n_neg = sum(1 for v in vals if v < 0)
    if n_pos == len(vals):
        return 'consistently_positive'
    if n_neg == len(vals):
        return 'consistently_negative'
    return 'unstable'


def test_forward_split_pruning(etd_events: list[dict]) -> dict:
    """
    For every ETD symbol in cohort, compute mean excess_return per period.
    Classify as consistently_positive / unstable / consistently_negative.

    PASS conditions:
      - All pruned symbols are NOT consistently_positive in all periods
      - No pruned symbol is positive in forward period
    """
    # Group excess_return by (symbol, period)
    groups: dict[tuple[str, str], list[float]] = {}
    for ev in etd_events:
        if ev.get('state') != ETD_STATE:
            continue
        if ev['symbol'] not in COHORT_SYMBOLS:
            continue
        key = (ev['symbol'], _period_label(ev['event_hour_ts']))
        groups.setdefault(key, []).append(float(ev['excess_return']))

    rows: list[dict] = []
    pruning_stable = True
    removed_fwd_positive: list[str] = []

    for sym in sorted(COHORT_SYMBOLS):
        t_vals = groups.get((sym, 'train'), [])
        v_vals = groups.get((sym, 'val'), [])
        f_vals = groups.get((sym, 'forward'), [])

        t_m = round(sum(t_vals) / len(t_vals), 4) if t_vals else None
        v_m = round(sum(v_vals) / len(v_vals), 4) if v_vals else None
        f_m = round(sum(f_vals) / len(f_vals), 4) if f_vals else None

        classification = _classify(t_m, v_m, f_m)
        pruned_flag    = sym in ETD_REMOVED_SYMS

        # Pruning is unstable if a removed symbol is positive in forward
        if pruned_flag and f_m is not None and f_m > 0:
            pruning_stable = False
            removed_fwd_positive.append(sym)

        rows.append({
            'symbol':         sym,
            'pruned':         pruned_flag,
            'train_mean':     t_m,
            'val_mean':       v_m,
            'forward_mean':   f_m,
            'train_n':        len(t_vals),
            'val_n':          len(v_vals),
            'forward_n':      len(f_vals),
            'classification': classification,
        })

    return {
        'pass':                   pruning_stable,
        'rows':                   rows,
        'removed_fwd_positive':   removed_fwd_positive,
    }


# ---------------------------------------------------------------------------
# Test 3 — Symbol contribution concentration
# ---------------------------------------------------------------------------

def test_concentration(trades: list[dict]) -> dict:
    """
    Compute % of total PnL and % of variance per symbol.
    PASS if no symbol > CONCENTRATION_CEIL of total absolute PnL.
    """
    sym_returns: dict[str, list[float]] = {}
    for t in trades:
        sym_returns.setdefault(t['symbol'], []).append(t['cost_adj'])

    total_pnl = sum(t['cost_adj'] for t in trades)
    total_var = sum((t['cost_adj'] - sum(t2['cost_adj'] for t2 in trades) / len(trades)) ** 2
                    for t in trades) if len(trades) > 1 else 0.0

    rows: list[dict] = []
    max_pnl_frac = 0.0
    for sym in sorted(sym_returns):
        rets      = sym_returns[sym]
        sym_pnl   = sum(rets)
        mean_all  = total_pnl / len(trades) if trades else 0.0
        sym_var   = sum((r - mean_all) ** 2 for r in rets)
        pnl_frac  = round(sym_pnl / total_pnl, 4) if total_pnl != 0 else None
        var_frac  = round(sym_var / total_var, 4)  if total_var > 0  else None
        max_pnl_frac = max(max_pnl_frac, abs(pnl_frac) if pnl_frac is not None else 0.0)
        rows.append({
            'symbol':    sym,
            'n':         len(rets),
            'pnl':       round(sym_pnl, 2),
            'pnl_frac':  pnl_frac,
            'var_frac':  var_frac,
        })

    rows.sort(key=lambda r: abs(r.get('pnl_frac') or 0), reverse=True)
    return {
        'pass':         max_pnl_frac <= CONCENTRATION_CEIL,
        'max_pnl_frac': round(max_pnl_frac, 4),
        'total_pnl':    round(total_pnl, 2),
        'rows':         rows,
    }


# ---------------------------------------------------------------------------
# Test 4 — Sensitivity (pruning robustness)
# ---------------------------------------------------------------------------

def test_sensitivity(etd_events: list[dict], baseline_sharpe: Optional[float]) -> dict:
    """
    Reintroduce each removed ETD symbol individually.
    PASS if Sharpe does not drop by more than SENSITIVITY_SHARPE_DROP vs baseline.
    """
    rows: list[dict] = []
    all_pass = True

    for sym in ETD_REMOVED_SYMS:
        syms   = ETD_KEPT_SYMS + [sym]
        trades = simulate_portfolio(etd_events, syms)
        stats  = compute_stats(trades, label=f'add_{sym}')
        new_sharpe = stats.get('sharpe')

        if baseline_sharpe is not None and new_sharpe is not None:
            drop  = round(baseline_sharpe - new_sharpe, 4)
            fails = drop > SENSITIVITY_SHARPE_DROP
        elif new_sharpe is None:
            drop  = None
            fails = True
        else:
            drop  = None
            fails = False

        if fails:
            all_pass = False

        rows.append({
            'symbol_added':      sym,
            'n':                 stats.get('n', 0),
            'mean':              stats.get('mean'),
            'sharpe':            new_sharpe,
            'sharpe_drop':       drop,
            'max_drawdown':      stats.get('max_drawdown'),
            'stable':            not fails,
        })

    return {
        'pass': all_pass,
        'rows': rows,
        'baseline_sharpe': baseline_sharpe,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_header(section: str) -> None:
    log.info('')
    log.info('=' * 80)
    log.info('  %s', section)
    log.info('=' * 80)


def print_test1(result: dict) -> None:
    print_header('TEST 1 — Tail dependency')
    log.info('  Threshold: mean > %.2f AND Sharpe >= %.2f after removing top-3',
             TAIL_MEAN_FLOOR, TAIL_SHARPE_FLOOR)
    log.info('')
    log.info('  Top-3 trades by excess return (pre-cost):')
    for t in result.get('top3_trades', []):
        log.info('    %s  %s  excess=+%.4f',
                 t['entry_ts'], t['symbol'], t['excess_return'])
    log.info('')
    log.info('  %-20s  %5s  %8s  %8s  %8s  %7s',
             'scenario', 'n', 'mean', 'sharpe', 'max_dd', 'win%')
    log.info('  ' + '-' * 66)
    for row in result.get('rows', []):
        n_rem = row.get('n_removed', 0)
        lbl   = 'baseline' if n_rem == 0 else f'minus_top_{n_rem}'
        log.info('  %-20s  %5d  %8s  %8s  %8s  %6.1f%%',
                 lbl,
                 row.get('n', 0),
                 f'{row["mean"]:+.4f}'        if row.get('mean')        is not None else 'N/A',
                 f'{row["sharpe"]:+.4f}'      if row.get('sharpe')      is not None else 'N/A',
                 f'{row["max_drawdown"]:.4f}' if row.get('max_drawdown') is not None else 'N/A',
                 (row.get('win_rate') or 0) * 100,
                 )
    log.info('  Result: %s', 'PASS' if result.get('pass') else 'FAIL')


def print_test2(result: dict) -> None:
    print_header('TEST 2 — Forward-split pruning stability')
    log.info('  Periods: train=pre-2022  val=2022-2023  forward=2024+')
    log.info('  Excess return = realized - symbol-direction baseline')
    log.info('')
    log.info('  %-12s  %-7s  %10s  %6s  %10s  %6s  %10s  %6s  %-25s',
             'symbol', 'pruned',
             'train_mean', 'n',
             'val_mean', 'n',
             'fwd_mean', 'n',
             'classification')
    log.info('  ' + '-' * 100)
    for row in result.get('rows', []):
        log.info(
            '  %-12s  %-7s  %10s  %6d  %10s  %6d  %10s  %6d  %-25s',
            row['symbol'],
            'PRUNED' if row.get('pruned') else 'kept',
            f'{row["train_mean"]:+.4f}'   if row.get('train_mean')   is not None else 'N/A',
            row.get('train_n', 0),
            f'{row["val_mean"]:+.4f}'     if row.get('val_mean')     is not None else 'N/A',
            row.get('val_n', 0),
            f'{row["forward_mean"]:+.4f}' if row.get('forward_mean') is not None else 'N/A',
            row.get('forward_n', 0),
            row.get('classification', ''),
        )
    log.info('')
    if result.get('removed_fwd_positive'):
        log.info('  WARNING: removed symbols positive in forward: %s',
                 result['removed_fwd_positive'])
    log.info('  Result: %s', 'PASS' if result.get('pass') else 'FAIL')


def print_test3(result: dict) -> None:
    print_header('TEST 3 — Symbol PnL concentration')
    log.info('  Ceiling: no single symbol > %.0f%% of total PnL',
             CONCENTRATION_CEIL * 100)
    log.info('  Total PnL: %.2f SD30d', result.get('total_pnl', 0))
    log.info('')
    log.info('  %-12s  %5s  %10s  %8s  %9s',
             'symbol', 'n', 'pnl', 'pnl_%', 'var_%')
    log.info('  ' + '-' * 50)
    for row in result.get('rows', []):
        log.info(
            '  %-12s  %5d  %10.2f  %8s  %9s',
            row['symbol'],
            row.get('n', 0),
            row.get('pnl', 0),
            f'{row["pnl_frac"]*100:+.1f}%' if row.get('pnl_frac') is not None else 'N/A',
            f'{row["var_frac"]*100:.1f}%'   if row.get('var_frac') is not None else 'N/A',
        )
    log.info('  Max single-symbol PnL share: %.1f%%  (ceiling=%.0f%%)',
             result.get('max_pnl_frac', 0) * 100, CONCENTRATION_CEIL * 100)
    log.info('  Result: %s', 'PASS' if result.get('pass') else 'FAIL')


def print_test4(result: dict) -> None:
    print_header('TEST 4 — Sensitivity (pruning robustness)')
    log.info('  Baseline Sharpe: %s',
             f'{result["baseline_sharpe"]:+.4f}' if result.get('baseline_sharpe') is not None else 'N/A')
    log.info('  Collapse threshold: Sharpe drop > %.2f', SENSITIVITY_SHARPE_DROP)
    log.info('')
    log.info('  %-12s  %5s  %8s  %8s  %10s  %10s  %8s',
             'symbol_added', 'n', 'mean', 'sharpe', 'sharpe_drop', 'max_dd', 'stable')
    log.info('  ' + '-' * 70)
    for row in result.get('rows', []):
        log.info(
            '  %-12s  %5d  %8s  %8s  %10s  %10s  %8s',
            row['symbol_added'],
            row.get('n', 0),
            f'{row["mean"]:+.4f}'         if row.get('mean')         is not None else 'N/A',
            f'{row["sharpe"]:+.4f}'       if row.get('sharpe')       is not None else 'N/A',
            f'{row["sharpe_drop"]:+.4f}'  if row.get('sharpe_drop')  is not None else 'N/A',
            f'{row["max_drawdown"]:.4f}'  if row.get('max_drawdown') is not None else 'N/A',
            'YES' if row.get('stable') else 'NO',
        )
    log.info('  Result: %s', 'PASS' if result.get('pass') else 'FAIL')


def print_recommendation(
    t1: dict, t2: dict, t3: dict, t4: dict,
    baseline_stats: dict,
) -> str:
    print_header('FINAL RECOMMENDATION')
    results = {
        'tail_dependency':    t1['pass'],
        'pruning_stability':  t2['pass'],
        'concentration':      t3['pass'],
        'sensitivity':        t4['pass'],
    }
    n_pass  = sum(results.values())
    n_total = len(results)
    log.info('  Test summary: %d/%d pass', n_pass, n_total)
    log.info('')
    for test, passed in results.items():
        log.info('  [%s]  %s', 'PASS' if passed else 'FAIL', test)
    log.info('')

    if all(results.values()):
        verdict = 'PROMOTE_ETD'
        log.info('  RECOMMENDATION: PROMOTE ETD to sizing research candidate.')
        log.info('  All four robustness tests pass.')
        log.info('  Proceed with pruned ETD symbols: %s', ETD_KEPT_SYMS)
    elif not results['tail_dependency']:
        verdict = 'ISOLATE_CTU'
        log.info('  RECOMMENDATION: ISOLATE_CTU. ETD is tail-driven.')
        log.info('  Signal collapses after removing top trades.')
        log.info('  Do not size ETD. Proceed with CTU only.')
    elif not results['pruning_stability']:
        verdict = 'REVERT_PRUNING'
        fwd_pos = t2.get('removed_fwd_positive', [])
        log.info('  RECOMMENDATION: REVERT_PRUNING. Pruning is out-of-sample unstable.')
        log.info('  Removed symbols positive in forward: %s', fwd_pos)
        log.info('  Continue monitoring without pruning. Do not size yet.')
    elif not results['concentration']:
        verdict = 'CAP_EXPOSURE'
        log.info('  RECOMMENDATION: CAP_EXPOSURE. Single symbol dominates PnL.')
        log.info('  Apply max 1 position per symbol before sizing research.')
    else:
        verdict = 'PROMOTE_ETD_CAUTION'
        failed  = [k for k, v in results.items() if not v]
        log.info('  RECOMMENDATION: PROMOTE_ETD with caution.')
        log.info('  Failing sensitivity test: %s', failed)
        log.info('  ETD pruning is fragile to symbol reintroduction.')
        log.info('  Keep pruning as-is; do not expand symbol set.')

    return verdict


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

def write_outputs(
    output_path: Optional[str],
    memo_path: Optional[str],
    t1: dict, t2: dict, t3: dict, t4: dict,
    verdict: str,
    baseline_stats: dict,
) -> None:
    if output_path:
        rows: list[dict] = []
        rows.append({'section': 'verdict', 'key': 'overall', 'value': verdict})
        for test, passed in [
            ('tail_dependency', t1['pass']),
            ('pruning_stability', t2['pass']),
            ('concentration', t3['pass']),
            ('sensitivity', t4['pass']),
        ]:
            rows.append({'section': 'test_results', 'key': test,
                         'value': 'PASS' if passed else 'FAIL'})
        for row in t1.get('rows', []):
            rows.append({'section': 'tail_test', 'key': f'n_removed_{row["n_removed"]}',
                         'value': f'mean={row.get("mean")} sharpe={row.get("sharpe")} max_dd={row.get("max_drawdown")}'})
        for row in t3.get('rows', []):
            rows.append({'section': 'concentration', 'key': row['symbol'],
                         'value': f'pnl_frac={row.get("pnl_frac")} var_frac={row.get("var_frac")}'})
        for row in t4.get('rows', []):
            rows.append({'section': 'sensitivity', 'key': row['symbol_added'],
                         'value': f'sharpe={row.get("sharpe")} drop={row.get("sharpe_drop")} stable={row.get("stable")}'})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fields = ['section', 'key', 'value']
        with open(output_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        log.info('Results written to %s', output_path)

    if memo_path:
        import datetime
        lines: list[str] = [
            'ETD ROBUSTNESS VALIDATION — RECOMMENDATION MEMO',
            f'Generated: {datetime.date.today()}',
            f'Issue #96: ETD tail-removal and forward-split analysis',
            '',
            f'VERDICT: {verdict}',
            '',
            '=' * 72,
            'TEST RESULTS',
            '-' * 40,
            f'  [{"PASS" if t1["pass"] else "FAIL"}]  tail_dependency',
            f'  [{"PASS" if t2["pass"] else "FAIL"}]  pruning_stability',
            f'  [{"PASS" if t3["pass"] else "FAIL"}]  concentration',
            f'  [{"PASS" if t4["pass"] else "FAIL"}]  sensitivity',
            '',
            '=' * 72,
            'TAIL DEPENDENCY TEST',
            '-' * 40,
        ]
        for row in t1.get('rows', []):
            n_rem = row.get('n_removed', 0)
            lbl   = 'baseline' if n_rem == 0 else f'minus_top_{n_rem}'
            lines.append(
                f'  {lbl:<20}  mean={row.get("mean")}  sharpe={row.get("sharpe")}  max_dd={row.get("max_drawdown")}'
            )
        lines += ['', 'Top-3 trades removed:']
        for t in t1.get('top3_trades', []):
            lines.append(f'  {t["entry_ts"]}  {t["symbol"]}  excess=+{t["excess_return"]:.4f}')
        lines += [
            '',
            '=' * 72,
            'FORWARD-SPLIT PRUNING STABILITY',
            '-' * 40,
        ]
        for row in t2.get('rows', []):
            flag = 'PRUNED' if row.get('pruned') else 'kept  '
            lines.append(
                f'  {row["symbol"]:<12}  {flag}  train={row.get("train_mean")}  val={row.get("val_mean")}  fwd={row.get("forward_mean")}  [{row.get("classification")}]'
            )
        if t2.get('removed_fwd_positive'):
            lines.append(f'  WARNING: forward-positive removed symbols: {t2["removed_fwd_positive"]}')
        lines += [
            '',
            '=' * 72,
            'SYMBOL PNL CONCENTRATION',
            '-' * 40,
        ]
        for row in t3.get('rows', []):
            pct_pnl = f'{row["pnl_frac"]*100:+.1f}%' if row.get('pnl_frac') is not None else 'N/A'
            pct_var = f'{row["var_frac"]*100:.1f}%'   if row.get('var_frac') is not None else 'N/A'
            lines.append(f'  {row["symbol"]:<12}  pnl={row.get("pnl"):>8.2f}  pnl%={pct_pnl}  var%={pct_var}')
        lines += [
            '',
            '=' * 72,
            'SENSITIVITY (PRUNING ROBUSTNESS)',
            '-' * 40,
            f'  Baseline Sharpe: {t4.get("baseline_sharpe")}',
        ]
        for row in t4.get('rows', []):
            stable = 'STABLE' if row.get('stable') else 'FRAGILE'
            lines.append(
                f'  +{row["symbol_added"]:<12}  sharpe={row.get("sharpe")}  drop={row.get("sharpe_drop")}  [{stable}]'
            )
        Path(memo_path).parent.mkdir(parents=True, exist_ok=True)
        with open(memo_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        log.info('Memo written to %s', memo_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ETD robustness validation (Issue #96).'
    )
    parser.add_argument('--output', default=None, help='CSV output path')
    parser.add_argument('--memo',   default=None, help='Memo text output path')
    args = parser.parse_args()

    log.info('Connecting to %s', TIMESCALE_DSN or f'{PGHOST}:{PGPORT}/{PGDATABASE}')
    log.info('Loading FX breakout events ...')
    all_events = load_events()
    log.info('Loaded %d state-labeled events', len(all_events))
    if not all_events:
        log.error('No events found.')
        sys.exit(1)

    baselines     = compute_baselines(all_events)
    excess_events = apply_excess_return(all_events, baselines)

    etd_all_events = [
        ev for ev in excess_events
        if ev['symbol'] in COHORT_SYMBOLS
    ]
    log.info('ETD cohort events (all symbols): %d',
             sum(1 for ev in etd_all_events if ev.get('state') == ETD_STATE))

    # Baseline: ETD pruned portfolio
    baseline_trades = simulate_portfolio(etd_all_events, ETD_KEPT_SYMS)
    baseline_stats  = compute_stats(baseline_trades, label='etd_pruned_baseline')
    log.info('Baseline ETD pruned: n=%d  mean=%s  sharpe=%s  max_dd=%s',
             baseline_stats.get('n', 0),
             f'{baseline_stats["mean"]:+.4f}'       if baseline_stats.get('mean')        is not None else 'N/A',
             f'{baseline_stats["sharpe"]:+.4f}'     if baseline_stats.get('sharpe')      is not None else 'N/A',
             f'{baseline_stats["max_drawdown"]:.4f}' if baseline_stats.get('max_drawdown') is not None else 'N/A',
             )

    # Test 1
    t1 = test_tail_dependency(baseline_trades)
    print_test1(t1)

    # Test 2
    t2 = test_forward_split_pruning(etd_all_events)
    print_test2(t2)

    # Test 3
    t3 = test_concentration(baseline_trades)
    print_test3(t3)

    # Test 4
    t4 = test_sensitivity(etd_all_events, baseline_stats.get('sharpe'))
    print_test4(t4)

    # Recommendation
    verdict = print_recommendation(t1, t2, t3, t4, baseline_stats)

    write_outputs(args.output, args.memo, t1, t2, t3, t4, verdict, baseline_stats)


if __name__ == '__main__':
    main()

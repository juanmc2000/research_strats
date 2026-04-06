# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_regime_aware_etd.py
Issue #97: Regime-Aware ETD Validation (USD/CHF Chaos Filter)

Tests whether filtering out CHAOTIC_REGIME trades (gold trailing-12m > X% AND
silver trailing-12m > X%) improves USD/CHF ETD risk-adjusted performance.

Outputs
-------
  --output  results/regime_aware_etd.csv
  --memo    results/regime_aware_etd_memo.txt

Usage
-----
  python scripts/markov_regime_aware_etd.py \
      --output results/regime_aware_etd.csv \
      --memo   results/regime_aware_etd_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Constants ─────────────────────────────────────────────────────────────────

MEDIUM_COST: float = 0.07
ETD_STATE: str = 'EXPANDING_TRENDING_DOWN'
USD_CHF: str = 'USD/CHF'

CHAOTIC_THRESHOLD: float = 20.0          # both metals must exceed this % to be CHAOTIC
SENSITIVITY_THRESHOLDS: list[float] = [15.0, 20.0, 25.0]

TRAIN_END_YEAR: int = 2022               # train = [2015, TRAIN_END_YEAR)
VAL_END_YEAR: int = 2024                 # val   = [TRAIN_END_YEAR, VAL_END_YEAR)
                                         # fwd   = [VAL_END_YEAR, ...)

# Acceptance criteria
SHARPE_IMPROVE_MIN: float = 0.05         # filtered Sharpe - baseline Sharpe >= this
DD_REDUCE_MIN_FRAC: float = 0.20         # max_dd must shrink by >= 20%
MAX_TRADE_REDUCTION: float = 0.40        # filter must not remove > 40% of trades


# ── Database helpers ──────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


def load_etd_trades() -> list[dict]:
    """Load USD/CHF ETD breakout events with exit metadata."""
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                be.symbol,
                be.event_hour_ts,
                be.breakout_direction,
                be.vol_ratio_20_100,
                be.efficiency_20,
                be.realized_return_sd30d,
                be.forward_horizon_hours,
                be.bars_held,
                be.exit_reason,
                be.stop_hit_flag,
                be.vol_ratio_20_100 AS vr,
                be.efficiency_20    AS ef
            FROM features.breakout_events be
            WHERE be.symbol = %s
              AND be.exit_reason IS NOT NULL
              AND be.entry_range_sd_30d > 0
              AND be.vol_ratio_20_100 IS NOT NULL
              AND be.efficiency_20 IS NOT NULL
              AND be.realized_return_sd30d IS NOT NULL
            ORDER BY be.event_hour_ts
        """, (USD_CHF,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_metal_monthly(symbol: str, price_table: str) -> dict:
    """
    Load monthly close prices for a metal symbol.
    Returns {datetime: float} keyed by month-truncated datetime.
    """
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT
                date_trunc('month', mp.date) AS month,
                last(mp.bid_close, mp.date)  AS close_price
            FROM market_data.{price_table} mp
            JOIN market_data.symbols s ON s.id = mp.symbol_id
            WHERE s.symbol = %s
              AND mp.date >= '2014-01-01'
            GROUP BY 1
            ORDER BY 1
        """, (symbol,))
        result = {r['month']: float(r['close_price']) for r in cur.fetchall()}
    conn.close()
    return result


# ── State classification ───────────────────────────────────────────────────────

def vol_bucket(vr: float) -> str:
    if vr < 0.80:
        return 'CONTRACTING'
    if vr <= 1.20:
        return 'NEUTRAL'
    return 'EXPANDING'


def eff_bucket(ef: float) -> str:
    if ef < 0.30:
        return 'CHOPPY'
    if ef <= 0.60:
        return 'MIXED'
    return 'TRENDING'


def classify_state(vr: float, ef: float, direction: str) -> str:
    return f'{vol_bucket(vr)}_{eff_bucket(ef)}_{direction}'


# ── Regime augmentation ───────────────────────────────────────────────────────

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


def augment_regime(
    trades: list[dict],
    gold_prices: dict,
    silver_prices: dict,
    threshold: float,
) -> list[dict]:
    """Add gold_trail_12m, silver_trail_12m, regime_flag to each trade."""
    gold_sorted = sorted(gold_prices)
    silver_sorted = sorted(silver_prices)
    for tr in trades:
        ts = tr['event_hour_ts']
        g = _trail_12m(gold_prices, gold_sorted, ts)
        s = _trail_12m(silver_prices, silver_sorted, ts)
        tr['gold_trail_12m'] = g
        tr['silver_trail_12m'] = s
        if g is not None and s is not None and g > threshold and s > threshold:
            tr['regime_flag'] = 'CHAOTIC'
        else:
            tr['regime_flag'] = 'NORMAL'
    return trades


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> Optional[dict]:
    """Compute full performance stats from a list of augmented trade dicts."""
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


# ── Period split helper ───────────────────────────────────────────────────────

def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


# ── Section 1: Regime segmentation ────────────────────────────────────────────

def regime_segmentation(trades: list[dict]) -> dict[str, Optional[dict]]:
    chaotic = [tr for tr in trades if tr['regime_flag'] == 'CHAOTIC']
    normal  = [tr for tr in trades if tr['regime_flag'] == 'NORMAL']
    return {
        'all':     compute_stats(trades),
        'normal':  compute_stats(normal),
        'chaotic': compute_stats(chaotic),
    }


# ── Section 2: Time-split validation ──────────────────────────────────────────

def time_split_comparison(
    baseline: list[dict],
    filtered: list[dict],
) -> dict[str, dict]:
    """Return per-split stats for baseline and filtered."""
    result = {}
    for label, base_split, filt_split in [
        ('full',  baseline, filtered),
        ('train', [tr for tr in baseline if tr['year'] < TRAIN_END_YEAR],
                  [tr for tr in filtered if tr['year'] < TRAIN_END_YEAR]),
        ('val',   [tr for tr in baseline if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
                  [tr for tr in filtered if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR]),
        ('fwd',   [tr for tr in baseline if tr['year'] >= VAL_END_YEAR],
                  [tr for tr in filtered if tr['year'] >= VAL_END_YEAR]),
    ]:
        result[label] = {
            'baseline': compute_stats(base_split),
            'filtered': compute_stats(filt_split),
        }
    return result


# ── Section 3: Sensitivity checks ─────────────────────────────────────────────

def sensitivity_checks(
    trades: list[dict],
    gold_prices: dict,
    silver_prices: dict,
) -> list[dict]:
    rows = []
    for thresh in SENSITIVITY_THRESHOLDS:
        augmented = augment_regime(
            [dict(tr) for tr in trades],
            gold_prices,
            silver_prices,
            thresh,
        )
        filt = [tr for tr in augmented if tr['regime_flag'] == 'NORMAL']
        s_base = compute_stats(augmented)
        s_filt = compute_stats(filt)
        n_chaotic = sum(1 for tr in augmented if tr['regime_flag'] == 'CHAOTIC')
        rows.append(dict(
            threshold=thresh,
            n_total=len(augmented),
            n_chaotic=n_chaotic,
            n_filtered=len(filt),
            frac_removed=n_chaotic / len(augmented) if augmented else 0.0,
            baseline_sharpe=s_base['sharpe'] if s_base else None,
            filtered_sharpe=s_filt['sharpe'] if s_filt else None,
            baseline_max_dd=s_base['max_dd'] if s_base else None,
            filtered_max_dd=s_filt['max_dd'] if s_filt else None,
        ))
    return rows


# ── Section 4: Concentration impact ───────────────────────────────────────────

def concentration_impact(
    baseline: list[dict],
    filtered: list[dict],
) -> dict:
    def usdchf_frac(trades: list[dict]) -> float:
        total_pnl = sum(tr['excess'] for tr in trades)
        sym_pnl = sum(tr['excess'] for tr in trades if tr['symbol'] == USD_CHF)
        return sym_pnl / total_pnl if total_pnl else 0.0

    return dict(
        baseline_usdchf_frac=usdchf_frac(baseline),
        filtered_usdchf_frac=usdchf_frac(filtered),
        baseline_n=len(baseline),
        filtered_n=len(filtered),
    )


# ── Section 5: Trade behaviour diagnostics ────────────────────────────────────

def trade_behaviour_diagnostics(
    baseline: list[dict],
    filtered: list[dict],
) -> dict:
    def diag(trades: list[dict]) -> dict:
        n = len(trades)
        if n == 0:
            return {}
        # bars_held is the holding duration
        bars = [tr['bars_held'] for tr in trades if tr.get('bars_held') is not None]
        avg_bars = sum(bars) / len(bars) if bars else float('nan')
        stop_outs = sum(1 for tr in trades if tr.get('exit_reason') == 'trailing_stop')
        vr_vals = [float(tr['vr']) for tr in trades if tr.get('vr') is not None]
        avg_vr = sum(vr_vals) / len(vr_vals) if vr_vals else float('nan')
        ef_vals = [float(tr['ef']) for tr in trades if tr.get('ef') is not None]
        avg_ef = sum(ef_vals) / len(ef_vals) if ef_vals else float('nan')
        return dict(
            n=n,
            avg_bars_held=avg_bars,
            stop_out_rate=stop_outs / n,
            avg_vol_ratio=avg_vr,
            avg_efficiency=avg_ef,
        )
    return {
        'baseline': diag(baseline),
        'filtered': diag(filtered),
        'chaotic':  diag([tr for tr in baseline if tr['regime_flag'] == 'CHAOTIC']),
        'normal':   diag([tr for tr in baseline if tr['regime_flag'] == 'NORMAL']),
    }


# ── Section 6: Acceptance criteria ────────────────────────────────────────────

def evaluate_criteria(
    baseline_stats: dict,
    filtered_stats: dict,
    splits: dict,
    sensitivity: list[dict],
    n_total: int,
    n_filtered: int,
) -> dict:
    scores: dict[str, bool] = {}
    details: dict[str, str] = {}

    # C1: Sharpe improves by >= 0.05
    delta_sharpe = filtered_stats['sharpe'] - baseline_stats['sharpe']
    scores['sharpe_improves'] = delta_sharpe >= SHARPE_IMPROVE_MIN
    details['sharpe_improves'] = (
        f"baseline={baseline_stats['sharpe']:+.4f}  "
        f"filtered={filtered_stats['sharpe']:+.4f}  "
        f"delta={delta_sharpe:+.4f}  "
        f"(need >= +{SHARPE_IMPROVE_MIN})"
    )

    # C2: Max drawdown reduces by >= 20%
    dd_reduction = (baseline_stats['max_dd'] - filtered_stats['max_dd']) / baseline_stats['max_dd']
    scores['drawdown_reduces'] = dd_reduction >= DD_REDUCE_MIN_FRAC
    details['drawdown_reduces'] = (
        f"baseline={baseline_stats['max_dd']:.2f}  "
        f"filtered={filtered_stats['max_dd']:.2f}  "
        f"reduction={dd_reduction:.1%}  "
        f"(need >= {DD_REDUCE_MIN_FRAC:.0%})"
    )

    # C3: Left tail improves
    tail_improves = filtered_stats['p_lt_neg1'] < baseline_stats['p_lt_neg1']
    scores['left_tail_improves'] = tail_improves
    details['left_tail_improves'] = (
        f"baseline P(<-1)={baseline_stats['p_lt_neg1']:.3f}  "
        f"filtered P(<-1)={filtered_stats['p_lt_neg1']:.3f}"
    )

    # C4: Improvement holds in forward split
    fwd_b = splits['fwd']['baseline']
    fwd_f = splits['fwd']['filtered']
    if fwd_b and fwd_f:
        fwd_holds = fwd_f['sharpe'] > fwd_b['sharpe']
        scores['fwd_holds'] = fwd_holds
        details['fwd_holds'] = (
            f"fwd baseline sharpe={fwd_b['sharpe']:+.4f}  "
            f"fwd filtered sharpe={fwd_f['sharpe']:+.4f}"
        )
    else:
        scores['fwd_holds'] = False
        details['fwd_holds'] = 'insufficient forward data'

    # C5: Threshold stability (range of filtered Sharpe across 15/20/25%)
    filt_sharpes = [r['filtered_sharpe'] for r in sensitivity if r['filtered_sharpe'] is not None]
    if len(filt_sharpes) >= 2:
        sharpe_range = max(filt_sharpes) - min(filt_sharpes)
        scores['threshold_stable'] = sharpe_range < 0.10
        details['threshold_stable'] = (
            f"Sharpe range across thresholds={sharpe_range:.4f}  (need < 0.10)"
        )
    else:
        scores['threshold_stable'] = False
        details['threshold_stable'] = 'insufficient sensitivity data'

    # C6: Trade count does not collapse
    frac_removed = (n_total - n_filtered) / n_total if n_total else 1.0
    scores['trade_count_ok'] = frac_removed <= MAX_TRADE_REDUCTION
    details['trade_count_ok'] = (
        f"removed {n_total-n_filtered}/{n_total} = {frac_removed:.1%}  "
        f"(limit {MAX_TRADE_REDUCTION:.0%})"
    )

    n_pass = sum(1 for v in scores.values() if v)
    n_total_crit = len(scores)

    if n_pass == n_total_crit:
        verdict = 'PASS_REGIME_FILTER_VALIDATED'
    elif n_pass >= 4:
        verdict = 'PASS_WITH_CAVEATS'
    else:
        verdict = 'FAIL_REGIME_HYPOTHESIS_REJECTED'

    return dict(scores=scores, details=details, n_pass=n_pass, n_total=n_total_crit, verdict=verdict)


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(
    output_path: str,
    baseline_stats: dict,
    filtered_stats: dict,
    regime_seg: dict,
    splits: dict,
    sensitivity: list[dict],
    conc: dict,
    behaviour: dict,
    criteria: dict,
) -> None:
    rows: list[tuple] = []

    def stat_rows(prefix: str, s: Optional[dict]) -> None:
        if s is None:
            return
        for k, v in s.items():
            rows.append((prefix, k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    stat_rows('baseline', baseline_stats)
    stat_rows('filtered', filtered_stats)

    for regime, s in regime_seg.items():
        stat_rows(f'regime_{regime}', s)

    for split_name, d in splits.items():
        stat_rows(f'split_{split_name}_baseline', d['baseline'])
        stat_rows(f'split_{split_name}_filtered', d['filtered'])

    for row in sensitivity:
        pref = f'sensitivity_{row["threshold"]:.0f}pct'
        for k, v in row.items():
            if k == 'threshold':
                continue
            rows.append((pref, k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    rows.append(('concentration', 'baseline_usdchf_frac', f'{conc["baseline_usdchf_frac"]:.4f}'))
    rows.append(('concentration', 'filtered_usdchf_frac', f'{conc["filtered_usdchf_frac"]:.4f}'))
    rows.append(('concentration', 'baseline_n', str(conc['baseline_n'])))
    rows.append(('concentration', 'filtered_n', str(conc['filtered_n'])))

    for group, diag in behaviour.items():
        for k, v in diag.items():
            rows.append((f'behaviour_{group}', k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    rows.append(('verdict', 'result', criteria['verdict']))
    rows.append(('verdict', 'n_pass', str(criteria['n_pass'])))
    rows.append(('verdict', 'n_total', str(criteria['n_total'])))
    for k, v in criteria['scores'].items():
        rows.append(('criterion', k, 'PASS' if v else 'FAIL'))
    for k, v in criteria['details'].items():
        rows.append(('criterion_detail', k, v))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['section', 'key', 'value'])
        writer.writerows(rows)
    print(f'CSV written: {output_path}')


def write_memo(
    memo_path: str,
    baseline_stats: dict,
    filtered_stats: dict,
    regime_seg: dict,
    splits: dict,
    sensitivity: list[dict],
    conc: dict,
    behaviour: dict,
    criteria: dict,
) -> None:
    lines = []
    sep = '=' * 72

    def s(label: str) -> str:
        return f'  {label}'

    lines += [
        'REGIME-AWARE ETD VALIDATION — USD/CHF CHAOS FILTER',
        'Issue #97',
        'Generated: 2026-03-31',
        '',
        sep,
        f'VERDICT: {criteria["verdict"]}  ({criteria["n_pass"]}/{criteria["n_total"]} criteria pass)',
        sep,
        '',
        'REGIME DEFINITION',
        '-' * 40,
        f'  CHAOTIC : gold_trail_12m > {CHAOTIC_THRESHOLD:.0f}% AND silver_trail_12m > {CHAOTIC_THRESHOLD:.0f}%',
        '  NORMAL  : everything else',
        '',
        'ACCEPTANCE CRITERIA',
        '-' * 40,
    ]
    for k, passed in criteria['scores'].items():
        tag = '[PASS]' if passed else '[FAIL]'
        lines.append(f'  {tag}  {k}')
        lines.append(f'          {criteria["details"][k]}')

    lines += [
        '',
        'REGIME SEGMENTATION',
        '-' * 40,
    ]
    for regime in ['all', 'normal', 'chaotic']:
        s_obj = regime_seg[regime]
        if s_obj:
            lines.append(
                f'  {regime:<10} n={s_obj["n"]:>4}  '
                f'mean={s_obj["mean"]:>+8.4f}  '
                f'sharpe={s_obj["sharpe"]:>+8.4f}  '
                f'win%={s_obj["win_rate"]:.1%}  '
                f'max_dd={s_obj["max_dd"]:.2f}'
            )

    lines += [
        '',
        'BASELINE vs FILTERED (full history)',
        '-' * 40,
        f'  {"":12} {"n":>5} {"mean":>8} {"sharpe":>8} {"win%":>6} {"payoff":>7} {"max_dd":>8} {"P(<-1)":>8}',
    ]
    for lbl, s_obj in [('baseline', baseline_stats), ('filtered', filtered_stats)]:
        if s_obj:
            lines.append(
                f'  {lbl:<12} {s_obj["n"]:>5} {s_obj["mean"]:>+8.4f} '
                f'{s_obj["sharpe"]:>+8.4f} {s_obj["win_rate"]:>5.1%} '
                f'{s_obj["payoff"]:>7.3f} {s_obj["max_dd"]:>8.2f} '
                f'{s_obj["p_lt_neg1"]:>8.3f}'
            )

    lines += ['', 'TIME-SPLIT COMPARISON', '-' * 40]
    header = f'  {"split":<8} {"base_n":>7} {"base_sh":>8} {"filt_n":>7} {"filt_sh":>8} {"delta_sh":>9}'
    lines.append(header)
    for split_name in ['full', 'train', 'val', 'fwd']:
        d = splits[split_name]
        b = d['baseline']
        f = d['filtered']
        b_n = b['n'] if b else 0
        b_sh = b['sharpe'] if b else float('nan')
        f_n = f['n'] if f else 0
        f_sh = f['sharpe'] if f else float('nan')
        delta = f_sh - b_sh if b and f else float('nan')
        lines.append(
            f'  {split_name:<8} {b_n:>7} {b_sh:>+8.4f} {f_n:>7} {f_sh:>+8.4f} {delta:>+9.4f}'
        )

    lines += ['', 'SENSITIVITY CHECK (threshold variants)', '-' * 40]
    lines.append(f'  {"thresh":>7} {"n_chaotic":>10} {"frac_rem":>9} {"base_sh":>8} {"filt_sh":>8} {"delta_sh":>9}')
    for row in sensitivity:
        b_sh = row['baseline_sharpe'] or float('nan')
        f_sh = row['filtered_sharpe'] or float('nan')
        delta = f_sh - b_sh
        lines.append(
            f'  {row["threshold"]:>6.0f}%  {row["n_chaotic"]:>9}  '
            f'{row["frac_removed"]:>8.1%}  {b_sh:>+8.4f}  {f_sh:>+8.4f}  {delta:>+9.4f}'
        )

    lines += [
        '',
        'CONCENTRATION IMPACT',
        '-' * 40,
        f'  USD/CHF PnL fraction — baseline: {conc["baseline_usdchf_frac"]:.1%}',
        f'  USD/CHF PnL fraction — filtered: {conc["filtered_usdchf_frac"]:.1%}',
    ]

    beh_b = behaviour['baseline']
    beh_f = behaviour['filtered']
    beh_c = behaviour['chaotic']
    beh_n = behaviour['normal']
    lines += [
        '',
        'TRADE BEHAVIOUR DIAGNOSTICS',
        '-' * 40,
        f'  {"":12} {"avg_bars":>9} {"stop_out%":>10} {"avg_vr":>8} {"avg_ef":>8}',
    ]
    for lbl, beh in [('baseline', beh_b), ('filtered', beh_f), ('chaotic', beh_c), ('normal', beh_n)]:
        if not beh:
            continue
        lines.append(
            f'  {lbl:<12} '
            f'{beh.get("avg_bars_held", float("nan")):>9.1f} '
            f'{beh.get("stop_out_rate", float("nan")):>9.1%} '
            f'{beh.get("avg_vol_ratio", float("nan")):>8.4f} '
            f'{beh.get("avg_efficiency", float("nan")):>8.4f}'
        )

    lines += [
        '',
        'RECOMMENDATION',
        '-' * 40,
    ]
    verdict = criteria['verdict']
    if verdict == 'PASS_REGIME_FILTER_VALIDATED':
        lines += [
            '  REGIME FILTER VALIDATED. Proceed to:',
            '  1. Generalise regime detection with non-metals proxies',
            '  2. Integrate filter into multi-symbol ETD sleeve',
            '  3. Proceed to position sizing research',
        ]
    elif verdict == 'PASS_WITH_CAVEATS':
        lines += [
            '  FILTER SHOWS BENEFIT WITH CAVEATS.',
            '  Review failed criteria before generalising.',
            f'  Failed: {[k for k,v in criteria["scores"].items() if not v]}',
        ]
    else:
        lines += [
            '  REGIME HYPOTHESIS REJECTED.',
            '  Filter does not robustly improve performance.',
            '  Revisit signal definition or state classification.',
        ]
    lines.append('')

    os.makedirs(os.path.dirname(memo_path) or '.', exist_ok=True)
    with open(memo_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Memo written: {memo_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Issue #97: Regime-Aware ETD Validation')
    parser.add_argument('--output', default='results/regime_aware_etd.csv')
    parser.add_argument('--memo',   default='results/regime_aware_etd_memo.txt')
    args = parser.parse_args()

    print('Loading USD/CHF ETD trades...')
    raw_trades = load_etd_trades()

    # Classify state and filter to ETD only
    all_events = []
    bl_groups: dict[tuple, list[float]] = defaultdict(list)
    for tr in raw_trades:
        vr = float(tr['vr']); ef = float(tr['ef']); d = tr['breakout_direction']
        if d not in ('UP', 'DOWN'):
            continue
        tr['state'] = classify_state(vr, ef, d)
        bl_groups[(tr['symbol'], d)].append(float(tr['realized_return_sd30d']))
        all_events.append(tr)

    baselines = {k: sum(v) / len(v) for k, v in bl_groups.items()}

    etd_trades = [tr for tr in all_events if tr['state'] == ETD_STATE]
    for tr in etd_trades:
        bl = baselines.get((tr['symbol'], tr['breakout_direction']), 0.0)
        tr['excess'] = float(tr['realized_return_sd30d']) - bl - MEDIUM_COST
        tr['year'] = tr['event_hour_ts'].year

    print(f'  ETD trades: {len(etd_trades)}')

    print('Loading XAU/USD monthly prices (minute_prices)...')
    gold_prices = load_metal_monthly('XAU/USD', 'minute_prices')
    print(f'  Gold months: {len(gold_prices)}')

    print('Loading XAG/USD monthly prices (hourly_prices)...')
    silver_prices = load_metal_monthly('XAG/USD', 'hourly_prices')
    print(f'  Silver months: {len(silver_prices)}')

    print('Augmenting trades with regime flags...')
    etd_trades = augment_regime(etd_trades, gold_prices, silver_prices, CHAOTIC_THRESHOLD)

    baseline = etd_trades
    filtered = [tr for tr in etd_trades if tr['regime_flag'] == 'NORMAL']
    n_chaotic = sum(1 for tr in etd_trades if tr['regime_flag'] == 'CHAOTIC')
    print(f'  Chaotic trades: {n_chaotic}  Normal trades: {len(filtered)}')

    print('Computing statistics...')
    baseline_stats = compute_stats(baseline)
    filtered_stats = compute_stats(filtered)

    print('Running regime segmentation...')
    regime_seg = regime_segmentation(baseline)

    print('Running time-split comparison...')
    splits = time_split_comparison(baseline, filtered)

    print('Running sensitivity checks...')
    sensitivity = sensitivity_checks(
        [dict(tr) for tr in etd_trades],
        gold_prices,
        silver_prices,
    )

    print('Computing concentration impact...')
    conc = concentration_impact(baseline, filtered)

    print('Computing trade behaviour diagnostics...')
    behaviour = trade_behaviour_diagnostics(baseline, filtered)

    print('Evaluating acceptance criteria...')
    criteria = evaluate_criteria(
        baseline_stats,
        filtered_stats,
        splits,
        sensitivity,
        n_total=len(baseline),
        n_filtered=len(filtered),
    )

    print(f'\nVERDICT: {criteria["verdict"]} ({criteria["n_pass"]}/{criteria["n_total"]})')
    for k, v in criteria['scores'].items():
        tag = 'PASS' if v else 'FAIL'
        print(f'  [{tag}]  {k}  — {criteria["details"][k]}')

    print('\nWriting outputs...')
    write_csv(
        args.output,
        baseline_stats, filtered_stats,
        regime_seg, splits, sensitivity,
        conc, behaviour, criteria,
    )
    write_memo(
        args.memo,
        baseline_stats, filtered_stats,
        regime_seg, splits, sensitivity,
        conc, behaviour, criteria,
    )


if __name__ == '__main__':
    main()

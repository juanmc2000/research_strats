# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_atr_stability_filter.py
Issue #98: ATR Stability Filter via Linear Regression (Regime Predictability Test)

Tests whether volatility instability (detected via rolling linear regression on daily
ATR) can be used as an ex-ante filter to improve USD/CHF ETD trade quality.

Three strategies compared:
  A) Baseline        -- all ETD trades, no filter
  B) Exit filter     -- trades where instability triggers during the holding window
                        receive an early exit (return = 0, cost-adjusted)
  C) Entry+exit      -- trades where instability is already active at entry are
                        skipped entirely

Acceptance criteria (all must pass for PASS verdict):
  1. Sharpe improves >= +0.05 (vs baseline)
  2. Max drawdown reduces >= 15%
  3. Left tail P(< -1 SD) decreases
  4. Effect persists in forward period (2024+)
  5. Results consistent using residual std AND standard error

Usage
-----
  python scripts/markov_atr_stability_filter.py \
      --output results/atr_stability_filter.csv \
      --memo   results/atr_stability_filter_memo.txt
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
ETD_STATE: str = 'EXPANDING_TRENDING_DOWN'
USD_CHF: str = 'USD/CHF'

ATR_WINDOW: int = 30            # rolling regression window (days)
FORECAST_HORIZON: int = 15      # days ahead to forecast
INSTABILITY_K: float = 2.0      # |forecast - current| > k * sigma → UNSTABLE
PERSIST_DAYS: int = 5           # consecutive unstable days required to trigger
HOLD_WINDOW_DAYS: int = 15      # trading days to check for instability post-entry

# Sensitivity grid — informational only, not parameter tuning
SENSITIVITY_K_VALUES: list[float] = [0.50, 0.75, 1.00, 1.50, 2.00]

TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024

# Acceptance criteria
SHARPE_IMPROVE_MIN: float = 0.05
DD_REDUCE_MIN_FRAC: float = 0.15
SIGMA_CONSISTENCY_SHARPE_DIFF: float = 0.05   # max allowed gap between σ_resid and SE results


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


def load_daily_atr(symbol: str) -> list[dict]:
    """Aggregate hourly bid_high/bid_low to daily ATR = (daily_high - daily_low)."""
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                DATE(hp.date AT TIME ZONE 'UTC') AS trading_day,
                MAX(hp.bid_high) AS day_high,
                MIN(hp.bid_low)  AS day_low
            FROM market_data.hourly_prices hp
            JOIN market_data.symbols s ON s.id = hp.symbol_id
            WHERE s.symbol = %s
              AND hp.date >= '2014-07-01'
            GROUP BY 1
            ORDER BY 1
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    for r in rows:
        r['atr'] = float(r['day_high']) - float(r['day_low'])
    return rows


def load_etd_trades(symbol: str) -> list[dict]:
    """Load ETD breakout events for a single symbol."""
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
                be.bars_held,
                be.exit_reason,
                be.max_favorable_excursion_sd30d,
                be.max_adverse_excursion_sd30d
            FROM features.breakout_events be
            WHERE be.symbol = %s
              AND be.exit_reason IS NOT NULL
              AND be.entry_range_sd_30d > 0
              AND be.vol_ratio_20_100 IS NOT NULL
              AND be.efficiency_20 IS NOT NULL
              AND be.realized_return_sd30d IS NOT NULL
            ORDER BY be.event_hour_ts
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── State classification ────────────────────────────────────────────────────────

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


# ── Linear regression ──────────────────────────────────────────────────────────

def linear_regression(
    y: list[float],
) -> tuple[float, float, float, float]:
    """
    Fit ATR_t = alpha + beta * t for t in {0, 1, ..., n-1}.

    Returns (alpha, beta, sigma_resid, se_slope) where:
      sigma_resid = sqrt(RSS / (n-2))      residual standard deviation
      se_slope    = sigma_resid / sqrt(Sxx) standard error of the slope
    """
    n = len(y)
    if n < 3:
        nan = float('nan')
        return nan, nan, nan, nan
    x = list(range(n))
    x_mean = (n - 1) / 2.0       # exact for 0..n-1
    y_mean = sum(y) / n
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    if sxx == 0.0:
        nan = float('nan')
        return nan, nan, nan, nan
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    rss = sum((yi - (alpha + beta * xi)) ** 2 for xi, yi in zip(x, y))
    sigma_resid = math.sqrt(rss / (n - 2))
    se_slope = sigma_resid / math.sqrt(sxx)
    return alpha, beta, sigma_resid, se_slope


# ── Rolling ATR regression + instability flags ─────────────────────────────────

def build_instability_series(
    daily_atr: list[dict],
    use_se: bool = False,
) -> dict[date, dict]:
    """
    For each trading day with sufficient history, compute:
      - alpha, beta, sigma (residual std or SE)
      - forecast_15d ATR
      - deviation from current ATR
      - raw instability flag (deviation > k*sigma)
      - triggered flag (raw flag persisted >= PERSIST_DAYS consecutive days)

    Returns {trading_day: {...}} dict.
    """
    result: dict[date, dict] = {}
    n_total = len(daily_atr)

    # precompute raw flags first, then apply persistence filter
    raw_flags: dict[date, bool] = {}
    reg_data: dict[date, dict] = {}

    for i in range(ATR_WINDOW - 1, n_total):
        window = daily_atr[i - ATR_WINDOW + 1: i + 1]
        y = [r['atr'] for r in window]
        day = daily_atr[i]['trading_day']
        alpha, beta, sigma_resid, se_slope = linear_regression(y)
        sigma = se_slope if use_se else sigma_resid
        if math.isnan(alpha) or sigma == 0.0:
            raw_flags[day] = False
            continue
        # forecast ATR at t = (ATR_WINDOW - 1) + FORECAST_HORIZON
        x_forecast = float(ATR_WINDOW - 1 + FORECAST_HORIZON)
        forecast_15d = alpha + beta * x_forecast
        current_atr = daily_atr[i]['atr']
        deviation = abs(forecast_15d - current_atr)
        raw_unstable = deviation > INSTABILITY_K * sigma
        raw_flags[day] = raw_unstable
        reg_data[day] = dict(
            alpha=alpha,
            beta=beta,
            sigma_resid=sigma_resid,
            se_slope=se_slope,
            forecast_15d=forecast_15d,
            current_atr=current_atr,
            deviation=deviation,
            threshold=INSTABILITY_K * sigma,
        )

    # apply persistence filter: triggered only after PERSIST_DAYS consecutive raw unstable
    days_sorted = sorted(raw_flags)
    consecutive = 0
    for d in days_sorted:
        if raw_flags[d]:
            consecutive += 1
        else:
            consecutive = 0
        triggered = (consecutive >= PERSIST_DAYS)
        rd = reg_data.get(d, {})
        result[d] = dict(
            raw_unstable=raw_flags[d],
            consecutive_unstable=consecutive,
            triggered=triggered,
            **rd,
        )
    return result


# ── Trade annotation ───────────────────────────────────────────────────────────

def annotate_trades(
    trades: list[dict],
    instability: dict[date, dict],
    trading_days_sorted: list[date],
) -> list[dict]:
    """
    For each trade add:
      entry_date           -- date of entry_hour_ts
      unstable_at_entry    -- instability.triggered on entry_date
      unstable_in_window   -- any triggered day in first HOLD_WINDOW_DAYS after entry
      n_unstable_in_window -- count of triggered days in that window
    """
    # build a lookup: date -> index in sorted trading days
    day_index = {d: i for i, d in enumerate(trading_days_sorted)}

    for tr in trades:
        entry_dt = tr['event_hour_ts'].date()
        # find nearest trading day at or before entry
        nearest = None
        for d in reversed(trading_days_sorted):
            if d <= entry_dt:
                nearest = d
                break
        tr['entry_date'] = nearest
        if nearest is None or nearest not in instability:
            tr['unstable_at_entry'] = False
            tr['unstable_in_window'] = False
            tr['n_unstable_in_window'] = 0
            continue

        tr['unstable_at_entry'] = instability[nearest]['triggered']

        # check forward window
        idx = day_index.get(nearest, 0)
        window_days = trading_days_sorted[idx: idx + HOLD_WINDOW_DAYS]
        n_unstable = sum(1 for d in window_days if instability.get(d, {}).get('triggered', False))
        tr['unstable_in_window'] = n_unstable >= PERSIST_DAYS
        tr['n_unstable_in_window'] = n_unstable
    return trades


# ── Strategy simulation ─────────────────────────────────────────────────────────

def apply_strategies(trades: list[dict]) -> dict[str, list[dict]]:
    """
    A) baseline    -- all trades, original excess returns
    B) exit        -- trades where instability triggers during hold get return=0-MEDIUM_COST
    C) entry_exit  -- trades where instability already active at entry are skipped
    """
    result: dict[str, list[dict]] = {'A': [], 'B': [], 'C': []}
    for tr in trades:
        # Strategy A: always include
        result['A'].append({**tr, 'strat_excess': tr['excess']})

        # Strategy B: unstable during hold window → early exit (cost only, no market P&L)
        if tr['unstable_in_window']:
            result['B'].append({**tr, 'strat_excess': -MEDIUM_COST})
        else:
            result['B'].append({**tr, 'strat_excess': tr['excess']})

        # Strategy C: unstable at entry → skip entirely
        if not tr['unstable_at_entry']:
            result['C'].append({**tr, 'strat_excess': tr['excess']})

    return result


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict], key: str = 'strat_excess') -> Optional[dict]:
    vals = [tr[key] for tr in trades]
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
        if peak - cum > max_dd:
            max_dd = peak - cum
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


def time_splits(
    strat_trades: dict[str, list[dict]],
) -> dict[str, dict[str, Optional[dict]]]:
    """Return per-split stats for each strategy."""
    def split(trades: list[dict], y0: int, y1: int) -> list[dict]:
        return [tr for tr in trades if y0 <= tr['event_hour_ts'].year < y1]

    periods = {
        'full':  (0,               9999),
        'train': (0,               TRAIN_END_YEAR),
        'val':   (TRAIN_END_YEAR,  VAL_END_YEAR),
        'fwd':   (VAL_END_YEAR,    9999),
    }
    result: dict[str, dict[str, Optional[dict]]] = {}
    for period, (y0, y1) in periods.items():
        result[period] = {
            strat: compute_stats(split(trades, y0, y1))
            for strat, trades in strat_trades.items()
        }
    return result


# ── Instability diagnostics ────────────────────────────────────────────────────

def instability_diagnostics(instability: dict[date, dict]) -> dict:
    """Compute % unstable, avg duration of unstable episodes, etc."""
    days_sorted = sorted(instability)
    if not days_sorted:
        return {}
    total = len(days_sorted)
    n_triggered = sum(1 for d in days_sorted if instability[d]['triggered'])
    n_raw = sum(1 for d in days_sorted if instability[d]['raw_unstable'])

    # compute episode durations
    episodes: list[int] = []
    current_len = 0
    for d in days_sorted:
        if instability[d]['triggered']:
            current_len += 1
        else:
            if current_len > 0:
                episodes.append(current_len)
            current_len = 0
    if current_len > 0:
        episodes.append(current_len)

    avg_episode = sum(episodes) / len(episodes) if episodes else 0.0
    max_episode = max(episodes) if episodes else 0

    return dict(
        total_days=total,
        n_triggered=n_triggered,
        n_raw_unstable=n_raw,
        pct_triggered=n_triggered / total,
        pct_raw_unstable=n_raw / total,
        n_episodes=len(episodes),
        avg_episode_days=avg_episode,
        max_episode_days=max_episode,
    )


# ── Sensitivity diagnostic ────────────────────────────────────────────────────

def sensitivity_diagnostic(
    daily_atr: list[dict],
    etd_trades: list[dict],
    trading_days_sorted: list[date],
) -> list[dict]:
    """
    Informational sweep over k values to characterise why the filter fails.
    Does NOT tune thresholds — purely documents the mechanism.
    """
    rows = []
    for k in SENSITIVITY_K_VALUES:
        for persist in (1, PERSIST_DAYS):
            instab = build_instability_series_k(daily_atr, k, persist)
            td = annotate_trades(
                [dict(tr) for tr in etd_trades], instab, trading_days_sorted
            )
            strats = apply_strategies(td)
            n_unstable_at_entry = sum(1 for t in td if t['unstable_at_entry'])
            n_unstable_in_window = sum(1 for t in td if t['unstable_in_window'])
            # % of days triggered
            n_trig = sum(1 for d in instab if instab[d]['triggered'])
            pct_trig = n_trig / len(instab) if instab else 0.0
            s_a = compute_stats(strats['A'])
            s_b = compute_stats(strats['B'])
            s_c = compute_stats(strats['C'])
            rows.append(dict(
                k=k,
                persist=persist,
                pct_days_triggered=pct_trig,
                n_unstable_at_entry=n_unstable_at_entry,
                n_unstable_in_window=n_unstable_in_window,
                sharpe_A=s_a['sharpe'] if s_a else float('nan'),
                sharpe_B=s_b['sharpe'] if s_b else float('nan'),
                sharpe_C=s_c['sharpe'] if s_c else float('nan'),
                max_dd_A=s_a['max_dd'] if s_a else float('nan'),
                max_dd_B=s_b['max_dd'] if s_b else float('nan'),
                max_dd_C=s_c['max_dd'] if s_c else float('nan'),
            ))
    return rows


def build_instability_series_k(
    daily_atr: list[dict],
    k: float,
    persist: int,
) -> dict[date, dict]:
    """build_instability_series with custom k and persist values."""
    raw_flags: dict[date, bool] = {}
    reg_data: dict[date, dict] = {}
    n_total = len(daily_atr)

    for i in range(ATR_WINDOW - 1, n_total):
        window = daily_atr[i - ATR_WINDOW + 1: i + 1]
        y = [r['atr'] for r in window]
        day = daily_atr[i]['trading_day']
        alpha, beta, sigma_resid, _ = linear_regression(y)
        if math.isnan(alpha) or sigma_resid == 0.0:
            raw_flags[day] = False
            continue
        x_forecast = float(ATR_WINDOW - 1 + FORECAST_HORIZON)
        forecast_15d = alpha + beta * x_forecast
        current_atr = daily_atr[i]['atr']
        deviation = abs(forecast_15d - current_atr)
        raw_flags[day] = deviation > k * sigma_resid
        reg_data[day] = dict(forecast_15d=forecast_15d, current_atr=current_atr, deviation=deviation)

    result: dict[date, dict] = {}
    days_sorted = sorted(raw_flags)
    consecutive = 0
    for d in days_sorted:
        if raw_flags[d]:
            consecutive += 1
        else:
            consecutive = 0
        result[d] = dict(
            raw_unstable=raw_flags[d],
            consecutive_unstable=consecutive,
            triggered=(consecutive >= persist),
            **reg_data.get(d, {}),
        )
    return result


# ── Sigma consistency check ────────────────────────────────────────────────────

def sigma_consistency(
    trades_base: list[dict],
    instability_resid: dict[date, dict],
    instability_se: dict[date, dict],
    trading_days_sorted: list[date],
) -> dict:
    """
    Re-run strategies using SE instead of σ_resid and compare Sharpe.
    """
    trades_se = annotate_trades(
        [dict(tr) for tr in trades_base],
        instability_se,
        trading_days_sorted,
    )
    strats_se = apply_strategies(trades_se)

    result = {}
    for strat in ('B', 'C'):
        s_resid = compute_stats(apply_strategies(
            annotate_trades([dict(tr) for tr in trades_base], instability_resid, trading_days_sorted)
        )[strat])
        s_se = compute_stats(strats_se[strat])
        result[strat] = dict(
            sharpe_resid=s_resid['sharpe'] if s_resid else float('nan'),
            sharpe_se=s_se['sharpe'] if s_se else float('nan'),
            delta=abs((s_resid['sharpe'] if s_resid else 0) - (s_se['sharpe'] if s_se else 0)),
        )
    return result


# ── Acceptance criteria ─────────────────────────────────────────────────────────

def evaluate_criteria(
    splits: dict[str, dict[str, Optional[dict]]],
    sigma_check: dict,
) -> dict:
    baseline_full = splits['full']['A']
    b_full = splits['full']['B']
    c_full = splits['full']['C']
    baseline_fwd = splits['fwd']['A']
    b_fwd = splits['fwd']['B']
    c_fwd = splits['fwd']['C']

    best_strat = 'B'
    best_full = b_full
    best_fwd = b_fwd
    if c_full and b_full:
        if c_full['sharpe'] > b_full['sharpe']:
            best_strat = 'C'
            best_full = c_full
            best_fwd = c_fwd

    scores: dict[str, bool] = {}
    details: dict[str, str] = {}

    # C1: Sharpe improves >= 0.05
    if baseline_full and best_full:
        delta_sh = best_full['sharpe'] - baseline_full['sharpe']
        scores['sharpe_improves'] = delta_sh >= SHARPE_IMPROVE_MIN
        details['sharpe_improves'] = (
            f"baseline={baseline_full['sharpe']:+.4f}  "
            f"best({best_strat})={best_full['sharpe']:+.4f}  "
            f"delta={delta_sh:+.4f}  (need >= +{SHARPE_IMPROVE_MIN})"
        )
    else:
        scores['sharpe_improves'] = False
        details['sharpe_improves'] = 'insufficient data'

    # C2: Max drawdown reduces >= 15%
    if baseline_full and best_full:
        dd_red = (baseline_full['max_dd'] - best_full['max_dd']) / baseline_full['max_dd']
        scores['drawdown_reduces'] = dd_red >= DD_REDUCE_MIN_FRAC
        details['drawdown_reduces'] = (
            f"baseline={baseline_full['max_dd']:.2f}  "
            f"best({best_strat})={best_full['max_dd']:.2f}  "
            f"reduction={dd_red:.1%}  (need >= {DD_REDUCE_MIN_FRAC:.0%})"
        )
    else:
        scores['drawdown_reduces'] = False
        details['drawdown_reduces'] = 'insufficient data'

    # C3: Left tail improves
    if baseline_full and best_full:
        tail_improves = best_full['p_lt_neg1'] < baseline_full['p_lt_neg1']
        scores['left_tail_improves'] = tail_improves
        details['left_tail_improves'] = (
            f"baseline P(<-1)={baseline_full['p_lt_neg1']:.3f}  "
            f"best({best_strat}) P(<-1)={best_full['p_lt_neg1']:.3f}"
        )
    else:
        scores['left_tail_improves'] = False
        details['left_tail_improves'] = 'insufficient data'

    # C4: Effect persists forward
    if baseline_fwd and best_fwd:
        fwd_holds = best_fwd['sharpe'] > baseline_fwd['sharpe']
        scores['fwd_holds'] = fwd_holds
        details['fwd_holds'] = (
            f"fwd baseline={baseline_fwd['sharpe']:+.4f}  "
            f"fwd best({best_strat})={best_fwd['sharpe']:+.4f}"
        )
    else:
        scores['fwd_holds'] = False
        details['fwd_holds'] = 'insufficient forward data'

    # C5: Sigma consistency
    b_delta = sigma_check.get('B', {}).get('delta', float('nan'))
    c_delta = sigma_check.get('C', {}).get('delta', float('nan'))
    best_delta = b_delta if best_strat == 'B' else c_delta
    if math.isnan(best_delta):
        scores['sigma_consistent'] = False
        details['sigma_consistent'] = 'could not compute'
    else:
        scores['sigma_consistent'] = best_delta < SIGMA_CONSISTENCY_SHARPE_DIFF
        details['sigma_consistent'] = (
            f"Sharpe gap (σ_resid vs SE) for {best_strat}: {best_delta:.4f}  "
            f"(need < {SIGMA_CONSISTENCY_SHARPE_DIFF})"
        )

    n_pass = sum(1 for v in scores.values() if v)
    n_total = len(scores)

    if n_pass == n_total:
        verdict = 'PASS_ATR_FILTER_VALIDATED'
    elif n_pass >= 3:
        verdict = 'PASS_WITH_CAVEATS'
    else:
        verdict = 'FAIL_ATR_HYPOTHESIS_REJECTED'

    return dict(
        scores=scores,
        details=details,
        n_pass=n_pass,
        n_total=n_total,
        verdict=verdict,
        best_strategy=best_strat,
    )


# ── Output writers ──────────────────────────────────────────────────────────────

def write_csv(
    output_path: str,
    splits: dict,
    diag: dict,
    sigma_check: dict,
    criteria: dict,
    sensitivity: list[dict],
) -> None:
    rows: list[tuple] = []

    for period, strat_stats in splits.items():
        for strat, s in strat_stats.items():
            if s is None:
                continue
            pref = f'split_{period}_strat_{strat}'
            for k, v in s.items():
                rows.append((pref, k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    for k, v in diag.items():
        rows.append(('instability_diag', k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    for strat, sc in sigma_check.items():
        for k, v in sc.items():
            rows.append((f'sigma_check_strat_{strat}', k, f'{v:.4f}' if isinstance(v, float) else str(v)))

    rows.append(('verdict', 'result', criteria['verdict']))
    rows.append(('verdict', 'n_pass', str(criteria['n_pass'])))
    rows.append(('verdict', 'n_total', str(criteria['n_total'])))
    rows.append(('verdict', 'best_strategy', criteria['best_strategy']))
    for k, v in criteria['scores'].items():
        rows.append(('criterion', k, 'PASS' if v else 'FAIL'))
    for k, v in criteria['details'].items():
        rows.append(('criterion_detail', k, v))

    for row in sensitivity:
        pref = f'sensitivity_k{row["k"]:.2f}_p{row["persist"]}'
        for k2, v in row.items():
            if k2 in ('k', 'persist'):
                continue
            rows.append((pref, k2, f'{v:.4f}' if isinstance(v, float) else str(v)))

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['section', 'key', 'value'])
        writer.writerows(rows)
    print(f'CSV written: {output_path}')


def write_memo(
    memo_path: str,
    splits: dict,
    diag: dict,
    sigma_check: dict,
    criteria: dict,
    sensitivity: list[dict],
) -> None:
    lines = []
    sep = '=' * 72

    lines += [
        'ATR STABILITY FILTER — USD/CHF ETD REGIME PREDICTABILITY TEST',
        'Issue #98',
        'Generated: 2026-03-31',
        '',
        sep,
        f'VERDICT: {criteria["verdict"]}  ({criteria["n_pass"]}/{criteria["n_total"]} criteria pass)',
        f'Best strategy: {criteria["best_strategy"]}',
        sep,
        '',
        'FILTER DEFINITION',
        '-' * 40,
        f'  ATR window      : {ATR_WINDOW} days',
        f'  Forecast horizon: {FORECAST_HORIZON} days',
        f'  k (threshold)   : {INSTABILITY_K}',
        f'  Persistence     : {PERSIST_DAYS} consecutive days',
        f'  Hold window     : {HOLD_WINDOW_DAYS} days post-entry',
        '',
        '  UNSTABLE if: |ATR_pred_15d - ATR_current| > 2 * sigma_resid',
        '  TRIGGERED if: UNSTABLE for >= 5 consecutive trading days',
        '',
        '  Strategy A: baseline (no filter)',
        '  Strategy B: exit  -- unstable during hold window -> return = 0',
        '  Strategy C: entry+exit -- skip if unstable at entry',
        '',
        'ACCEPTANCE CRITERIA',
        '-' * 40,
    ]
    for k, passed in criteria['scores'].items():
        tag = '[PASS]' if passed else '[FAIL]'
        lines.append(f'  {tag}  {k}')
        lines.append(f'          {criteria["details"][k]}')

    lines += ['', 'PERFORMANCE SUMMARY (full history)', '-' * 40]
    hdr = f'  {"strat":<5} {"n":>5} {"mean":>8} {"sharpe":>8} {"win%":>6} {"payoff":>7} {"max_dd":>8} {"P(<-1)":>8}'
    lines.append(hdr)
    for strat in ('A', 'B', 'C'):
        s = splits['full'][strat]
        if not s:
            continue
        lines.append(
            f'  {strat:<5} {s["n"]:>5} {s["mean"]:>+8.4f} {s["sharpe"]:>+8.4f} '
            f'{s["win_rate"]:>5.1%} {s["payoff"]:>7.3f} {s["max_dd"]:>8.2f} {s["p_lt_neg1"]:>8.3f}'
        )

    lines += ['', 'TIME-SPLIT COMPARISON', '-' * 40]
    lines.append(f'  {"period":<7} {"strat":<5} {"n":>5} {"mean":>8} {"sharpe":>8} {"max_dd":>8}')
    for period in ('full', 'train', 'val', 'fwd'):
        for strat in ('A', 'B', 'C'):
            s = splits[period][strat]
            if not s:
                lines.append(f'  {period:<7} {strat:<5}   n/a')
                continue
            lines.append(
                f'  {period:<7} {strat:<5} {s["n"]:>5} {s["mean"]:>+8.4f} '
                f'{s["sharpe"]:>+8.4f} {s["max_dd"]:>8.2f}'
            )
        lines.append('')

    lines += ['INSTABILITY DIAGNOSTICS', '-' * 40]
    lines.append(f'  Total days with regression:    {diag.get("total_days", 0)}')
    lines.append(f'  Days triggered (>= 5 consec):  {diag.get("n_triggered", 0)} ({diag.get("pct_triggered", 0):.1%})')
    lines.append(f'  Days raw unstable:             {diag.get("n_raw_unstable", 0)} ({diag.get("pct_raw_unstable", 0):.1%})')
    lines.append(f'  Number of episodes:            {diag.get("n_episodes", 0)}')
    lines.append(f'  Avg episode length (days):     {diag.get("avg_episode_days", 0):.1f}')
    lines.append(f'  Max episode length (days):     {diag.get("max_episode_days", 0)}')

    lines += ['', 'SENSITIVITY DIAGNOSTIC (k values, informational only)', '-' * 40]
    lines.append(
        f'  {"k":>5}  {"persist":>7}  {"pct_trig":>9}  {"n_entry":>7}  '
        f'{"sh_A":>8}  {"sh_B":>8}  {"sh_C":>8}  {"dd_A":>8}  {"dd_B":>8}  {"dd_C":>8}'
    )
    for row in sensitivity:
        lines.append(
            f'  {row["k"]:>5.2f}  {row["persist"]:>7}  {row["pct_days_triggered"]:>8.1%}  '
            f'{row["n_unstable_at_entry"]:>7}  {row["sharpe_A"]:>+8.4f}  '
            f'{row["sharpe_B"]:>+8.4f}  {row["sharpe_C"]:>+8.4f}  '
            f'{row["max_dd_A"]:>8.2f}  {row["max_dd_B"]:>8.2f}  {row["max_dd_C"]:>8.2f}'
        )

    lines += ['', 'SIGMA CONSISTENCY CHECK (sigma_resid vs SE)', '-' * 40]
    for strat, sc in sigma_check.items():
        lines.append(
            f'  Strategy {strat}: sharpe_resid={sc["sharpe_resid"]:+.4f}  '
            f'sharpe_SE={sc["sharpe_se"]:+.4f}  '
            f'gap={sc["delta"]:.4f}'
        )

    lines += ['', 'RECOMMENDATION', '-' * 40]
    verdict = criteria['verdict']
    if verdict == 'PASS_ATR_FILTER_VALIDATED':
        lines += [
            '  ATR STABILITY FILTER VALIDATED.',
            '  Next steps:',
            '  1. Replace linear regression with GARCH for volatility modelling',
            '  2. Generalise filter to all ETD symbols',
            '  3. Integrate as core regime filter in pre-prod qualifier',
        ]
    elif verdict == 'PASS_WITH_CAVEATS':
        lines += [
            '  FILTER SHOWS BENEFIT WITH CAVEATS.',
            f'  Failed: {[k for k,v in criteria["scores"].items() if not v]}',
            '  Do not generalise until failed criteria are resolved.',
        ]
    else:
        lines += [
            '  ATR STABILITY HYPOTHESIS REJECTED.',
            '  Volatility predictability (linear regression) does not reliably',
            '  improve ETD trade quality.',
            '  Recommended next step: return to price-based conditioning.',
        ]
    lines.append('')

    os.makedirs(os.path.dirname(memo_path) or '.', exist_ok=True)
    with open(memo_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Memo written: {memo_path}')


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Issue #98: ATR Stability Filter')
    parser.add_argument('--output', default='results/atr_stability_filter.csv')
    parser.add_argument('--memo',   default='results/atr_stability_filter_memo.txt')
    args = parser.parse_args()

    print('Loading daily ATR for USD/CHF...')
    daily_atr = load_daily_atr(USD_CHF)
    print(f'  {len(daily_atr)} trading days  ({daily_atr[0]["trading_day"]} to {daily_atr[-1]["trading_day"]})')

    print('Building instability series (sigma_resid)...')
    instability_resid = build_instability_series(daily_atr, use_se=False)

    print('Building instability series (SE)...')
    instability_se = build_instability_series(daily_atr, use_se=True)

    trading_days_sorted = sorted(instability_resid)

    print(f'Loading USD/CHF ETD trades...')
    raw = load_etd_trades(USD_CHF)

    # classify state, compute baseline and excess
    bl_groups: dict[tuple, list[float]] = defaultdict(list)
    for tr in raw:
        d = tr['breakout_direction']
        if d not in ('UP', 'DOWN'):
            continue
        bl_groups[(USD_CHF, d)].append(float(tr['realized_return_sd30d']))
    baselines = {k: sum(v) / len(v) for k, v in bl_groups.items()}

    etd_trades = []
    for tr in raw:
        vr = float(tr['vol_ratio_20_100']); ef = float(tr['efficiency_20'])
        d = tr['breakout_direction']
        if d not in ('UP', 'DOWN'):
            continue
        state = f'{vol_bucket(vr)}_{eff_bucket(ef)}_{d}'
        if state != ETD_STATE:
            continue
        bl = baselines.get((USD_CHF, d), 0.0)
        tr['excess'] = float(tr['realized_return_sd30d']) - bl - MEDIUM_COST
        tr['year'] = tr['event_hour_ts'].year
        etd_trades.append(tr)

    print(f'  {len(etd_trades)} ETD trades')

    print('Annotating trades with instability flags...')
    etd_trades = annotate_trades(etd_trades, instability_resid, trading_days_sorted)

    n_unstable_entry = sum(1 for tr in etd_trades if tr['unstable_at_entry'])
    n_unstable_window = sum(1 for tr in etd_trades if tr['unstable_in_window'])
    print(f'  Unstable at entry:    {n_unstable_entry}')
    print(f'  Unstable in window:   {n_unstable_window}')

    print('Applying strategies...')
    strat_trades = apply_strategies(etd_trades)
    for strat, trades in strat_trades.items():
        s = compute_stats(trades)
        if s:
            print(f'  Strategy {strat}: n={s["n"]:>3}  sharpe={s["sharpe"]:>+.4f}  max_dd={s["max_dd"]:.2f}')

    print('Computing time splits...')
    splits = time_splits(strat_trades)

    print('Computing instability diagnostics...')
    diag = instability_diagnostics(instability_resid)
    print(f'  Triggered days: {diag["n_triggered"]} ({diag["pct_triggered"]:.1%})  '
          f'Episodes: {diag["n_episodes"]}  Avg duration: {diag["avg_episode_days"]:.1f}d')

    print('Running sensitivity diagnostic (informational)...')
    sensitivity = sensitivity_diagnostic(daily_atr, etd_trades, trading_days_sorted)
    print(f'  {"k":>5}  {"persist":>7}  {"pct_trig":>9}  {"n_entry":>7}  {"sh_B":>8}  {"sh_C":>8}')
    for row in sensitivity:
        print(f'  {row["k"]:>5.2f}  {row["persist"]:>7}  {row["pct_days_triggered"]:>8.1%}  '
              f'{row["n_unstable_at_entry"]:>7}  {row["sharpe_B"]:>+8.4f}  {row["sharpe_C"]:>+8.4f}')

    print('Running sigma consistency check...')
    sigma_check = sigma_consistency(etd_trades, instability_resid, instability_se, trading_days_sorted)
    for strat, sc in sigma_check.items():
        print(f'  Strategy {strat}: sharpe_resid={sc["sharpe_resid"]:+.4f}  sharpe_SE={sc["sharpe_se"]:+.4f}  gap={sc["delta"]:.4f}')

    print('Evaluating acceptance criteria...')
    criteria = evaluate_criteria(splits, sigma_check)

    print(f'\nVERDICT: {criteria["verdict"]} ({criteria["n_pass"]}/{criteria["n_total"]})')
    for k, v in criteria['scores'].items():
        tag = 'PASS' if v else 'FAIL'
        print(f'  [{tag}]  {k}  — {criteria["details"][k]}')

    print('\nWriting outputs...')
    write_csv(args.output, splits, diag, sigma_check, criteria, sensitivity)
    write_memo(args.memo, splits, diag, sigma_check, criteria, sensitivity)


if __name__ == '__main__':
    main()

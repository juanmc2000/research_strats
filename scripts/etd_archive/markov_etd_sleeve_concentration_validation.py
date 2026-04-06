# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_sleeve_concentration_validation.py
Issue #104: ETD Sleeve Concentration Robustness and Second-Engine Validation

Determines whether GBP/USD is a genuine second alpha engine in the frozen ETD
sleeve, or whether the sleeve is effectively just USD/CHF with modest
diversification benefit.

Frozen strategy object (Issues #101/#102):
  state   = EXPANDING_TRENDING_DOWN
  filter  = ATR-transition filter only (atr_bucket == TRANSITION_1_2 excluded)
  symbols = USD/CHF, GBP/USD
  weights = equal-weight 50/50 (primary reference)

This script does NOT:
  - introduce new filters
  - alter thresholds or signal definitions
  - apply shock filter to GBP/USD
  - optimize weights
  - include Kelly, pyramiding, or dynamic sizing

Parts:
  1. Rolling concentration stability (quarterly + 4-quarter rolling)
  2. Leave-one-symbol-out (LOSO) validation
  3. Weak-USD/CHF period dependency test
  4. Drawdown episode analysis
  5. Forward-only decision ruling

Pre-declared decision rules (written before any results are seen):
  - GBP/USD ACTIVE   if it improves fwd Sharpe >= 0 AND fwd MaxDD is not
                     worsened by > LOSO_DD_DEGRADE_THRESHOLD; OR it reduces
                     fwd MaxDD by >= LOSO_DD_RELIEF_MIN even if fwd Sharpe
                     declines by no more than LOSO_SHARPE_DEGRADE_THRESHOLD.
  - GBP/USD MONITOR  if it provides evidence of benefit on one metric but
                     forward trade count for GBP/USD alone is < MIN_FWD_TRADES.
  - GBP/USD REMOVE   if neither return improvement nor risk reduction is
                     demonstrated in the forward period.

Usage
-----
  python scripts/markov_etd_sleeve_concentration_validation.py \\
      --memo results/etd_sleeve_concentration_memo.txt
"""

from __future__ import annotations

import argparse
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
WEIGHTS: dict[str, float] = {'USD/CHF': 0.50, 'GBP/USD': 0.50}

# ── Pre-declared decision thresholds (set before results are inspected) ───────

MIN_FWD_TRADES: int = 10

# Concentration flags
CONC_HIGH_THRESHOLD: float = 0.70    # flag window if USD/CHF PnL% > 70%
CONC_VERY_HIGH_THRESHOLD: float = 0.80  # flag window if USD/CHF PnL% > 80%
GBPUSD_LOW_THRESHOLD: float = 0.20   # flag window if GBP/USD PnL% < 20%

# LOSO ruling thresholds
# GBP/USD is ACTIVE if (fwd Sharpe does not worsen by > threshold) AND
# (fwd MaxDD does not worsen by > threshold), considering it aids either metric.
LOSO_SHARPE_DEGRADE_THRESHOLD: float = 0.05   # max allowed fwd Sharpe decline vs USD/CHF alone
LOSO_DD_DEGRADE_THRESHOLD: float = 2.0        # max allowed fwd MaxDD increase vs USD/CHF alone
LOSO_DD_RELIEF_MIN: float = 1.0               # minimum fwd MaxDD reduction to qualify as relief

# Weak USD/CHF period: quarter where USD/CHF-only mean return <= 0
WEAK_USDCHF_MEAN_THRESHOLD: float = 0.0

# Drawdown episode: sleeve drawdown >= this threshold to qualify as an episode
DRAWDOWN_EPISODE_MIN: float = 1.0   # in SD30d units (cumulative sleeve loss from peak)

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
            continue

        excess = float(tr['realized_return_sd30d']) - MEDIUM_COST
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
        month = ts.month if hasattr(ts, 'month') else 1
        quarter = (month - 1) // 3 + 1

        trades.append(dict(
            symbol=symbol,
            event_hour_ts=ts,
            excess=excess,
            year=year,
            quarter=quarter,
            quarter_label=f'Q{year}Q{quarter}',
        ))
    return trades


def build_sleeve_trades(symbol_trades: dict[str, list[dict]],
                        weights: dict[str, float]) -> list[dict]:
    """Merge per-symbol trades into sleeve-level trades, weighted excess return."""
    combined = []
    for symbol, w in weights.items():
        for tr in symbol_trades.get(symbol, []):
            entry = dict(tr)
            entry['excess'] = tr['excess'] * w
            combined.append(entry)
    combined.sort(key=lambda x: x['event_hour_ts'])
    return combined


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> Optional[dict]:
    if not trades:
        return None
    vals = [tr['excess'] for tr in trades]
    n = len(vals)
    mn = sum(vals) / n
    sd = statistics.stdev(vals) if n > 1 else 0.0
    sharpe = mn / sd if sd > 0 else 0.0
    win_rate = sum(1 for v in vals if v > 0) / n
    wins = [v for v in vals if v > 0]
    losses = [abs(v) for v in vals if v < 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
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
    )


def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


# ── Symbol contribution helpers ────────────────────────────────────────────────

def compute_symbol_pnl_contribution(
    symbol_trades: dict[str, list[dict]],
    weights: dict[str, float],
    trade_subset: list[dict],
) -> dict[str, dict]:
    """
    Given a list of sleeve-level trades (trade_subset), compute per-symbol
    PnL%, variance%, and drawdown% contributions.
    Returns dict[symbol] -> {pnl_pct, var_pct, dd_pct, n}.
    """
    result: dict[str, dict] = {}
    timestamps = {tr['event_hour_ts'] for tr in trade_subset}

    sym_pnls: dict[str, float] = {}
    sym_vars: dict[str, float] = {}
    sym_dds: dict[str, float] = {}

    for symbol, w in weights.items():
        sym_sub = [tr for tr in symbol_trades.get(symbol, [])
                   if tr['event_hour_ts'] in timestamps]
        if not sym_sub:
            sym_pnls[symbol] = 0.0
            sym_vars[symbol] = 0.0
            sym_dds[symbol] = 0.0
            result[symbol] = dict(n=0, pnl_pct=0.0, var_pct=0.0, dd_pct=0.0)
            continue
        vals = [tr['excess'] * w for tr in sym_sub]
        pnl = sum(vals)
        mean_v = pnl / len(vals)
        var = sum((v - mean_v) ** 2 for v in vals) if len(vals) > 1 else 0.0
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
        sym_pnls[symbol] = pnl
        sym_vars[symbol] = var
        sym_dds[symbol] = max_dd
        result[symbol] = dict(n=len(sym_sub), pnl_raw=pnl, var_raw=var, dd_raw=max_dd)

    total_pnl = sum(abs(v) for v in sym_pnls.values())
    total_var = sum(sym_vars.values())
    total_dd = sum(sym_dds.values())

    for symbol in weights:
        d = result[symbol]
        d['pnl_pct'] = abs(sym_pnls[symbol]) / total_pnl if total_pnl > 0 else 0.0
        d['var_pct'] = sym_vars[symbol] / total_var if total_var > 0 else 0.0
        d['dd_pct'] = sym_dds[symbol] / total_dd if total_dd > 0 else 0.0

    return result


# ── Part 1: Rolling concentration stability ────────────────────────────────────

def part1_concentration(
    symbol_trades: dict[str, list[dict]],
) -> list[dict]:
    """
    Per calendar quarter and rolling 4-quarter window: sleeve stats + symbol
    contribution %. Returns rows ready for etd_sleeve_concentration_monitor.
    """
    rows = []

    all_sleeve = build_sleeve_trades(symbol_trades, WEIGHTS)

    # Collect all quarter labels in chronological order
    all_quarters: list[str] = sorted(
        {tr['quarter_label'] for sl in symbol_trades.values() for tr in sl}
    )

    def _quarter_key(ql: str) -> tuple:
        # 'Q2022Q1' -> (2022, 1)
        parts = ql.split('Q')
        return (int(parts[1]), int(parts[2]))

    all_quarters.sort(key=_quarter_key)

    def _contributions_for_window(
        window_trades: list[dict],
        window_label: str,
        split_name: str,
    ) -> None:
        if not window_trades:
            return
        sleeve_stats = compute_stats(window_trades)
        contribs = compute_symbol_pnl_contribution(symbol_trades, WEIGHTS, window_trades)
        for symbol in SLEEVE_SYMBOLS:
            c = contribs.get(symbol, {})
            pnl_pct = c.get('pnl_pct', 0.0)
            usdchf_pct = contribs.get('USD/CHF', {}).get('pnl_pct', 0.0)
            sym_sub = [tr for tr in symbol_trades.get(symbol, [])
                       if tr['event_hour_ts'] in {t['event_hour_ts'] for t in window_trades}]
            sym_stats = compute_stats([{'excess': tr['excess'] * WEIGHTS[symbol]}
                                       for tr in sym_sub]) if sym_sub else None
            rows.append(dict(
                sleeve_name=SLEEVE_NAME,
                symbol=symbol,
                split_name=split_name,
                window_label=window_label,
                n_trades=c.get('n', 0),
                pnl_contribution_pct=pnl_pct,
                variance_contribution_pct=c.get('var_pct', 0.0),
                drawdown_contribution_pct=c.get('dd_pct', 0.0),
                mean_return=sym_stats['mean'] if sym_stats else None,
                sharpe=sym_stats['sharpe'] if sym_stats else None,
                max_drawdown=sym_stats['max_dd'] if sym_stats else None,
                conc_gt70_flag=(symbol == 'USD/CHF' and usdchf_pct > CONC_HIGH_THRESHOLD),
                conc_gt80_flag=(symbol == 'USD/CHF' and usdchf_pct > CONC_VERY_HIGH_THRESHOLD),
            ))

    # Calendar-quarter windows
    quarter_trade_map: dict[str, list[dict]] = defaultdict(list)
    for tr in all_sleeve:
        quarter_trade_map[tr['quarter_label']].append(tr)

    for ql in all_quarters:
        q_trades = quarter_trade_map[ql]
        if not q_trades:
            continue
        # Determine split for this quarter
        year = _quarter_key(ql)[0]
        if year < TRAIN_END_YEAR:
            split = 'train'
        elif year < VAL_END_YEAR:
            split = 'val'
        else:
            split = 'fwd'
        _contributions_for_window(q_trades, ql, split)

    # Rolling 4-quarter windows
    for i in range(len(all_quarters) - 3):
        window_quarters = all_quarters[i:i + 4]
        window_trades = []
        for ql in window_quarters:
            window_trades.extend(quarter_trade_map[ql])
        if not window_trades:
            continue
        label = f'r4q_{window_quarters[-1]}'
        # Split: use end quarter's period
        end_year = _quarter_key(window_quarters[-1])[0]
        if end_year < TRAIN_END_YEAR:
            split = 'train'
        elif end_year < VAL_END_YEAR:
            split = 'val'
        else:
            split = 'fwd'
        _contributions_for_window(window_trades, label, split)

    # Standard splits (full / train / val / fwd)
    sleeve_splits = split_trades(all_sleeve)
    for sp_name, sp_trades in sleeve_splits.items():
        _contributions_for_window(sp_trades, sp_name, sp_name)

    return rows


# ── Part 2: Leave-one-symbol-out validation ────────────────────────────────────

def part2_loso(symbol_trades: dict[str, list[dict]]) -> list[dict]:
    """
    Three configurations:
      removed='none'    : both symbols, equal-weight
      removed='GBP/USD' : USD/CHF only, weight=1.0
      removed='USD/CHF' : GBP/USD only, weight=1.0
    """
    rows = []

    configs = [
        ('none',    {'USD/CHF': 0.50, 'GBP/USD': 0.50}),
        ('GBP/USD', {'USD/CHF': 1.00}),
        ('USD/CHF', {'GBP/USD': 1.00}),
    ]

    for removed, weights in configs:
        sleeve = build_sleeve_trades(symbol_trades, weights)
        splits = split_trades(sleeve)
        for sp_name, sp_trades in splits.items():
            s = compute_stats(sp_trades)
            rows.append(dict(
                sleeve_name=SLEEVE_NAME,
                removed_symbol=removed,
                split_name=sp_name,
                n_trades=s['n'] if s else 0,
                mean_return=s['mean'] if s else None,
                sharpe=s['sharpe'] if s else None,
                max_drawdown=s['max_dd'] if s else None,
                win_rate=s['win_rate'] if s else None,
                payoff=s['payoff'] if s else None,
                p_lt_neg1=s['p_lt_neg1'] if s else None,
                p_gt_pos1=s['p_gt_pos1'] if s else None,
            ))

    return rows


# ── Part 3: Weak USD/CHF period dependency ─────────────────────────────────────

def part3_weak_period(symbol_trades: dict[str, list[dict]]) -> list[dict]:
    """
    For each calendar quarter, compute USD/CHF-only mean return.
    Classify as 'weak' if mean <= WEAK_USDCHF_MEAN_THRESHOLD.
    Then compute sleeve behavior during weak vs strong periods.
    """
    rows = []

    usdchf_trades = symbol_trades.get('USD/CHF', [])
    gbpusd_trades = symbol_trades.get('GBP/USD', [])

    # Per-quarter USD/CHF mean return
    usdchf_by_quarter: dict[str, list[dict]] = defaultdict(list)
    for tr in usdchf_trades:
        usdchf_by_quarter[tr['quarter_label']].append(tr)

    weak_quarters: set[str] = set()
    strong_quarters: set[str] = set()
    for ql, qtrades in usdchf_by_quarter.items():
        if not qtrades:
            continue
        q_mean = sum(tr['excess'] for tr in qtrades) / len(qtrades)
        if q_mean <= WEAK_USDCHF_MEAN_THRESHOLD:
            weak_quarters.add(ql)
        else:
            strong_quarters.add(ql)

    def _sleeve_trades_in_quarters(quarters: set[str]) -> list[dict]:
        combined = []
        for symbol, w in WEIGHTS.items():
            for tr in symbol_trades.get(symbol, []):
                if tr['quarter_label'] in quarters:
                    entry = dict(tr)
                    entry['excess'] = tr['excess'] * w
                    combined.append(entry)
        combined.sort(key=lambda x: x['event_hour_ts'])
        return combined

    def _gbpusd_contribution_in_quarters(quarters: set[str]) -> tuple[float, float]:
        usdchf_pnl = sum(
            tr['excess'] * WEIGHTS['USD/CHF']
            for tr in usdchf_trades if tr['quarter_label'] in quarters
        )
        gbpusd_pnl = sum(
            tr['excess'] * WEIGHTS['GBP/USD']
            for tr in gbpusd_trades if tr['quarter_label'] in quarters
        )
        total = abs(usdchf_pnl) + abs(gbpusd_pnl)
        if total == 0:
            return 0.0, 0.0
        return abs(usdchf_pnl) / total, abs(gbpusd_pnl) / total

    for condition_label, quarters in [('weak_usdchf', weak_quarters),
                                       ('strong_usdchf', strong_quarters)]:
        sleeve_sub = _sleeve_trades_in_quarters(quarters)
        s = compute_stats(sleeve_sub)
        uc_pct, gb_pct = _gbpusd_contribution_in_quarters(quarters)

        # Determine split coverage (use the majority split of included quarters)
        for sp_name, year_filter in [
            ('full', lambda y: True),
            ('train', lambda y: y < TRAIN_END_YEAR),
            ('val', lambda y: TRAIN_END_YEAR <= y < VAL_END_YEAR),
            ('fwd', lambda y: y >= VAL_END_YEAR),
        ]:
            def _quarter_key_fn(ql: str) -> int:
                parts = ql.split('Q')
                return int(parts[1])

            sp_quarters = {ql for ql in quarters if year_filter(_quarter_key_fn(ql))}
            sp_sleeve = _sleeve_trades_in_quarters(sp_quarters)
            sp_s = compute_stats(sp_sleeve)
            sp_uc_pct, sp_gb_pct = _gbpusd_contribution_in_quarters(sp_quarters)
            rows.append(dict(
                sleeve_name=SLEEVE_NAME,
                analysis_name='weak_usdchf_quarter',
                split_name=sp_name,
                condition_label=condition_label,
                n_trades=sp_s['n'] if sp_s else 0,
                mean_return=sp_s['mean'] if sp_s else None,
                sharpe=sp_s['sharpe'] if sp_s else None,
                max_drawdown=sp_s['max_dd'] if sp_s else None,
                win_rate=sp_s['win_rate'] if sp_s else None,
                pnl_contribution_usdchf=sp_uc_pct,
                pnl_contribution_gbpusd=sp_gb_pct,
            ))

    return rows


# ── Part 4: Drawdown episode analysis ─────────────────────────────────────────

def part4_drawdown_episodes(symbol_trades: dict[str, list[dict]]) -> list[dict]:
    """
    Walk the equal-weight sleeve equity curve and identify drawdown episodes
    where the sleeve falls >= DRAWDOWN_EPISODE_MIN from a local peak.
    For each episode, compute per-symbol PnL contribution.
    """
    rows = []

    all_sleeve = build_sleeve_trades(symbol_trades, WEIGHTS)
    if not all_sleeve:
        return rows

    # Build equity curve
    cum = 0.0
    peak = 0.0
    in_drawdown = False
    episode_start_idx: Optional[int] = None
    episodes: list[tuple[int, int]] = []

    for i, tr in enumerate(all_sleeve):
        cum += tr['excess']
        if cum > peak:
            if in_drawdown:
                episodes.append((episode_start_idx, i - 1))
                in_drawdown = False
            peak = cum
        dd = peak - cum
        if dd >= DRAWDOWN_EPISODE_MIN and not in_drawdown:
            in_drawdown = True
            episode_start_idx = i

    if in_drawdown:
        episodes.append((episode_start_idx, len(all_sleeve) - 1))

    usdchf_ts = {tr['event_hour_ts'] for tr in symbol_trades.get('USD/CHF', [])}
    gbpusd_ts = {tr['event_hour_ts'] for tr in symbol_trades.get('GBP/USD', [])}

    for ep_idx, (start, end) in enumerate(episodes):
        ep_trades = all_sleeve[start:end + 1]
        if not ep_trades:
            continue

        ep_ts = {tr['event_hour_ts'] for tr in ep_trades}

        uc_pnl = sum(
            tr['excess'] * WEIGHTS['USD/CHF']
            for tr in symbol_trades.get('USD/CHF', [])
            if tr['event_hour_ts'] in ep_ts
        )
        gb_pnl = sum(
            tr['excess'] * WEIGHTS['GBP/USD']
            for tr in symbol_trades.get('GBP/USD', [])
            if tr['event_hour_ts'] in ep_ts
        )
        total = abs(uc_pnl) + abs(gb_pnl)
        uc_pct = abs(uc_pnl) / total if total > 0 else 0.0
        gb_pct = abs(gb_pnl) / total if total > 0 else 0.0

        s = compute_stats(ep_trades)
        start_year = ep_trades[0]['year']
        if start_year < TRAIN_END_YEAR:
            split = 'train'
        elif start_year < VAL_END_YEAR:
            split = 'val'
        else:
            split = 'fwd'

        label = f'ep{ep_idx + 1:02d}'
        rows.append(dict(
            sleeve_name=SLEEVE_NAME,
            analysis_name='drawdown_episode',
            split_name=split,
            condition_label=label,
            n_trades=s['n'] if s else 0,
            mean_return=s['mean'] if s else None,
            sharpe=s['sharpe'] if s else None,
            max_drawdown=s['max_dd'] if s else None,
            win_rate=s['win_rate'] if s else None,
            pnl_contribution_usdchf=uc_pct,
            pnl_contribution_gbpusd=gb_pct,
        ))

    return rows


# ── Part 5: Forward-only decision ─────────────────────────────────────────────

def part5_decision(loso_rows: list[dict]) -> str:
    """
    Apply pre-declared rules to produce KEEP / MONITOR / REMOVE ruling.

    Pre-declared rules:
      1. Compute fwd Sharpe and fwd MaxDD for:
           a) full sleeve (removed='none')
           b) USD/CHF-only (removed='GBP/USD')
           c) GBP/USD-only (removed='USD/CHF')
      2. GBP/USD is ACTIVE if (fwd Sharpe of full sleeve >= USD/CHF-only fwd Sharpe
                                - LOSO_SHARPE_DEGRADE_THRESHOLD)
                              AND full sleeve fwd MaxDD <= USD/CHF-only fwd MaxDD
                                + LOSO_DD_DEGRADE_THRESHOLD
                              AND (Sharpe improves OR MaxDD reduces by >= LOSO_DD_RELIEF_MIN)
      3. GBP/USD is MONITOR if it passes one of the two metrics but forward
         GBP/USD-only trade count < MIN_FWD_TRADES.
      4. GBP/USD is REMOVE if it fails both metrics.
    """
    fwd = {row['removed_symbol']: row
           for row in loso_rows if row['split_name'] == 'fwd'}

    full_fwd = fwd.get('none', {})
    usdchf_fwd = fwd.get('GBP/USD', {})
    gbpusd_fwd = fwd.get('USD/CHF', {})

    full_sharpe = full_fwd.get('sharpe') or 0.0
    uc_sharpe = usdchf_fwd.get('sharpe') or 0.0
    full_dd = full_fwd.get('max_drawdown') or 0.0
    uc_dd = usdchf_fwd.get('max_drawdown') or 0.0
    gb_n = gbpusd_fwd.get('n_trades') or 0

    sharpe_ok = full_sharpe >= uc_sharpe - LOSO_SHARPE_DEGRADE_THRESHOLD
    dd_ok = full_dd <= uc_dd + LOSO_DD_DEGRADE_THRESHOLD
    sharpe_improves = full_sharpe > uc_sharpe
    dd_relieves = (uc_dd - full_dd) >= LOSO_DD_RELIEF_MIN

    if sharpe_ok and dd_ok and (sharpe_improves or dd_relieves):
        ruling = 'GBP/USD ACTIVE'
        reason = (
            f'Full sleeve fwd Sharpe {full_sharpe:.3f} vs USD/CHF-only {uc_sharpe:.3f} '
            f'(delta {full_sharpe - uc_sharpe:+.3f}, threshold -{LOSO_SHARPE_DEGRADE_THRESHOLD}). '
            f'Full sleeve fwd MaxDD {full_dd:.3f} vs USD/CHF-only {uc_dd:.3f} '
            f'(delta {full_dd - uc_dd:+.3f}, threshold +{LOSO_DD_DEGRADE_THRESHOLD}). '
            f'{"Sharpe improves." if sharpe_improves else ""} '
            f'{"MaxDD relief >= threshold." if dd_relieves else ""}'
        )
    elif gb_n < MIN_FWD_TRADES:
        ruling = 'GBP/USD MONITOR'
        reason = (
            f'GBP/USD-only forward trade count = {gb_n} < {MIN_FWD_TRADES}. '
            f'Insufficient forward evidence. '
            f'Full sleeve fwd Sharpe {full_sharpe:.3f}, USD/CHF-only {uc_sharpe:.3f}.'
        )
    elif sharpe_ok or dd_ok:
        ruling = 'GBP/USD MONITOR'
        reason = (
            f'One metric passes but not both. '
            f'Sharpe OK: {sharpe_ok}, DD OK: {dd_ok}. '
            f'Full fwd Sharpe {full_sharpe:.3f} vs USD/CHF-only {uc_sharpe:.3f}. '
            f'Full fwd MaxDD {full_dd:.3f} vs USD/CHF-only {uc_dd:.3f}.'
        )
    else:
        ruling = 'GBP/USD REMOVE'
        reason = (
            f'Both metrics fail. '
            f'Sharpe: full {full_sharpe:.3f} vs USD/CHF-only {uc_sharpe:.3f} '
            f'(delta {full_sharpe - uc_sharpe:+.3f}, need >= -{LOSO_SHARPE_DEGRADE_THRESHOLD}). '
            f'MaxDD: full {full_dd:.3f} vs USD/CHF-only {uc_dd:.3f} '
            f'(delta {full_dd - uc_dd:+.3f}, limit +{LOSO_DD_DEGRADE_THRESHOLD}).'
        )

    return f'{ruling}\n  Reason: {reason}'


# ── DB upserts ─────────────────────────────────────────────────────────────────

def upsert_concentration_monitor(conn, rows: list[dict]) -> None:
    sql = """
        INSERT INTO features.etd_sleeve_concentration_monitor (
            sleeve_name, symbol, split_name, window_label,
            n_trades, pnl_contribution_pct, variance_contribution_pct,
            drawdown_contribution_pct, mean_return, sharpe, max_drawdown,
            conc_gt70_flag, conc_gt80_flag, updated_at
        ) VALUES (
            %(sleeve_name)s, %(symbol)s, %(split_name)s, %(window_label)s,
            %(n_trades)s, %(pnl_contribution_pct)s, %(variance_contribution_pct)s,
            %(drawdown_contribution_pct)s, %(mean_return)s, %(sharpe)s, %(max_drawdown)s,
            %(conc_gt70_flag)s, %(conc_gt80_flag)s, now()
        )
        ON CONFLICT (sleeve_name, symbol, split_name, window_label)
        DO UPDATE SET
            n_trades                  = EXCLUDED.n_trades,
            pnl_contribution_pct      = EXCLUDED.pnl_contribution_pct,
            variance_contribution_pct = EXCLUDED.variance_contribution_pct,
            drawdown_contribution_pct = EXCLUDED.drawdown_contribution_pct,
            mean_return               = EXCLUDED.mean_return,
            sharpe                    = EXCLUDED.sharpe,
            max_drawdown              = EXCLUDED.max_drawdown,
            conc_gt70_flag            = EXCLUDED.conc_gt70_flag,
            conc_gt80_flag            = EXCLUDED.conc_gt80_flag,
            updated_at                = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    conn.commit()
    print(f'  {len(rows)} rows written to etd_sleeve_concentration_monitor')


def upsert_loso(conn, rows: list[dict]) -> None:
    sql = """
        INSERT INTO features.etd_sleeve_loso_validation (
            sleeve_name, removed_symbol, split_name,
            n_trades, mean_return, sharpe, max_drawdown,
            win_rate, payoff, p_lt_neg1, p_gt_pos1, updated_at
        ) VALUES (
            %(sleeve_name)s, %(removed_symbol)s, %(split_name)s,
            %(n_trades)s, %(mean_return)s, %(sharpe)s, %(max_drawdown)s,
            %(win_rate)s, %(payoff)s, %(p_lt_neg1)s, %(p_gt_pos1)s, now()
        )
        ON CONFLICT (sleeve_name, removed_symbol, split_name)
        DO UPDATE SET
            n_trades    = EXCLUDED.n_trades,
            mean_return = EXCLUDED.mean_return,
            sharpe      = EXCLUDED.sharpe,
            max_drawdown = EXCLUDED.max_drawdown,
            win_rate    = EXCLUDED.win_rate,
            payoff      = EXCLUDED.payoff,
            p_lt_neg1   = EXCLUDED.p_lt_neg1,
            p_gt_pos1   = EXCLUDED.p_gt_pos1,
            updated_at  = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    conn.commit()
    print(f'  {len(rows)} rows written to etd_sleeve_loso_validation')


def upsert_dependency(conn, rows: list[dict]) -> None:
    sql = """
        INSERT INTO features.etd_sleeve_dependency_analysis (
            sleeve_name, analysis_name, split_name, condition_label,
            n_trades, mean_return, sharpe, max_drawdown, win_rate,
            pnl_contribution_usdchf, pnl_contribution_gbpusd, updated_at
        ) VALUES (
            %(sleeve_name)s, %(analysis_name)s, %(split_name)s, %(condition_label)s,
            %(n_trades)s, %(mean_return)s, %(sharpe)s, %(max_drawdown)s, %(win_rate)s,
            %(pnl_contribution_usdchf)s, %(pnl_contribution_gbpusd)s, now()
        )
        ON CONFLICT (sleeve_name, analysis_name, split_name, condition_label)
        DO UPDATE SET
            n_trades                  = EXCLUDED.n_trades,
            mean_return               = EXCLUDED.mean_return,
            sharpe                    = EXCLUDED.sharpe,
            max_drawdown              = EXCLUDED.max_drawdown,
            win_rate                  = EXCLUDED.win_rate,
            pnl_contribution_usdchf   = EXCLUDED.pnl_contribution_usdchf,
            pnl_contribution_gbpusd   = EXCLUDED.pnl_contribution_gbpusd,
            updated_at                = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    conn.commit()
    print(f'  {len(rows)} rows written to etd_sleeve_dependency_analysis')


# ── Memo ───────────────────────────────────────────────────────────────────────

def write_memo(
    memo_path: str,
    loso_rows: list[dict],
    conc_rows: list[dict],
    dep_rows: list[dict],
    ruling: str,
) -> None:
    lines = []
    W = 76

    def section(title: str) -> None:
        lines.append('=' * W)
        lines.append(title)
        lines.append('=' * W)

    def sub(title: str) -> None:
        lines.append(title)
        lines.append('-' * len(title))

    section('ETD SLEEVE CONCENTRATION ROBUSTNESS  (Issue #104)')
    lines.append('Frozen sleeve: USD/CHF + GBP/USD, ATR-transition filter only.')
    lines.append('Reference weight: equal-weight 50/50.')
    lines.append('')

    section('PRE-DECLARED DECISION RULES')
    lines.append(f'  GBP/USD ACTIVE   : fwd Sharpe not worse than USD/CHF-only by > {LOSO_SHARPE_DEGRADE_THRESHOLD}')
    lines.append(f'                     AND fwd MaxDD not worse than USD/CHF-only by > {LOSO_DD_DEGRADE_THRESHOLD}')
    lines.append(f'                     AND (Sharpe improves OR MaxDD reduces by >= {LOSO_DD_RELIEF_MIN})')
    lines.append(f'  GBP/USD MONITOR  : one metric passes, or forward n < {MIN_FWD_TRADES}')
    lines.append(f'  GBP/USD REMOVE   : both metrics fail')
    lines.append(f'  Concentration flags: USD/CHF PnL% > {CONC_HIGH_THRESHOLD:.0%} or > {CONC_VERY_HIGH_THRESHOLD:.0%}')
    lines.append(f'  Weak USD/CHF: quarter mean return <= {WEAK_USDCHF_MEAN_THRESHOLD}')
    lines.append('')

    # ── Part 2: LOSO ──────────────────────────────────────────────────────────
    section('PART 2 — LEAVE-ONE-SYMBOL-OUT VALIDATION')
    hdr = f'  {"Config":<20} {"split":<6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"P<-1":>7} {"P>+1":>7}'
    lines.append(hdr)
    lines.append('  ' + '-' * (len(hdr) - 2))

    label_map = {
        'none':    'both (50/50)',
        'GBP/USD': 'USD/CHF only',
        'USD/CHF': 'GBP/USD only',
    }
    for removed in ['none', 'GBP/USD', 'USD/CHF']:
        for sp in SPLITS:
            r = next((x for x in loso_rows
                      if x['removed_symbol'] == removed and x['split_name'] == sp), None)
            if not r:
                continue
            n = r['n_trades'] or 0
            sh = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else '  n/a'
            dd = f"{r['max_drawdown']:.3f}" if r['max_drawdown'] is not None else '  n/a'
            wr = f"{r['win_rate']:.3f}" if r['win_rate'] is not None else '  n/a'
            pl = f"{r['p_lt_neg1']:.3f}" if r['p_lt_neg1'] is not None else '  n/a'
            pg = f"{r['p_gt_pos1']:.3f}" if r['p_gt_pos1'] is not None else '  n/a'
            lines.append(f'  {label_map[removed]:<20} {sp:<6} {n:>5} {sh:>8} {dd:>8} {wr:>7} {pl:>7} {pg:>7}')
        lines.append('')

    # ── Part 1: Concentration (standard splits only) ──────────────────────────
    section('PART 1 — CONCENTRATION MONITOR (STANDARD SPLITS)')
    hdr2 = f'  {"Scheme":<10} {"split":<6} {"Symbol":<10} {"PnL%":>7} {"Var%":>7} {"DD%":>7} {"n":>5} {"Conc>70":>8} {"Conc>80":>8}'
    lines.append(hdr2)
    lines.append('  ' + '-' * (len(hdr2) - 2))

    std_splits_rows = [r for r in conc_rows if r['window_label'] in SPLITS]
    for sp in SPLITS:
        for sym in SLEEVE_SYMBOLS:
            r = next((x for x in std_splits_rows
                      if x['split_name'] == sp and x['window_label'] == sp
                      and x['symbol'] == sym), None)
            if not r:
                continue
            lines.append(
                f'  {"equal_weight":<10} {sp:<6} {sym:<10} '
                f'{r["pnl_contribution_pct"]:>7.1%} '
                f'{r["variance_contribution_pct"]:>7.1%} '
                f'{r["drawdown_contribution_pct"]:>7.1%} '
                f'{r["n_trades"] or 0:>5} '
                f'{"YES" if r["conc_gt70_flag"] else "no":>8} '
                f'{"YES" if r["conc_gt80_flag"] else "no":>8}'
            )
        lines.append('')

    # Flagged concentration windows
    flagged = [r for r in conc_rows
               if r['symbol'] == 'USD/CHF' and r['conc_gt70_flag']
               and r['window_label'] not in SPLITS]
    if flagged:
        sub('WINDOWS WITH USD/CHF CONCENTRATION > 70%')
        for r in sorted(flagged, key=lambda x: x['window_label']):
            lines.append(
                f'  {r["window_label"]:<20} {r["split_name"]:<6} '
                f'USD/CHF PnL%={r["pnl_contribution_pct"]:.1%}  '
                f'n={r["n_trades"] or 0}'
            )
        lines.append('')

    # ── Part 3: Weak USD/CHF periods ─────────────────────────────────────────
    section('PART 3 — WEAK USD/CHF PERIOD DEPENDENCY')
    lines.append('  Weak = USD/CHF quarter mean return <= 0')
    lines.append('')
    hdr3 = f'  {"Condition":<20} {"split":<6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"USD/CHF%":>9} {"GBP/USD%":>9}'
    lines.append(hdr3)
    lines.append('  ' + '-' * (len(hdr3) - 2))

    for cond in ['weak_usdchf', 'strong_usdchf']:
        for sp in SPLITS:
            r = next((x for x in dep_rows
                      if x['analysis_name'] == 'weak_usdchf_quarter'
                      and x['condition_label'] == cond
                      and x['split_name'] == sp), None)
            if not r or not r['n_trades']:
                continue
            sh = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else '  n/a'
            dd = f"{r['max_drawdown']:.3f}" if r['max_drawdown'] is not None else '  n/a'
            lines.append(
                f'  {cond:<20} {sp:<6} {r["n_trades"] or 0:>5} {sh:>8} {dd:>8} '
                f'{r["pnl_contribution_usdchf"] or 0:>9.1%} '
                f'{r["pnl_contribution_gbpusd"] or 0:>9.1%}'
            )
        lines.append('')

    # ── Part 4: Drawdown episodes ─────────────────────────────────────────────
    section('PART 4 — DRAWDOWN EPISODE ANALYSIS')
    lines.append(f'  Episodes where sleeve cumulative loss >= {DRAWDOWN_EPISODE_MIN} SD30d from peak.')
    lines.append('')
    hdr4 = f'  {"Episode":<10} {"split":<6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"USD/CHF%":>9} {"GBP/USD%":>9}'
    lines.append(hdr4)
    lines.append('  ' + '-' * (len(hdr4) - 2))

    ep_rows = [r for r in dep_rows if r['analysis_name'] == 'drawdown_episode']
    if ep_rows:
        for r in ep_rows:
            sh = f"{r['sharpe']:.3f}" if r['sharpe'] is not None else '  n/a'
            dd = f"{r['max_drawdown']:.3f}" if r['max_drawdown'] is not None else '  n/a'
            wr = f"{r['win_rate']:.3f}" if r['win_rate'] is not None else '  n/a'
            lines.append(
                f'  {r["condition_label"]:<10} {r["split_name"]:<6} '
                f'{r["n_trades"] or 0:>5} {sh:>8} {dd:>8} {wr:>7} '
                f'{r["pnl_contribution_usdchf"] or 0:>9.1%} '
                f'{r["pnl_contribution_gbpusd"] or 0:>9.1%}'
            )
    else:
        lines.append('  No drawdown episodes >= threshold detected.')
    lines.append('')

    # ── Part 5: Decision ──────────────────────────────────────────────────────
    section('PART 5 — FORWARD-ONLY DECISION')
    for line in ruling.split('\n'):
        lines.append(f'  {line}')
    lines.append('')

    memo_text = '\n'.join(lines) + '\n'
    with open(memo_path, 'w') as f:
        f.write(memo_text)
    print(f'\nMemo written to {memo_path}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ETD sleeve concentration robustness (Issue #104)'
    )
    parser.add_argument('--memo', default='results/etd_sleeve_concentration_memo.txt')
    args = parser.parse_args()

    print(f'Loading ATR-filtered ETD trades for frozen sleeve: {SLEEVE_SYMBOLS}')
    symbol_trades: dict[str, list[dict]] = {}
    for sym in SLEEVE_SYMBOLS:
        trades = build_filtered_trades(sym)
        symbol_trades[sym] = trades
        print(f'  {sym:<10} n={len(trades)}')

    print('\nPart 1: Rolling concentration stability...')
    conc_rows = part1_concentration(symbol_trades)

    print('Part 2: Leave-one-symbol-out validation...')
    loso_rows = part2_loso(symbol_trades)

    print('Part 3: Weak USD/CHF period dependency...')
    dep_rows = part3_weak_period(symbol_trades)

    print('Part 4: Drawdown episode analysis...')
    dep_rows += part4_drawdown_episodes(symbol_trades)

    print('Part 5: Forward-only decision ruling...')
    ruling = part5_decision(loso_rows)

    print('\nUpserting results...')
    conn = connect()
    upsert_concentration_monitor(conn, conc_rows)
    upsert_loso(conn, loso_rows)
    upsert_dependency(conn, dep_rows)
    conn.close()

    print('\n' + '=' * 76)
    print('ETD SLEEVE CONCENTRATION ROBUSTNESS  (Issue #104)')
    print('=' * 76)

    # ── LOSO summary ──────────────────────────────────────────────────────────
    print('\nLEAVE-ONE-SYMBOL-OUT — FORWARD PERIOD')
    label_map = {'none': 'both (50/50)', 'GBP/USD': 'USD/CHF only', 'USD/CHF': 'GBP/USD only'}
    print(f'  {"Config":<20} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"WinR":>7} {"P<-1":>7}')
    print('  ' + '-' * 58)
    for removed in ['none', 'GBP/USD', 'USD/CHF']:
        r = next((x for x in loso_rows
                  if x['removed_symbol'] == removed and x['split_name'] == 'fwd'), None)
        if not r:
            continue
        print(
            f'  {label_map[removed]:<20} {r["n_trades"] or 0:>5} '
            f'{r["sharpe"] or 0:>8.3f} {r["max_drawdown"] or 0:>8.3f} '
            f'{r["win_rate"] or 0:>7.3f} {r["p_lt_neg1"] or 0:>7.3f}'
        )

    print(f'\nCONCENTRATION — FORWARD SPLIT')
    for sym in SLEEVE_SYMBOLS:
        r = next((x for x in conc_rows
                  if x['symbol'] == sym and x['window_label'] == 'fwd'), None)
        if r:
            flag = ' [FLAG >70%]' if r['conc_gt70_flag'] else ''
            print(f'  {sym:<10} PnL%={r["pnl_contribution_pct"]:.1%}  '
                  f'Var%={r["variance_contribution_pct"]:.1%}{flag}')

    print(f'\nWEAK USD/CHF PERIODS — FORWARD')
    for cond in ['weak_usdchf', 'strong_usdchf']:
        r = next((x for x in dep_rows
                  if x['analysis_name'] == 'weak_usdchf_quarter'
                  and x['condition_label'] == cond
                  and x['split_name'] == 'fwd'), None)
        if r and r['n_trades']:
            print(
                f'  {cond:<20} n={r["n_trades"] or 0:>3}  '
                f'Sharpe={r["sharpe"] or 0:>7.3f}  '
                f'USD/CHF%={r["pnl_contribution_usdchf"] or 0:.1%}  '
                f'GBP/USD%={r["pnl_contribution_gbpusd"] or 0:.1%}'
            )

    print(f'\nRULING')
    print(f'  {ruling}')

    write_memo(args.memo, loso_rows, conc_rows, dep_rows, ruling)


if __name__ == '__main__':
    main()

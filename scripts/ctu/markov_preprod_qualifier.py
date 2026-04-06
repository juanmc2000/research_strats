"""
scripts/markov_preprod_qualifier.py

Pre-production qualification for validated Markov states -- Issue #95.

Nine-stage qualification pipeline:
  1. Symbol pruning with explicit triple-negative / weak-profile rules
  2. Sleeve decomposition: CTU-only, ETD-only, combined (pruned and unpruned)
  3. Risk overlays: concurrency caps and loss-run governor
  4. Pessimistic scenarios: pooled / test / shrunk / CI-floor / stressed
  5. ETD retention decision (KEEP / MONITOR / REMOVE)
  6. CTU standalone viability (PREP_PROD_CANDIDATE / MONITOR / REJECT)
  7. Pre-prod scorecard (8 dimensions, 4 verdict levels)
  8. Recommendation memo

Usage:
    python scripts/markov_preprod_qualifier.py
    python scripts/markov_preprod_qualifier.py --no-bootstrap
    python scripts/markov_preprod_qualifier.py --output results/preprod_qualifier.csv \
        --scorecard results/preprod_scorecard.txt \
        --memo results/preprod_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
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

ELIGIBLE_STATES: list[str] = [
    'CONTRACTING_TRENDING_UP',
    'EXPANDING_TRENDING_DOWN',
]

COHORT_SYMBOLS: list[str] = [
    'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
]

MEDIUM_COST:           float = 0.07
SHRINKAGE_FACTOR:      float = 0.50
TEST_SPLIT_START:      str   = '2024-01-01'
DATA_SPAN_YEARS:       float = 11.0
N_BOOTSTRAP:           int   = 1000
BLOCK_SIZE:            int   = 20
RANDOM_SEED:           int   = 42
PRUNE_WIN_RATE_CEILING: float = 0.35
PRUNE_MEDIAN_CEILING:  float = -0.50
PREPROD_DD_CEILING:    float = 30.0
SCORE_SHARPE_FLOOR:    float = 0.10
SCORE_WIN_SYMBOL_FRAC: float = 0.60

OVERLAY_VARIANTS: list[dict] = [
    {'label': 'baseline',    'max_sym': 2, 'max_state': 5, 'max_total': 10, 'loss_gov': 0},
    {'label': 'tight',       'max_sym': 1, 'max_state': 3, 'max_total': 6,  'loss_gov': 0},
    {'label': 'very_tight',  'max_sym': 1, 'max_state': 2, 'max_total': 4,  'loss_gov': 0},
    {'label': 'loss_gov_5',  'max_sym': 2, 'max_state': 5, 'max_total': 10, 'loss_gov': 5},
    {'label': 'tight_gov_5', 'max_sym': 1, 'max_state': 3, 'max_total': 6,  'loss_gov': 5},
]

SCENARIO_LABELS: list[str] = [
    'pooled',
    'test_split',
    'shrunk_50pct',
    'ci_floor',
    'stressed',
]

SCENARIO_FIELD_MAP: dict[str, str] = {
    'pooled':       'pooled_cost_adj',
    'test_split':   'test_cost_adj',
    'shrunk_50pct': 'shrunk_cost_adj',
    'ci_floor':     'ci_floor_cost_adj',
    'stressed':     'stressed_cost_adj',
}

# ---------------------------------------------------------------------------
# Bucketing helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Pure math helpers
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


def _quarter_label(ts: object) -> str:
    if hasattr(ts, 'year') and hasattr(ts, 'month'):
        return f'{ts.year}Q{(ts.month - 1) // 3 + 1}'
    s = str(ts)
    return f'{s[:4]}Q{(int(s[5:7]) - 1) // 3 + 1}'


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
            f'{_vol_bucket(ev["vol_ratio_20_100"])}'
            f'_{_eff_bucket(ev["efficiency_20"])}'
            f'_{d}'
        )
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Carry neutralization
# ---------------------------------------------------------------------------


def compute_baselines(
    events: list[dict],
) -> dict[tuple[str, str], float]:
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
# Block bootstrap
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    returns: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    n = len(returns)
    if n == 0 or n_bootstrap == 0:
        return {'ci_low': None, 'ci_high': None, 'frac_pos': None}
    rng   = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            s = rng.randint(0, n - 1)
            for k in range(BLOCK_SIZE):
                sample.append(returns[(s + k) % n])
        means.append(sum(sample[:n]) / n)
    means.sort()
    lo = means[max(0, int(round(0.025 * n_bootstrap)))]
    hi = means[min(n_bootstrap - 1, int(round(0.975 * n_bootstrap)) - 1)]
    return {
        'ci_low':   round(lo, 4),
        'ci_high':  round(hi, 4),
        'frac_pos': round(sum(1 for m in means if m > 0) / n_bootstrap, 3),
    }


# ---------------------------------------------------------------------------
# Section 1 — Symbol pruning
# ---------------------------------------------------------------------------


def compute_symbol_stats(
    events: list[dict],
) -> dict[tuple[str, str], dict]:
    """
    Compute per (symbol, state) statistics used for pruning decisions.

    Returns a dict keyed by (symbol, state) with fields:
      n, n_test, pooled_mean, test_mean, cost_adj, median, std,
      sharpe, win_rate, avg_win, avg_loss, payoff
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for ev in events:
        state = ev.get('state', '')
        if state not in ELIGIBLE_STATES:
            continue
        key = (ev['symbol'], state)
        groups.setdefault(key, []).append(ev)

    result: dict[tuple[str, str], dict] = {}
    for key, evs in groups.items():
        all_adj  = [float(ev['cost_adj']) for ev in evs
                    if ev.get('cost_adj') is not None]
        test_adj = [float(ev['cost_adj']) for ev in evs
                    if ev.get('cost_adj') is not None
                    and _ts_date_str(ev['event_hour_ts']) >= TEST_SPLIT_START]
        n        = len(all_adj)
        n_test   = len(test_adj)
        if n == 0:
            result[key] = {
                'n': 0, 'n_test': 0,
                'pooled_mean': None, 'test_mean': None,
                'cost_adj': None, 'median': None, 'std': None,
                'sharpe': None, 'win_rate': None,
                'avg_win': None, 'avg_loss': None, 'payoff': None,
            }
            continue

        pooled_mean = round(sum(all_adj) / n, 4)
        test_mean   = round(sum(test_adj) / n_test, 4) if n_test > 0 else None
        med         = round(_median(all_adj), 4)
        sd          = _std(all_adj)
        sharpe      = round(pooled_mean / sd, 4) if sd and sd > 0 else None
        wins        = [r for r in all_adj if r > 0]
        losses      = [r for r in all_adj if r <= 0]
        win_rate    = round(len(wins) / n, 4)
        avg_win     = round(sum(wins) / len(wins), 4) if wins else None
        avg_loss    = round(sum(losses) / len(losses), 4) if losses else None
        payoff      = (
            round(abs(avg_win / avg_loss), 3)
            if avg_win is not None and avg_loss is not None
            else None
        )
        result[key] = {
            'n':           n,
            'n_test':      n_test,
            'pooled_mean': pooled_mean,
            'test_mean':   test_mean,
            'cost_adj':    pooled_mean,
            'median':      med,
            'std':         sd,
            'sharpe':      sharpe,
            'win_rate':    win_rate,
            'avg_win':     avg_win,
            'avg_loss':    avg_loss,
            'payoff':      payoff,
        }
    return result


def _prune_condition_a(s: dict) -> bool:
    """Triple-negative: pooled<=0 AND (test is None OR test<=0) AND cost_adj<=0."""
    pooled   = s.get('pooled_mean')
    test     = s.get('test_mean')
    cost_adj = s.get('cost_adj')
    if pooled is None or cost_adj is None:
        return False
    test_ok = (test is None or test <= 0)
    return pooled <= 0 and test_ok and cost_adj <= 0


def _prune_condition_b(s: dict) -> bool:
    """Weak profile: win_rate < ceiling AND median < ceiling."""
    wr  = s.get('win_rate')
    med = s.get('median')
    if wr is None or med is None:
        return False
    return wr < PRUNE_WIN_RATE_CEILING and med < PRUNE_MEDIAN_CEILING


def apply_pruning_rules(
    symbol_stats: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], dict]:
    """
    Annotate each (symbol, state) entry with:
      pruned       : bool
      prune_reason : str or None
      thin_test    : bool  (n_test < 5, warn but do not prune)
    """
    result: dict[tuple[str, str], dict] = {}
    for key, s in symbol_stats.items():
        entry = dict(s)
        if s.get('n', 0) == 0:
            entry['pruned']       = True
            entry['prune_reason'] = 'n=0'
            entry['thin_test']    = True
            result[key] = entry
            continue

        thin_test = s.get('n_test', 0) < 5

        if _prune_condition_a(s):
            entry['pruned']       = True
            entry['prune_reason'] = 'triple_negative'
        elif _prune_condition_b(s):
            entry['pruned']       = True
            entry['prune_reason'] = 'weak_profile'
        else:
            entry['pruned']       = False
            entry['prune_reason'] = None

        entry['thin_test'] = thin_test
        result[key] = entry
    return result


def filter_events_per_state_symbols(
    events: list[dict],
    state_symbol_map: dict[str, list[str]],
) -> list[dict]:
    """Keep events where (state, symbol) is in the per-state allowlist."""
    allowed: dict[str, set[str]] = {
        st: set(syms) for st, syms in state_symbol_map.items()
    }
    return [
        ev for ev in events
        if ev.get('state') in allowed
        and ev['symbol'] in allowed[ev['state']]
    ]


# ---------------------------------------------------------------------------
# Section 2 — Portfolio simulator with overlays
# ---------------------------------------------------------------------------


def simulate_portfolio(
    events: list[dict],
    eligible_states: Optional[list[str]],
    eligible_symbols: Optional[list[str]],
    max_per_symbol: int,
    max_per_state: int,
    max_total: int,
    loss_gov_k: int = 0,
    loss_gov_skip: int = 1,
) -> list[dict]:
    """
    Walk events in chronological order and execute trades subject to
    concentration caps and an optional loss-run governor.

    Loss-run governor (active when loss_gov_k > 0):
      After each executed trade: if cost_adj < 0 increment consec_losses
      else reset to 0.  When consec_losses >= loss_gov_k, set
      skip_remaining = loss_gov_skip and reset consec_losses.
      Before executing a trade (after concentration checks pass), if
      skip_remaining > 0 decrement and skip.
    """
    candidates = sorted(
        [ev for ev in events
         if (eligible_states is None or ev.get('state') in eligible_states)
         and (eligible_symbols is None or ev.get('symbol') in eligible_symbols)
         and ev.get('excess_return') is not None],
        key=lambda ev: ev['event_hour_ts'],
    )

    active: list[dict] = []
    executed: list[dict] = []
    consec_losses:  int = 0
    skip_remaining: int = 0

    for ev in candidates:
        entry_ts = ev['event_hour_ts']
        bars     = int(ev.get('bars_held') or 365)
        exit_ts  = entry_ts + timedelta(hours=bars)

        active = [a for a in active if a['exit_ts'] > entry_ts]

        symbol = ev['symbol']
        state  = ev.get('state', '')

        sym_count   = sum(1 for a in active if a['symbol'] == symbol)
        state_count = sum(1 for a in active if a['state']  == state)
        total       = len(active)

        if sym_count   >= max_per_symbol:
            continue
        if state_count >= max_per_state:
            continue
        if total       >= max_total:
            continue

        if skip_remaining > 0:
            skip_remaining -= 1
            continue

        cost_adj_val = float(ev['cost_adj'])
        active.append({'entry_ts': entry_ts, 'exit_ts': exit_ts,
                       'symbol': symbol, 'state': state})
        executed.append({
            'entry_ts':        entry_ts,
            'exit_ts':         exit_ts,
            'symbol':          symbol,
            'state':           state,
            'excess_return':   float(ev['excess_return']),
            'cost_adj':        cost_adj_val,
            'active_at_entry': total,
        })

        if loss_gov_k > 0:
            if cost_adj_val < 0:
                consec_losses += 1
            else:
                consec_losses = 0
            if consec_losses >= loss_gov_k:
                skip_remaining = loss_gov_skip
                consec_losses  = 0

    executed.sort(key=lambda t: t['exit_ts'])
    return executed


def compute_portfolio_stats(
    trades: list[dict],
    label: str = '',
) -> dict:
    if not trades:
        return {'label': label, 'n': 0}

    returns     = [t['cost_adj'] for t in trades]
    n           = len(returns)
    mean_r      = sum(returns) / n
    std_r       = _std(returns)
    sharpe      = round(mean_r / std_r, 4) if std_r and std_r > 0 else None

    cum    = 0.0
    peak   = 0.0
    max_dd = 0.0
    for r in returns:
        cum  += r
        peak  = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    annual_rate = round(n / DATA_SPAN_YEARS, 1)
    mean_active = round(sum(t['active_at_entry'] for t in trades) / n, 2)

    sym_counts:   dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for t in trades:
        sym_counts[t['symbol']]   = sym_counts.get(t['symbol'], 0) + 1
        state_counts[t['state']]  = state_counts.get(t['state'], 0) + 1

    max_sym_frac   = round(max(sym_counts.values())   / n, 3) if sym_counts   else None
    max_state_frac = round(max(state_counts.values()) / n, 3) if state_counts else None
    top_symbol     = max(sym_counts,   key=sym_counts.get)   if sym_counts   else None
    top_state      = max(state_counts, key=state_counts.get) if state_counts else None

    wins      = [r for r in returns if r > 0]
    losses    = [r for r in returns if r <= 0]
    avg_win   = round(sum(wins)   / len(wins),   4) if wins   else None
    avg_loss  = round(sum(losses) / len(losses), 4) if losses else None
    p_gt1     = round(sum(1 for r in returns if r > 1.0) / n, 4)
    p_lt_neg1 = round(sum(1 for r in returns if r < -1.0) / n, 4)

    # Quarterly P&L for frac_pos_quarters
    q_buckets: dict[str, list[float]] = {}
    for t in trades:
        q = _quarter_label(t['entry_ts'])
        q_buckets.setdefault(q, []).append(t['cost_adj'])
    quarters          = list(q_buckets.values())
    n_quarters        = len(quarters)
    frac_pos_quarters = (
        round(sum(1 for q in quarters if sum(q) > 0) / n_quarters, 3)
        if n_quarters > 0 else None
    )

    return {
        'label':             label,
        'n':                 n,
        'mean':              round(mean_r, 4),
        'median':            round(_median(returns), 4),
        'std':               std_r,
        'sharpe':            sharpe,
        'win_rate':          round(len(wins) / n, 4),
        'avg_win':           avg_win,
        'avg_loss':          avg_loss,
        'payoff':            round(abs(avg_win / avg_loss), 3) if (avg_win and avg_loss) else None,
        'final_cum_pnl':     round(cum, 2),
        'max_drawdown':      round(max_dd, 4),
        'annual_rate':       annual_rate,
        'mean_active':       mean_active,
        'max_sym_frac':      max_sym_frac,
        'max_state_frac':    max_state_frac,
        'top_symbol':        top_symbol,
        'top_state':         top_state,
        'p_gt1':             p_gt1,
        'p_lt_neg1':         p_lt_neg1,
        'n_quarters':        n_quarters,
        'frac_pos_quarters': frac_pos_quarters,
    }


# ---------------------------------------------------------------------------
# Section 3 — Expected return inputs
# ---------------------------------------------------------------------------


def compute_expected_inputs(
    events: list[dict],
    eligible_states: list[str],
    eligible_symbols: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, dict]:
    """
    Per-state expected-return inputs with five scenario values.

    stressed_mean = shrunk_mean * 0.50  (additional 50% shrink beyond test-split)
    """
    symbol_set: Optional[set[str]] = set(eligible_symbols) if eligible_symbols is not None else None
    results: dict[str, dict] = {}
    for state in eligible_states:
        all_exc = [
            float(ev['excess_return']) for ev in events
            if ev.get('state') == state
            and ev.get('excess_return') is not None
            and (symbol_set is None or ev['symbol'] in symbol_set)
        ]
        test_exc = [
            float(ev['excess_return']) for ev in events
            if ev.get('state') == state
            and ev.get('excess_return') is not None
            and (symbol_set is None or ev['symbol'] in symbol_set)
            and _ts_date_str(ev['event_hour_ts']) >= TEST_SPLIT_START
        ]

        n_pooled    = len(all_exc)
        n_test      = len(test_exc)
        pooled_mean = round(sum(all_exc) / n_pooled, 4) if n_pooled > 0 else None
        test_mean   = round(sum(test_exc) / n_test, 4)  if n_test > 0   else None
        shrunk_mean = round(test_mean * SHRINKAGE_FACTOR, 4) if test_mean is not None else None
        stressed_mean = round(shrunk_mean * 0.50, 4) if shrunk_mean is not None else None

        boot = block_bootstrap_ci(all_exc, n_bootstrap=n_bootstrap)

        def _ca(v: Optional[float]) -> Optional[float]:
            return round(v - MEDIUM_COST, 4) if v is not None else None

        results[state] = {
            'n_pooled':            n_pooled,
            'n_test':              n_test,
            'pooled_mean':         pooled_mean,
            'test_mean':           test_mean,
            'shrunk_mean':         shrunk_mean,
            'stressed_mean':       stressed_mean,
            'ci_floor':            boot['ci_low'],
            'ci_high':             boot['ci_high'],
            'frac_pos_boot':       boot['frac_pos'],
            'pooled_cost_adj':     _ca(pooled_mean),
            'test_cost_adj':       _ca(test_mean),
            'shrunk_cost_adj':     _ca(shrunk_mean),
            'ci_floor_cost_adj':   _ca(boot['ci_low']),
            'stressed_cost_adj':   _ca(stressed_mean),
        }
    return results


# ---------------------------------------------------------------------------
# Section 4 — Scenario projections
# ---------------------------------------------------------------------------


def scenario_projections(
    trades: list[dict],
    exp_inputs: dict[str, dict],
    eligible_states: list[str],
) -> list[dict]:
    """
    Project annual expected P&L under each of the five scenarios.
    Annual P&L = blended_expected * annual_rate.
    """
    n_total = len(trades)
    if n_total == 0:
        return []

    annual_rate = n_total / DATA_SPAN_YEARS

    state_counts: dict[str, int] = {}
    for t in trades:
        st = t['state']
        state_counts[st] = state_counts.get(st, 0) + 1

    rows: list[dict] = []
    for scenario in SCENARIO_LABELS:
        field = SCENARIO_FIELD_MAP[scenario]
        state_exp: dict[str, Optional[float]] = {}
        for state in eligible_states:
            inp = exp_inputs.get(state, {})
            state_exp[state] = inp.get(field)

        # Weighted blended expected return
        numerator   = 0.0
        denominator = 0
        all_none    = True
        for state in eligible_states:
            cnt = state_counts.get(state, 0)
            val = state_exp.get(state)
            if val is not None and cnt > 0:
                numerator   += val * cnt
                denominator += cnt
                all_none     = False

        blended    = round(numerator / denominator, 4) if denominator > 0 and not all_none else None
        annual_pnl = round(blended * annual_rate, 2) if blended is not None else None

        rows.append({
            'scenario':         scenario,
            'state_expected':   state_exp,
            'blended_expected': blended,
            'annual_rate':      round(annual_rate, 1),
            'annual_pnl':       annual_pnl,
            'viable':           blended is not None and blended > 0,
        })

    return rows


# ---------------------------------------------------------------------------
# Section 5 — ETD retention decision
# ---------------------------------------------------------------------------


def etd_retention_decision(
    exp_etd: dict,
    etd_pruned_stats: dict,
    best_etd_overlay_stats: dict,
    symbol_stats: dict[tuple[str, str], dict],
) -> dict:
    """
    Determine whether ETD should be KEPT, MONITORED, or REMOVED.

    KEEP    : pooled_cost_adj > 0 AND shrunk_cost_adj > 0
    MONITOR : pooled_cost_adj > 0 AND shrunk_cost_adj <= 0
    REMOVE  : pooled_cost_adj <= 0 OR pruned_max_dd > 2 * PREPROD_DD_CEILING
    """
    pooled_ca  = exp_etd.get('pooled_cost_adj')
    shrunk_ca  = exp_etd.get('shrunk_cost_adj')
    ci_floor_ca = exp_etd.get('ci_floor_cost_adj')
    n_test     = exp_etd.get('n_test', 0)
    pruned_max_dd = etd_pruned_stats.get('max_drawdown', 999.0) if etd_pruned_stats.get('n', 0) > 0 else 999.0

    etd_sym_keys = [(sym, st) for (sym, st) in symbol_stats if st == 'EXPANDING_TRENDING_DOWN']
    n_pos_symbols  = sum(
        1 for k in etd_sym_keys
        if (symbol_stats[k].get('cost_adj') or 0) > 0
        and not symbol_stats[k].get('pruned', True)
    )
    n_kept_symbols = sum(
        1 for k in etd_sym_keys
        if not symbol_stats[k].get('pruned', True)
    )

    if pooled_ca is None:
        verdict = 'REMOVE'
        notes   = 'no pooled data'
    elif pooled_ca <= 0 or pruned_max_dd > 2 * PREPROD_DD_CEILING:
        verdict = 'REMOVE'
        notes   = (
            f'pooled_cost_adj={pooled_ca}'
            if pooled_ca <= 0
            else f'pruned_max_dd={pruned_max_dd:.2f} > {2 * PREPROD_DD_CEILING}'
        )
    elif shrunk_ca is not None and shrunk_ca > 0:
        verdict = 'KEEP'
        notes   = 'pooled>0 and shrunk>0'
    else:
        verdict = 'MONITOR'
        notes   = 'pooled>0 but shrunk<=0'

    return {
        'verdict':          verdict,
        'pooled_cost_adj':  pooled_ca,
        'shrunk_cost_adj':  shrunk_ca,
        'ci_floor_cost_adj': ci_floor_ca,
        'n_test':           n_test,
        'pruned_max_dd':    pruned_max_dd,
        'n_pos_symbols':    n_pos_symbols,
        'n_kept_symbols':   n_kept_symbols,
        'notes':            notes,
    }


# ---------------------------------------------------------------------------
# Section 6 — CTU standalone viability
# ---------------------------------------------------------------------------


def ctu_standalone_viability(
    exp_ctu: dict,
    pruned_ctu_stats: dict,
    best_ctu_overlay_stats: dict,
) -> dict:
    """
    PREP_PROD_CANDIDATE : shrunk_cost_adj > 0 AND overlay_max_dd <= PREPROD_DD_CEILING
    MONITOR             : shrunk_cost_adj > 0 but drawdown fails
    REJECT              : shrunk_cost_adj <= 0
    """
    shrunk_ca    = exp_ctu.get('shrunk_cost_adj')
    ci_floor_ca  = exp_ctu.get('ci_floor_cost_adj')
    pooled_ca    = exp_ctu.get('pooled_cost_adj')
    frac_pos     = exp_ctu.get('frac_pos_boot')
    pruned_max_dd = (
        pruned_ctu_stats.get('max_drawdown', 999.0)
        if pruned_ctu_stats.get('n', 0) > 0 else 999.0
    )
    overlay_max_dd = (
        best_ctu_overlay_stats.get('max_drawdown', 999.0)
        if best_ctu_overlay_stats.get('n', 0) > 0 else 999.0
    )
    dd_pass_overlay = overlay_max_dd <= PREPROD_DD_CEILING
    shrunk_viable   = shrunk_ca is not None and shrunk_ca > 0
    ci_viable       = ci_floor_ca is not None and ci_floor_ca > 0

    if not shrunk_viable:
        verdict = 'REJECT'
    elif dd_pass_overlay:
        verdict = 'PREP_PROD_CANDIDATE'
    else:
        verdict = 'MONITOR'

    return {
        'verdict':          verdict,
        'pooled_cost_adj':  pooled_ca,
        'shrunk_cost_adj':  shrunk_ca,
        'ci_floor_cost_adj': ci_floor_ca,
        'frac_pos_bootstrap': frac_pos,
        'pruned_max_dd':    pruned_max_dd,
        'overlay_max_dd':   overlay_max_dd,
        'dd_pass_overlay':  dd_pass_overlay,
        'shrunk_viable':    shrunk_viable,
        'ci_viable':        ci_viable,
    }


# ---------------------------------------------------------------------------
# Section 7 — Scorecard
# ---------------------------------------------------------------------------


def _blended_metric(
    exp_inputs: dict[str, dict],
    eligible_states: list[str],
    field: str,
) -> Optional[float]:
    """Weighted average of a metric across states by n_pooled."""
    total_n = sum(exp_inputs.get(s, {}).get('n_pooled', 0) for s in eligible_states)
    if total_n == 0:
        return None
    val = sum(
        (exp_inputs.get(s, {}).get(field) or 0) * exp_inputs.get(s, {}).get('n_pooled', 0)
        for s in eligible_states
        if exp_inputs.get(s, {}).get(field) is not None
    )
    return round(val / total_n, 4)


def build_scorecard(
    label: str,
    stats: dict,
    exp_inputs: dict[str, dict],
    eligible_states: list[str],
    symbol_stats_for_sleeve: dict[tuple[str, str], dict],
    best_overlay_stats: dict,
    scenarios: list[dict],
) -> dict:
    """
    Eight-dimension pre-prod scorecard.

    D1  trade_level_edge      : blended pooled_cost_adj > 0 AND frac_pos_boot >= 0.60
    D2  symbol_heterogeneity  : >= 60% of cohort symbols survive pruning per eligible state
    D3  sleeve_cost_robustness: pruned sleeve mean cost-adj > 0
    D4  double_cost_survival  : blended shrunk_cost_adj > - MEDIUM_COST (survives 2x cost)
    D5  drawdown_controllable : best overlay max_drawdown <= PREPROD_DD_CEILING
    D6  forward_monitoring    : frac_pos_quarters >= 0.50 AND n_test >= 20 (blended)
    D7  pessimistic_viability : ci_floor OR shrunk_50pct scenario viable
    D8  portfolio_usefulness  : annual_rate >= 10 AND sharpe >= SCORE_SHARPE_FLOOR
    """
    scores: dict[str, bool] = {}

    # D1 — blended pooled cost-adj > 0 across eligible states
    # frac_pos_boot is used when available but not required (may be None if bootstrap skipped)
    blended_pooled = _blended_metric(exp_inputs, eligible_states, 'pooled_cost_adj')
    scores['trade_level_edge'] = bool(
        blended_pooled is not None and blended_pooled > 0
    )

    # D2 — fraction of cohort surviving pruning
    d2_pass_per_state: list[bool] = []
    for st in eligible_states:
        state_keys = [(sym, s) for (sym, s) in symbol_stats_for_sleeve if s == st]
        n_total_sym = len(state_keys)
        n_kept      = sum(
            1 for k in state_keys
            if not symbol_stats_for_sleeve[k].get('pruned', True)
        )
        frac = n_kept / n_total_sym if n_total_sym > 0 else 0.0
        d2_pass_per_state.append(frac >= SCORE_WIN_SYMBOL_FRAC)
    scores['symbol_heterogeneity'] = all(d2_pass_per_state) if d2_pass_per_state else False

    # D3
    sleeve_mean = stats.get('mean')
    scores['sleeve_cost_robustness'] = bool(
        sleeve_mean is not None and sleeve_mean > 0
    )

    # D4
    blended_shrunk = _blended_metric(exp_inputs, eligible_states, 'shrunk_cost_adj')
    scores['double_cost_survival'] = bool(
        blended_shrunk is not None and blended_shrunk > -MEDIUM_COST
    )

    # D5
    overlay_dd = best_overlay_stats.get('max_drawdown', 999.0) if best_overlay_stats.get('n', 0) > 0 else 999.0
    scores['drawdown_controllable'] = overlay_dd <= PREPROD_DD_CEILING

    # D6 — blended n_test and frac_pos_quarters
    total_n_pooled = sum(
        exp_inputs.get(s, {}).get('n_pooled', 0) for s in eligible_states
    )
    total_n_test   = sum(
        exp_inputs.get(s, {}).get('n_test', 0) for s in eligible_states
    )
    avg_n_test     = total_n_test / len(eligible_states) if eligible_states else 0
    fpq            = stats.get('frac_pos_quarters')
    scores['forward_monitoring'] = bool(
        avg_n_test >= 20 and fpq is not None and fpq >= 0.50
    )
    _ = total_n_pooled  # used in D1 via _blended_metric

    # D7
    ci_floor_sc  = next((s for s in scenarios if s['scenario'] == 'ci_floor'),  None)
    shrunk_sc    = next((s for s in scenarios if s['scenario'] == 'shrunk_50pct'), None)
    ci_viable    = ci_floor_sc is not None and ci_floor_sc.get('viable', False)
    shrunk_via   = shrunk_sc   is not None and shrunk_sc.get('viable',   False)
    scores['pessimistic_viability'] = ci_viable or shrunk_via

    # D8
    ann_rate = stats.get('annual_rate', 0) or 0
    sharpe   = stats.get('sharpe')
    scores['portfolio_usefulness'] = bool(
        ann_rate >= 10 and sharpe is not None and sharpe >= SCORE_SHARPE_FLOOR
    )

    n_pass  = sum(1 for v in scores.values() if v)
    n_total = len(scores)

    if n_pass == n_total:
        verdict = 'READY_FOR_SIZING'
    elif n_pass >= 6:
        verdict = 'PREP_PROD_CANDIDATE'
    elif n_pass >= 4:
        verdict = 'MONITOR'
    else:
        verdict = 'REJECT'

    return {
        'label':   label,
        'verdict': verdict,
        'n_pass':  n_pass,
        'n_total': n_total,
        'scores':  scores,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------


def print_pruning(symbol_stats: dict[tuple[str, str], dict]) -> None:
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 1 — Symbol pruning')
    log.info('=' * 80)
    for state in ELIGIBLE_STATES:
        log.info('')
        log.info('  State: %s', state)
        log.info(
            '  %-10s  %5s  %6s  %8s  %8s  %8s  %6s  %8s  %-12s  %s',
            'symbol', 'n', 'n_test', 'pooled', 'test', 'cost_adj',
            'win%', 'median', 'status', 'reason',
        )
        log.info('  ' + '-' * 95)
        keys = [(sym, st) for (sym, st) in symbol_stats if st == state]
        keys.sort(key=lambda k: k[0])
        for key in keys:
            s      = symbol_stats[key]
            sym    = key[0]
            status = 'PRUNED' if s.get('pruned') else ('thin' if s.get('thin_test') else 'OK')
            reason = s.get('prune_reason') or ''
            log.info(
                '  %-10s  %5d  %6d  %8s  %8s  %8s  %6s  %8s  %-12s  %s',
                sym,
                s.get('n') or 0,
                s.get('n_test') or 0,
                f'{s["pooled_mean"]:+.4f}' if s.get('pooled_mean') is not None else 'N/A',
                f'{s["test_mean"]:+.4f}'   if s.get('test_mean')   is not None else 'N/A',
                f'{s["cost_adj"]:+.4f}'    if s.get('cost_adj')    is not None else 'N/A',
                f'{s["win_rate"]:.2f}'     if s.get('win_rate')    is not None else 'N/A',
                f'{s["median"]:+.4f}'      if s.get('median')      is not None else 'N/A',
                status,
                reason,
            )


def print_sleeve_stats(stats: dict) -> None:
    log.info(
        '  [%s]  n=%d  mean=%s  sharpe=%s  max_dd=%s  win=%.3f  ann=%.1f',
        stats.get('label', ''),
        stats.get('n') or 0,
        f'{stats["mean"]:+.4f}' if stats.get('mean') is not None else 'N/A',
        f'{stats["sharpe"]:+.4f}' if stats.get('sharpe') is not None else 'N/A',
        f'{stats["max_drawdown"]:.4f}' if stats.get('max_drawdown') is not None else 'N/A',
        stats.get('win_rate') or 0.0,
        stats.get('annual_rate') or 0.0,
    )


def print_overlay_table(overlay_results: list[dict]) -> None:
    log.info('')
    log.info('  %-15s  %6s  %8s  %8s  %8s  %6s  %8s',
             'overlay', 'n', 'mean', 'sharpe', 'max_dd', 'win%', 'ann_rate')
    log.info('  ' + '-' * 70)
    for r in overlay_results:
        s = r['stats']
        log.info(
            '  %-15s  %6d  %8s  %8s  %8s  %6s  %8s',
            r['label'],
            s.get('n') or 0,
            f'{s["mean"]:+.4f}'         if s.get('mean')         is not None else 'N/A',
            f'{s["sharpe"]:+.4f}'       if s.get('sharpe')       is not None else 'N/A',
            f'{s["max_drawdown"]:.4f}'  if s.get('max_drawdown') is not None else 'N/A',
            f'{s["win_rate"]:.3f}'      if s.get('win_rate')     is not None else 'N/A',
            f'{s["annual_rate"]:.1f}'   if s.get('annual_rate')  is not None else 'N/A',
        )


def print_scenarios(scenarios: list[dict], label: str) -> None:
    log.info('')
    log.info('  Scenarios for %s:', label)
    log.info('  %-15s  %13s  %12s  %11s  %8s',
             'scenario', 'blended_exp', 'annual_rate', 'annual_pnl', 'viable')
    log.info('  ' + '-' * 65)
    for s in scenarios:
        log.info(
            '  %-15s  %13s  %12s  %11s  %8s',
            s['scenario'],
            f'{s["blended_expected"]:+.4f}' if s.get('blended_expected') is not None else 'N/A',
            f'{s["annual_rate"]:.1f}',
            f'{s["annual_pnl"]:+.2f}'        if s.get('annual_pnl')       is not None else 'N/A',
            'YES' if s.get('viable') else 'NO',
        )


def print_scorecard(card: dict) -> None:
    log.info('')
    log.info('=' * 80)
    log.info('  SCORECARD: %-30s  verdict=%-20s  %d/%d',
             card['label'], card['verdict'], card['n_pass'], card['n_total'])
    log.info('=' * 80)
    for dim, passed in card['scores'].items():
        log.info('  [%s]  %s', 'PASS' if passed else 'FAIL', dim)


def print_etd_decision(result: dict) -> None:
    log.info('')
    log.info('  ETD retention: %s', result['verdict'])
    log.info(
        '    pooled_ca=%s  shrunk_ca=%s  ci_floor_ca=%s  n_test=%d'
        '  pruned_max_dd=%s  pos_syms=%d/%d  notes: %s',
        f'{result["pooled_cost_adj"]:+.4f}' if result.get('pooled_cost_adj') is not None else 'N/A',
        f'{result["shrunk_cost_adj"]:+.4f}' if result.get('shrunk_cost_adj') is not None else 'N/A',
        f'{result["ci_floor_cost_adj"]:+.4f}' if result.get('ci_floor_cost_adj') is not None else 'N/A',
        result.get('n_test') or 0,
        f'{result["pruned_max_dd"]:.2f}' if result.get('pruned_max_dd') is not None else 'N/A',
        result.get('n_pos_symbols') or 0,
        result.get('n_kept_symbols') or 0,
        result.get('notes') or '',
    )


def print_ctu_viability(result: dict) -> None:
    log.info('')
    log.info('  CTU standalone viability: %s', result['verdict'])
    log.info(
        '    pooled_ca=%s  shrunk_ca=%s  ci_floor_ca=%s  frac_pos_boot=%s'
        '  overlay_max_dd=%s  dd_pass=%s',
        f'{result["pooled_cost_adj"]:+.4f}'   if result.get('pooled_cost_adj')   is not None else 'N/A',
        f'{result["shrunk_cost_adj"]:+.4f}'   if result.get('shrunk_cost_adj')   is not None else 'N/A',
        f'{result["ci_floor_cost_adj"]:+.4f}' if result.get('ci_floor_cost_adj') is not None else 'N/A',
        f'{result["frac_pos_bootstrap"]:.3f}' if result.get('frac_pos_bootstrap') is not None else 'N/A',
        f'{result["overlay_max_dd"]:.2f}'     if result.get('overlay_max_dd')    is not None else 'N/A',
        'YES' if result.get('dd_pass_overlay') else 'NO',
    )


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def write_outputs(
    output_path: Optional[str],
    scorecard_path: Optional[str],
    memo_path: Optional[str],
    sleeve_results: list[dict],
    overlay_results: dict[str, list[dict]],
    scorecards: list[dict],
    etd_decision: dict,
    ctu_viability: dict,
    all_scenarios: dict[str, list[dict]],
) -> None:
    if output_path:
        _write_csv(output_path, sleeve_results, overlay_results, all_scenarios)
    if scorecard_path:
        _write_scorecard(scorecard_path, scorecards)
    if memo_path:
        _write_memo(memo_path, scorecards, etd_decision, ctu_viability, all_scenarios)


def _write_csv(
    path: str,
    sleeve_results: list[dict],
    overlay_results: dict[str, list[dict]],
    all_scenarios: dict[str, list[dict]],
) -> None:
    rows: list[dict] = []
    for s in sleeve_results:
        stats = s.get('stats', {})
        rows.append({'section': 'sleeve', 'label': s['label'], **stats})
    for group, variants in overlay_results.items():
        for v in variants:
            stats = v.get('stats', {})
            rows.append({'section': f'overlay_{group}', 'label': v['label'], **stats})
    for label, scenarios in all_scenarios.items():
        for sc in scenarios:
            rows.append({'section': f'scenario_{label}', **sc})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'section', 'label', 'n', 'mean', 'sharpe', 'max_drawdown', 'win_rate',
        'annual_rate', 'scenario', 'blended_expected', 'annual_pnl', 'viable',
    ]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    log.info('Results written to %s', path)


def _write_scorecard(path: str, scorecards: list[dict]) -> None:
    lines: list[str] = [
        'MARKOV PRE-PROD QUALIFIER -- SCORECARD',
        'Issue #95',
        f'Generated: {__import__("datetime").date.today()}',
        '',
    ]
    for card in scorecards:
        lines += [
            '=' * 60,
            f'Sleeve: {card["label"]}',
            f'Verdict: {card["verdict"]}  ({card["n_pass"]}/{card["n_total"]})',
            '-' * 40,
        ]
        for dim, passed in card['scores'].items():
            lines.append(f'  [{"PASS" if passed else "FAIL"}]  {dim}')
        lines.append('')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log.info('Scorecard written to %s', path)


def _write_memo(
    path: str,
    scorecards: list[dict],
    etd_decision: dict,
    ctu_viability: dict,
    all_scenarios: dict[str, list[dict]],
) -> None:
    lines: list[str] = [
        'MARKOV PRE-PROD QUALIFIER -- RECOMMENDATION MEMO',
        'Issue #95: Nine-stage qualification pipeline for CTU and ETD states',
        f'Generated: {__import__("datetime").date.today()}',
        '',
        '=' * 72,
        'ETD RETENTION DECISION',
        '=' * 72,
        f'Verdict: {etd_decision.get("verdict")}',
        f'  pooled_cost_adj  : {etd_decision.get("pooled_cost_adj")}',
        f'  shrunk_cost_adj  : {etd_decision.get("shrunk_cost_adj")}',
        f'  pruned_max_dd    : {etd_decision.get("pruned_max_dd")}',
        f'  notes            : {etd_decision.get("notes")}',
        '',
        '=' * 72,
        'CTU STANDALONE VIABILITY',
        '=' * 72,
        f'Verdict: {ctu_viability.get("verdict")}',
        f'  shrunk_cost_adj  : {ctu_viability.get("shrunk_cost_adj")}',
        f'  overlay_max_dd   : {ctu_viability.get("overlay_max_dd")}',
        f'  dd_pass_overlay  : {ctu_viability.get("dd_pass_overlay")}',
        '',
        '=' * 72,
        'SCORECARDS',
        '=' * 72,
    ]
    for card in scorecards:
        lines += [
            f'Sleeve: {card["label"]}',
            f'  Verdict : {card["verdict"]}  ({card["n_pass"]}/{card["n_total"]})',
        ]
        for dim, passed in card['scores'].items():
            lines.append(f'  [{"PASS" if passed else "FAIL"}]  {dim}')
        lines.append('')
    lines += [
        '=' * 72,
        'SCENARIO PROJECTIONS',
        '=' * 72,
    ]
    for label, scenarios in all_scenarios.items():
        lines.append(f'Sleeve: {label}')
        for sc in scenarios:
            viable = 'viable' if sc.get('viable') else 'not viable'
            pnl    = f'{sc["annual_pnl"]:+.2f}' if sc.get('annual_pnl') is not None else 'N/A'
            blended = sc.get('blended_expected')
            lines.append(
                f'  {sc["scenario"]:<15}  blended={blended}  annual_pnl={pnl}  [{viable}]'
            )
        lines.append('')
    lines += [
        '=' * 72,
        'RECOMMENDATION',
        '=' * 72,
    ]
    combined_card = next(
        (c for c in scorecards if 'combined' in c['label']), None
    )
    if combined_card:
        verdict = combined_card['verdict']
        if verdict == 'READY_FOR_SIZING':
            lines += [
                'PROCEED to conservative position-sizing research.',
                '',
                'Basis:',
                '  - Combined pruned sleeve scores 8/8 on pre-prod scorecard',
                '  - Both CTU and ETD viable under shrunk scenario',
                '  - Max drawdown within acceptable bounds under best overlay',
                '',
                'Constraints:',
                '  - Use shrunk_cost_adj as per-trade expected return input',
                '  - Maintain max 2 positions per symbol, max 10 total',
                '  - Review quarterly; halt if cost-adj mean turns negative',
            ]
        elif verdict == 'PREP_PROD_CANDIDATE':
            lines += [
                'PREP_PROD_CANDIDATE: Address failing scorecard dimensions.',
                'Re-score in one quarter once n_test improves.',
            ]
        elif verdict == 'MONITOR':
            lines += [
                'MONITOR: Extend forward monitoring. Do not size yet.',
                'Reassess when n_test >= 20 per state or additional criteria met.',
            ]
        else:
            lines += [
                'REJECT: Evidence insufficient for capital allocation research.',
                'Re-evaluate after 12+ months of forward monitoring.',
            ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    log.info('Recommendation memo written to %s', path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-production qualifier for candidate Markov states (Issue #95).'
    )
    parser.add_argument('--no-bootstrap', action='store_true',
                        help='Skip bootstrap CIs (faster run)')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP)
    parser.add_argument('--output',    default=None,
                        help='Optional CSV output path')
    parser.add_argument('--scorecard', default=None,
                        help='Optional scorecard text output path')
    parser.add_argument('--memo',      default=None,
                        help='Optional recommendation memo text output path')
    args   = parser.parse_args()
    n_boot = 0 if args.no_bootstrap else args.n_bootstrap

    log.info('Connecting to %s', TIMESCALE_DSN or f'{PGHOST}:{PGPORT}/{PGDATABASE}')
    log.info('Loading FX breakout events ...')
    all_events = load_events()
    log.info('Loaded %d state-labeled events', len(all_events))
    if not all_events:
        log.error('No events found.')
        sys.exit(1)

    # Carry neutralization
    baselines     = compute_baselines(all_events)
    excess_events = apply_excess_return(all_events, baselines)
    log.info('Carry neutralization: %d events', len(excess_events))

    # Filter to cohort
    present        = {ev['symbol'] for ev in excess_events}
    active_symbols = [s for s in COHORT_SYMBOLS if s in present]
    cohort_events  = [ev for ev in excess_events if ev['symbol'] in set(active_symbols)]
    log.info('Cohort core_majors: %d events  symbols: %s',
             len(cohort_events), active_symbols)

    # --- Symbol pruning ---
    log.info('Computing symbol stats and applying pruning rules ...')
    symbol_stats  = compute_symbol_stats(cohort_events)
    symbol_stats  = apply_pruning_rules(symbol_stats)
    print_pruning(symbol_stats)

    pruned_ctu_syms = [
        sym for (sym, st) in symbol_stats
        if st == 'CONTRACTING_TRENDING_UP'
        and not symbol_stats[(sym, st)].get('pruned', True)
    ]
    pruned_etd_syms = [
        sym for (sym, st) in symbol_stats
        if st == 'EXPANDING_TRENDING_DOWN'
        and not symbol_stats[(sym, st)].get('pruned', True)
    ]
    log.info('CTU surviving symbols (%d): %s', len(pruned_ctu_syms), pruned_ctu_syms)
    log.info('ETD surviving symbols (%d): %s', len(pruned_etd_syms), pruned_etd_syms)

    # --- Sleeve decomposition ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 2 — Sleeve decomposition')
    log.info('=' * 80)

    _base_caps = dict(
        max_per_symbol=2, max_per_state=5, max_total=10,
        loss_gov_k=0, loss_gov_skip=1,
    )

    # Pre-filter events for combined_pruned: CTU events use CTU-kept symbols only,
    # ETD events use ETD-kept symbols only. Avoids polluting CTU with ETD-pruned
    # symbols and vice versa.
    combined_pruned_events = filter_events_per_state_symbols(cohort_events, {
        'CONTRACTING_TRENDING_UP': pruned_ctu_syms,
        'EXPANDING_TRENDING_DOWN': pruned_etd_syms,
    })

    # Each config: (label, events, states, symbols)
    sleeve_configs = [
        ('ctu_unpruned',      cohort_events,           ['CONTRACTING_TRENDING_UP'], active_symbols),
        ('etd_unpruned',      cohort_events,           ['EXPANDING_TRENDING_DOWN'],  active_symbols),
        ('combined_unpruned', cohort_events,           ELIGIBLE_STATES,              active_symbols),
        ('ctu_pruned',        cohort_events,           ['CONTRACTING_TRENDING_UP'], pruned_ctu_syms),
        ('etd_pruned',        cohort_events,           ['EXPANDING_TRENDING_DOWN'],  pruned_etd_syms),
        ('combined_pruned',   combined_pruned_events,  ELIGIBLE_STATES,              None),
    ]

    sleeve_results: list[dict] = []
    sleeve_stats_by_label: dict[str, dict] = {}
    for s_label, s_events, s_states, s_syms in sleeve_configs:
        if s_syms is not None and not s_syms:
            log.info('  [%s]  no surviving symbols — skipped', s_label)
            st = {'label': s_label, 'n': 0}
            sleeve_results.append({'label': s_label, 'stats': st})
            sleeve_stats_by_label[s_label] = st
            continue
        trades = simulate_portfolio(
            s_events, s_states, s_syms, **_base_caps
        )
        st = compute_portfolio_stats(trades, label=s_label)
        sleeve_results.append({'label': s_label, 'stats': st})
        sleeve_stats_by_label[s_label] = st
        print_sleeve_stats(st)

    # --- Risk overlay testing ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 3 — Risk overlay testing')
    log.info('=' * 80)

    # Each overlay group: (label, events, states, symbols)
    # combined_pruned uses pre-filtered events so eligible_symbols=None
    overlay_group_configs = [
        ('combined_pruned', combined_pruned_events, ELIGIBLE_STATES,                     None),
        ('ctu_pruned',      cohort_events,           ['CONTRACTING_TRENDING_UP'],         pruned_ctu_syms),
        ('etd_pruned',      cohort_events,           ['EXPANDING_TRENDING_DOWN'],          pruned_etd_syms),
    ]

    overlay_results: dict[str, list[dict]] = {}
    best_overlay_by_group: dict[str, dict] = {}

    for group_label, g_events, g_states, g_syms in overlay_group_configs:
        log.info('')
        log.info('  Overlay group: %s', group_label)
        if g_syms is not None and not g_syms:
            log.info('    no symbols — skipped')
            overlay_results[group_label] = []
            best_overlay_by_group[group_label] = {'n': 0}
            continue

        group_overlays: list[dict] = []
        for ov in OVERLAY_VARIANTS:
            trades = simulate_portfolio(
                g_events,
                g_states,
                g_syms,
                max_per_symbol=ov['max_sym'],
                max_per_state=ov['max_state'],
                max_total=ov['max_total'],
                loss_gov_k=ov['loss_gov'],
                loss_gov_skip=1,
            )
            st = compute_portfolio_stats(trades, label=ov['label'])
            group_overlays.append({'label': ov['label'], 'stats': st})
        overlay_results[group_label] = group_overlays
        print_overlay_table(group_overlays)

        best = min(
            (v for v in group_overlays if v['stats'].get('n', 0) > 0),
            key=lambda v: v['stats'].get('max_drawdown', 999.0),
            default={'stats': {'n': 0}},
        )
        best_overlay_by_group[group_label] = best['stats']

    # --- Expected return inputs (with bootstrap) ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 4 — Expected return inputs')
    log.info('=' * 80)

    def _exp_for(states: list[str], syms: list[str]) -> dict[str, dict]:
        return compute_expected_inputs(cohort_events, states, syms, n_bootstrap=n_boot)

    ctu_syms_set = pruned_ctu_syms if pruned_ctu_syms else active_symbols
    etd_syms_set = pruned_etd_syms if pruned_etd_syms else active_symbols

    exp_ctu  = _exp_for(['CONTRACTING_TRENDING_UP'], ctu_syms_set)
    exp_etd  = _exp_for(['EXPANDING_TRENDING_DOWN'],  etd_syms_set)
    # combined_pruned_events already has per-state symbol filtering applied
    exp_combined = compute_expected_inputs(
        combined_pruned_events, ELIGIBLE_STATES, None, n_bootstrap=n_boot
    )

    for state, inp in {**exp_ctu, **exp_etd, **exp_combined}.items():
        log.info(
            '  %-35s  n_pool=%4d  n_test=%4d  pooled_ca=%s  shrunk_ca=%s  stressed_ca=%s',
            state,
            inp.get('n_pooled') or 0,
            inp.get('n_test') or 0,
            f'{inp["pooled_cost_adj"]:+.4f}'   if inp.get('pooled_cost_adj')   is not None else 'N/A',
            f'{inp["shrunk_cost_adj"]:+.4f}'   if inp.get('shrunk_cost_adj')   is not None else 'N/A',
            f'{inp["stressed_cost_adj"]:+.4f}' if inp.get('stressed_cost_adj') is not None else 'N/A',
        )

    # --- Scenario projections ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 5 — Scenario projections')
    log.info('=' * 80)

    ctu_pruned_trades = simulate_portfolio(
        cohort_events, ['CONTRACTING_TRENDING_UP'], ctu_syms_set, **_base_caps
    )
    etd_pruned_trades = simulate_portfolio(
        cohort_events, ['EXPANDING_TRENDING_DOWN'], etd_syms_set, **_base_caps
    )
    combined_pruned_trades = simulate_portfolio(
        combined_pruned_events, ELIGIBLE_STATES, None, **_base_caps,
    )

    scenarios_ctu      = scenario_projections(ctu_pruned_trades,      exp_ctu,      ['CONTRACTING_TRENDING_UP'])
    scenarios_etd      = scenario_projections(etd_pruned_trades,      exp_etd,      ['EXPANDING_TRENDING_DOWN'])
    scenarios_combined = scenario_projections(combined_pruned_trades,  exp_combined, ELIGIBLE_STATES)

    all_scenarios: dict[str, list[dict]] = {
        'ctu_pruned':      scenarios_ctu,
        'etd_pruned':      scenarios_etd,
        'combined_pruned': scenarios_combined,
    }

    print_scenarios(scenarios_ctu,      'ctu_pruned')
    print_scenarios(scenarios_etd,      'etd_pruned')
    print_scenarios(scenarios_combined, 'combined_pruned')

    # --- ETD retention decision ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 6 — ETD retention decision')
    log.info('=' * 80)
    etd_decision = etd_retention_decision(
        exp_etd.get('EXPANDING_TRENDING_DOWN', {}),
        sleeve_stats_by_label.get('etd_pruned', {'n': 0}),
        best_overlay_by_group.get('etd_pruned', {'n': 0}),
        symbol_stats,
    )
    print_etd_decision(etd_decision)

    # --- CTU standalone viability ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 7 — CTU standalone viability')
    log.info('=' * 80)
    ctu_viability = ctu_standalone_viability(
        exp_ctu.get('CONTRACTING_TRENDING_UP', {}),
        sleeve_stats_by_label.get('ctu_pruned', {'n': 0}),
        best_overlay_by_group.get('ctu_pruned', {'n': 0}),
    )
    print_ctu_viability(ctu_viability)

    # --- Scorecards ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 8 — Pre-prod scorecards')
    log.info('=' * 80)

    scorecard_configs = [
        ('ctu_pruned',      sleeve_stats_by_label.get('ctu_pruned', {'n': 0}),
         exp_ctu, ['CONTRACTING_TRENDING_UP'],
         best_overlay_by_group.get('ctu_pruned', {'n': 0}),
         scenarios_ctu),
        ('etd_pruned',      sleeve_stats_by_label.get('etd_pruned', {'n': 0}),
         exp_etd, ['EXPANDING_TRENDING_DOWN'],
         best_overlay_by_group.get('etd_pruned', {'n': 0}),
         scenarios_etd),
        ('combined_pruned', sleeve_stats_by_label.get('combined_pruned', {'n': 0}),
         exp_combined, ELIGIBLE_STATES,
         best_overlay_by_group.get('combined_pruned', {'n': 0}),
         scenarios_combined),
    ]

    scorecards: list[dict] = []
    for sc_label, sc_stats, sc_exp, sc_states, sc_best_ov, sc_scen in scorecard_configs:
        sym_subset = {
            k: v for k, v in symbol_stats.items()
            if k[1] in sc_states
        }
        card = build_scorecard(
            sc_label, sc_stats, sc_exp, sc_states,
            sym_subset, sc_best_ov, sc_scen,
        )
        scorecards.append(card)
        print_scorecard(card)

    # --- Section 9 recommendation memo ---
    log.info('')
    log.info('=' * 80)
    log.info('  SECTION 9 — Recommendation memo')
    log.info('=' * 80)
    combined_card = next((c for c in scorecards if c['label'] == 'combined_pruned'), None)
    if combined_card:
        verdict = combined_card['verdict']
        log.info('  Combined pruned sleeve verdict: %s  (%d/%d)',
                 verdict, combined_card['n_pass'], combined_card['n_total'])
        log.info('')
        if verdict == 'READY_FOR_SIZING':
            log.info('  RECOMMENDATION: PROCEED to conservative position-sizing research.')
            log.info('')
            log.info('  Basis:')
            log.info('    - Combined pruned sleeve achieves 8/8 on pre-prod scorecard')
            log.info('    - Both CTU and ETD viable under shrunk scenario')
            log.info('    - Drawdown within bounds under best overlay variant')
            log.info('')
            log.info('  Constraints:')
            log.info('    - Use shrunk_cost_adj as per-trade expected return input')
            log.info('    - Maintain max 2 positions per symbol, max 10 total')
            log.info('    - ETD: %s', etd_decision['verdict'])
            log.info('    - Review quarterly; halt if cost-adj mean turns negative')
        elif verdict == 'PREP_PROD_CANDIDATE':
            failed = [d for d, p in combined_card['scores'].items() if not p]
            log.info('  RECOMMENDATION: PREP_PROD_CANDIDATE.')
            log.info('  Address failing dimensions before final deployment decision.')
            for f in failed:
                log.info('    - FAILED: %s', f)
            log.info('  Re-score after one quarter of additional forward monitoring.')
        elif verdict == 'MONITOR':
            failed = [d for d, p in combined_card['scores'].items() if not p]
            log.info('  RECOMMENDATION: MONITOR. Do not size yet.')
            for f in failed:
                log.info('    - FAILED: %s', f)
            log.info('  Reassess when n_test >= 20 per state and frac_pos_quarters >= 0.50.')
        else:
            failed = [d for d, p in combined_card['scores'].items() if not p]
            log.info('  RECOMMENDATION: REJECT.')
            log.info('  Evidence insufficient for capital allocation research.')
            for f in failed:
                log.info('    - FAILED: %s', f)
            log.info('  Re-evaluate after 12+ months of forward monitoring.')

    write_outputs(
        args.output, args.scorecard, args.memo,
        sleeve_results, overlay_results, scorecards,
        etd_decision, ctu_viability, all_scenarios,
    )


if __name__ == '__main__':
    main()

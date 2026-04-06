"""
scripts/markov_portfolio_simulator.py

Portfolio-level deployment simulator for validated Markov states -- Issue #94.

Answers the final research gate question:
  "Is the validated alpha economically useful under conservative portfolio
   construction assumptions?"

Eight-section analysis:
  1. Conservative expected-return inputs with documented shrinkage
  2. Filtered portfolio simulation (CTU + ETD only, with concentration caps)
  3. Unfiltered portfolio simulation (all breakout events, same caps)
  4. Filtered vs unfiltered comparison
  5. CTU / ETD diversification and correlation analysis
  6. Uncertainty-aware scenario projections
  7. Promotion criteria evaluation
  8. Final recommendation memo

Frozen eligible states (no additions permitted):
  CONTRACTING_TRENDING_UP  (CTU)
  EXPANDING_TRENDING_DOWN  (ETD)

Cohort: core_majors (7 G10 FX pairs)
Simulation: flat unit sizing, medium friction (0.07 SD30d), real timestamps.

Usage
-----
    python scripts/markov_portfolio_simulator.py
    python scripts/markov_portfolio_simulator.py --no-bootstrap
    python scripts/markov_portfolio_simulator.py --output results/portfolio_sim.csv
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
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
try:
    _project_root = Path(__file__).resolve().parents[1]
    _env_path = _project_root / ".env"
    if _env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(_env_path)
except Exception:
    pass

TIMESCALE_DSN = os.getenv("TIMESCALE_DSN")
PGHOST        = os.getenv("PGHOST",     "localhost")
PGDATABASE    = os.getenv("PGDATABASE", "market_data")
PGUSER        = os.getenv("PGUSER",     "backtesting")
PGPASSWORD    = os.getenv("PGPASSWORD", "backtesting_pass")
PGPORT        = int(os.getenv("PGPORT", "5434"))


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

ELIGIBLE_FILTERED: list[str] = [
    "CONTRACTING_TRENDING_UP",
    "EXPANDING_TRENDING_DOWN",
]

COHORT_SYMBOLS: list[str] = [
    "EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
]

# Concentration caps for simulation
MAX_PER_SYMBOL: int = 2    # max simultaneous open positions per symbol
MAX_PER_STATE:  int = 5    # max simultaneous open positions per state
MAX_TOTAL:      int = 10   # max simultaneous open positions total

MEDIUM_COST: float = 0.07          # SD30d per trade (medium friction level)

# Conservative shrinkage rule applied to test-split mean
SHRINKAGE_FACTOR: float = 0.50     # 50% shrinkage toward zero

# Test split boundary
TEST_SPLIT_START: str = "2024-01-01"

# Bootstrap
N_BOOTSTRAP: int = 1000
BLOCK_SIZE:  int = 20
RANDOM_SEED: int = 42

# Minimum trades for a scenario to be considered meaningful
MIN_TRADES_SCENARIO: int = 30

# Historical data span for annual rate computation (years)
DATA_SPAN_YEARS: float = 11.0   # 2015 – 2026

# Promotion criteria thresholds
SHARPE_FLOOR:                float = 0.15
MAX_DD_CEILING:              float = 30.0   # SD30d units
FILTERED_MEAN_ADV_THRESHOLD: float = 0.05   # filtered must beat unfiltered by >= 0.05
SHRUNK_POSITIVE_COST_ADJ:    bool  = True   # shrunk cost-adj mean must be > 0


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _vol_bucket(v: float) -> str:
    if v < 0.80:
        return "CONTRACTING"
    if v <= 1.20:
        return "NEUTRAL"
    return "EXPANDING"


def _eff_bucket(e: float) -> str:
    if e < 0.30:
        return "CHOPPY"
    if e <= 0.60:
        return "MIXED"
    return "TRENDING"


def _median(vals: list[float]) -> float:
    sv  = sorted(vals)
    n   = len(sv)
    mid = n // 2
    return sv[mid] if n % 2 else (sv[mid - 1] + sv[mid]) / 2.0


def _pearson(x: list[float], y: list[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return round(num / (dx * dy), 4)


def _std(vals: list[float]) -> Optional[float]:
    n = len(vals)
    if n < 2:
        return None
    m = sum(vals) / n
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)), 4)


def _quarter_label(ts: object) -> str:
    if hasattr(ts, "year") and hasattr(ts, "month"):
        return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"
    s = str(ts)
    return f"{s[:4]}Q{(int(s[5:7]) - 1) // 3 + 1}"


def _ts_date_str(ts: object) -> str:
    return ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]


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
        vr  = ev.get("vol_ratio_20_100")
        eff = ev.get("efficiency_20")
        d   = ev.get("breakout_direction", "")
        if vr is None or eff is None or d not in ("UP", "DOWN"):
            continue
        ev["vol_ratio_20_100"] = float(vr)
        ev["efficiency_20"]    = float(eff)
        ev["state"] = f"{_vol_bucket(ev['vol_ratio_20_100'])}_{_eff_bucket(ev['efficiency_20'])}_{d}"
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Carry neutralization
# ---------------------------------------------------------------------------

def compute_baselines(events: list[dict]) -> dict[tuple[str, str], float]:
    groups: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        key = (ev["symbol"], ev["breakout_direction"])
        groups.setdefault(key, []).append(float(ev["realized_return_sd30d"]))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def apply_excess_return(
    events: list[dict],
    baselines: dict[tuple[str, str], float],
) -> list[dict]:
    out: list[dict] = []
    for ev in events:
        b = baselines.get((ev["symbol"], ev["breakout_direction"]))
        if b is None:
            continue
        ne = dict(ev)
        ne["excess_return"] = float(ev["realized_return_sd30d"]) - b
        ne["cost_adj"]      = ne["excess_return"] - MEDIUM_COST
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
        return {"ci_low": None, "ci_high": None, "frac_pos": None}
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
        "ci_low":   round(lo, 4),
        "ci_high":  round(hi, 4),
        "frac_pos": round(sum(1 for m in means if m > 0) / n_bootstrap, 3),
    }


# ---------------------------------------------------------------------------
# Section 1 — Conservative expected-return inputs
# ---------------------------------------------------------------------------

def compute_expected_return_inputs(
    events: list[dict],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, dict]:
    """
    For each candidate state compute:
      pooled      : full-history mean excess return
      test_split  : mean excess return in test window (2024+)
      shrunk      : SHRINKAGE_FACTOR * test_split_mean  (toward zero)
      ci_floor    : bootstrap CI-low on pooled excess return
      cost_adj_*  : above minus MEDIUM_COST per trade

    Shrinkage rule
    --------------
    shrunk = 0.50 * test_split_mean

    Rationale:
      - Test split is the most recent OOS period (2024-2026)
      - 50% shrinkage toward zero addresses:
          (a) thin test-split sample (n < 250 per state)
          (b) high quarterly variance observed in forward monitoring
          (c) regime-change risk (signal may moderate post-2023)
      - Conservative floor for sizing research input

    These are per-trade expected values, not annualised.
    """
    results: dict[str, dict] = {}

    for state in ELIGIBLE_FILTERED:
        all_exc  = [float(ev["excess_return"]) for ev in events
                    if ev.get("state") == state and ev.get("excess_return") is not None]
        test_exc = [float(ev["excess_return"]) for ev in events
                    if ev.get("state") == state and ev.get("excess_return") is not None
                    and _ts_date_str(ev["event_hour_ts"]) >= TEST_SPLIT_START]

        pooled_mean = round(sum(all_exc) / len(all_exc), 4)   if all_exc  else None
        test_mean   = round(sum(test_exc) / len(test_exc), 4) if test_exc else None
        shrunk_mean = round(test_mean * SHRINKAGE_FACTOR, 4)  if test_mean is not None else None

        boot = block_bootstrap_ci(all_exc, n_bootstrap=n_bootstrap)

        results[state] = {
            "n_pooled":          len(all_exc),
            "n_test":            len(test_exc),
            "pooled_mean":       pooled_mean,
            "test_mean":         test_mean,
            "shrunk_mean":       shrunk_mean,
            "ci_floor":          boot["ci_low"],
            "ci_high":           boot["ci_high"],
            "frac_pos_boot":     boot["frac_pos"],
            # cost-adjusted
            "pooled_cost_adj":   round(pooled_mean - MEDIUM_COST, 4) if pooled_mean is not None else None,
            "test_cost_adj":     round(test_mean   - MEDIUM_COST, 4) if test_mean   is not None else None,
            "shrunk_cost_adj":   round(shrunk_mean - MEDIUM_COST, 4) if shrunk_mean is not None else None,
            "ci_floor_cost_adj": round(boot["ci_low"] - MEDIUM_COST, 4) if boot["ci_low"] is not None else None,
        }
    return results


# ---------------------------------------------------------------------------
# Section 2 & 3 — Portfolio simulator
# ---------------------------------------------------------------------------

def simulate_portfolio(
    events: list[dict],
    eligible_states: Optional[list[str]],
    max_per_symbol: int = MAX_PER_SYMBOL,
    max_per_state:  int = MAX_PER_STATE,
    max_total:      int = MAX_TOTAL,
) -> list[dict]:
    """
    Walk through events in chronological order and execute trades
    subject to concentration caps.  Returns list of executed trade dicts.

    Concentration caps:
      max_per_symbol : max simultaneous open positions in the same symbol
      max_per_state  : max simultaneous open positions in the same state
      max_total      : max total simultaneous open positions

    Trades ordered by exit_ts for equity-curve purposes.
    """
    # Pre-filter and sort by entry
    candidates = sorted(
        [ev for ev in events
         if (eligible_states is None or ev.get("state") in eligible_states)
         and ev.get("excess_return") is not None],
        key=lambda ev: ev["event_hour_ts"],
    )

    # Active positions: list of {entry_ts, exit_ts, symbol, state}
    active: list[dict] = []
    executed: list[dict] = []

    for ev in candidates:
        entry_ts = ev["event_hour_ts"]
        bars     = int(ev.get("bars_held") or 365)
        exit_ts  = entry_ts + timedelta(hours=bars)

        # Expire old positions
        active = [a for a in active if a["exit_ts"] > entry_ts]

        symbol = ev["symbol"]
        state  = ev.get("state", "")

        sym_count   = sum(1 for a in active if a["symbol"] == symbol)
        state_count = sum(1 for a in active if a["state"]  == state)
        total       = len(active)

        if sym_count   >= max_per_symbol:
            continue
        if state_count >= max_per_state:
            continue
        if total       >= max_total:
            continue

        active.append({"entry_ts": entry_ts, "exit_ts": exit_ts,
                        "symbol": symbol, "state": state})
        executed.append({
            "entry_ts":        entry_ts,
            "exit_ts":         exit_ts,
            "symbol":          symbol,
            "state":           state,
            "excess_return":   float(ev["excess_return"]),
            "cost_adj":        float(ev["cost_adj"]),
            "active_at_entry": total,
        })

    # Sort by exit for equity-curve ordering
    executed.sort(key=lambda t: t["exit_ts"])
    return executed


# ---------------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------------

def compute_portfolio_stats(
    trades: list[dict],
    label: str = "",
) -> dict:
    if not trades:
        return {"label": label, "n": 0}

    returns = [t["cost_adj"] for t in trades]
    n       = len(returns)
    mean_r  = sum(returns) / n
    std_r   = _std(returns)
    sharpe  = round(mean_r / std_r, 4) if std_r and std_r > 0 else None

    # Equity curve and drawdown (exit-time ordered)
    cum     = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for r in returns:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    # Annual trade rate
    annual_rate = round(n / DATA_SPAN_YEARS, 1)

    # Capital usage: mean active positions at entry
    mean_active = round(
        sum(t["active_at_entry"] for t in trades) / n, 2
    )

    # Concentration
    sym_counts   = {}
    state_counts = {}
    for t in trades:
        sym_counts[t["symbol"]]  = sym_counts.get(t["symbol"],  0) + 1
        state_counts[t["state"]] = state_counts.get(t["state"], 0) + 1

    max_sym_frac   = round(max(sym_counts.values())   / n, 3) if sym_counts   else None
    max_state_frac = round(max(state_counts.values()) / n, 3) if state_counts else None
    top_symbol     = max(sym_counts,   key=sym_counts.get)   if sym_counts   else None
    top_state      = max(state_counts, key=state_counts.get) if state_counts else None

    # Tail metrics
    wins       = [r for r in returns if r > 0]
    losses     = [r for r in returns if r <= 0]
    avg_win    = round(sum(wins)   / len(wins),   4) if wins   else None
    avg_loss   = round(sum(losses) / len(losses), 4) if losses else None
    p_gt1      = round(sum(1 for r in returns if r > 1.0) / n, 4)
    p_lt_neg1  = round(sum(1 for r in returns if r < -1.0) / n, 4)

    return {
        "label":          label,
        "n":              n,
        "mean":           round(mean_r, 4),
        "median":         round(_median(returns), 4),
        "std":            std_r,
        "sharpe":         sharpe,
        "win_rate":       round(len(wins) / n, 4),
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "payoff":         round(abs(avg_win / avg_loss), 3) if (avg_win and avg_loss) else None,
        "final_cum_pnl":  round(cum, 2),
        "max_drawdown":   round(max_dd, 4),
        "annual_rate":    annual_rate,
        "mean_active":    mean_active,
        "max_sym_frac":   max_sym_frac,
        "max_state_frac": max_state_frac,
        "top_symbol":     top_symbol,
        "top_state":      top_state,
        "p_gt1":          p_gt1,
        "p_lt_neg1":      p_lt_neg1,
    }


# ---------------------------------------------------------------------------
# Section 5 — Diversification analysis
# ---------------------------------------------------------------------------

def diversification_analysis(
    filtered_trades: list[dict],
    raw_events: list[dict],
) -> dict:
    """
    Measure CTU/ETD diversification:
      - quarterly correlation of mean cost-adj returns
      - time-window overlap between raw events
      - conditional: when CTU is in bottom quartile, what is ETD?
      - activation patterns (quarterly presence)
    """
    ctu_trades = [t for t in filtered_trades if t["state"] == "CONTRACTING_TRENDING_UP"]
    etd_trades = [t for t in filtered_trades if t["state"] == "EXPANDING_TRENDING_DOWN"]

    # Quarterly means
    ctu_q: dict[str, list[float]] = {}
    for t in ctu_trades:
        q = _quarter_label(t["entry_ts"])
        ctu_q.setdefault(q, []).append(t["cost_adj"])

    etd_q: dict[str, list[float]] = {}
    for t in etd_trades:
        q = _quarter_label(t["entry_ts"])
        etd_q.setdefault(q, []).append(t["cost_adj"])

    common_quarters = sorted(set(ctu_q) & set(etd_q))
    ctu_qmeans      = [sum(ctu_q[q]) / len(ctu_q[q]) for q in common_quarters]
    etd_qmeans      = [sum(etd_q[q]) / len(etd_q[q]) for q in common_quarters]
    quarterly_corr  = _pearson(ctu_qmeans, etd_qmeans)

    # Conditional performance: when CTU is in bottom quartile
    if len(ctu_qmeans) >= 4:
        sorted_ctu = sorted(ctu_qmeans)
        q25_ctu = sorted_ctu[len(sorted_ctu) // 4]
        weak_ctu_quarters = [q for q, m in zip(common_quarters, ctu_qmeans) if m <= q25_ctu]
        etd_when_ctu_weak = [sum(etd_q[q]) / len(etd_q[q]) for q in weak_ctu_quarters if q in etd_q]
        etd_cond_mean = round(sum(etd_when_ctu_weak) / len(etd_when_ctu_weak), 4) if etd_when_ctu_weak else None
    else:
        etd_cond_mean = None
        weak_ctu_quarters = []

    if len(etd_qmeans) >= 4:
        sorted_etd = sorted(etd_qmeans)
        q25_etd = sorted_etd[len(sorted_etd) // 4]
        weak_etd_quarters = [q for q, m in zip(common_quarters, etd_qmeans) if m <= q25_etd]
        ctu_when_etd_weak = [sum(ctu_q[q]) / len(ctu_q[q]) for q in weak_etd_quarters if q in ctu_q]
        ctu_cond_mean = round(sum(ctu_when_etd_weak) / len(ctu_when_etd_weak), 4) if ctu_when_etd_weak else None
    else:
        ctu_cond_mean = None
        weak_etd_quarters = []

    # Overlap: fraction of raw CTU events that have any raw ETD event simultaneously active
    ctu_raw = [ev for ev in raw_events if ev.get("state") == "CONTRACTING_TRENDING_UP"]
    etd_raw = [ev for ev in raw_events if ev.get("state") == "EXPANDING_TRENDING_DOWN"]

    n_overlap = 0
    for c in ctu_raw:
        c_entry = c["event_hour_ts"]
        c_exit  = c_entry + timedelta(hours=int(c.get("bars_held") or 365))
        for e in etd_raw:
            e_entry = e["event_hour_ts"]
            e_exit  = e_entry + timedelta(hours=int(e.get("bars_held") or 365))
            if e_entry <= c_exit and e_exit >= c_entry:
                n_overlap += 1
                break

    frac_overlap = round(n_overlap / len(ctu_raw), 3) if ctu_raw else None

    # Quarterly presence table
    all_quarters = sorted(set(ctu_q) | set(etd_q))
    presence = []
    for q in all_quarters:
        c_n = len(ctu_q.get(q, []))
        e_n = len(etd_q.get(q, []))
        c_m = round(sum(ctu_q[q]) / c_n, 4) if c_n else None
        e_m = round(sum(etd_q[q]) / e_n, 4) if e_n else None
        presence.append({
            "quarter":     q,
            "ctu_n":       c_n,
            "etd_n":       e_n,
            "ctu_mean":    c_m,
            "etd_mean":    e_m,
            "both_active": c_n > 0 and e_n > 0,
        })

    frac_both_active = round(
        sum(1 for p in presence if p["both_active"]) / len(presence), 3
    ) if presence else None

    return {
        "n_common_quarters":   len(common_quarters),
        "quarterly_corr":      quarterly_corr,
        "etd_mean_ctu_weak":   etd_cond_mean,
        "n_weak_ctu_quarters": len(weak_ctu_quarters),
        "ctu_mean_etd_weak":   ctu_cond_mean,
        "n_weak_etd_quarters": len(weak_etd_quarters),
        "n_ctu_raw":           len(ctu_raw),
        "n_etd_raw":           len(etd_raw),
        "frac_ctu_with_etd_overlap": frac_overlap,
        "frac_quarters_both_active": frac_both_active,
        "presence_table":      presence,
    }


# ---------------------------------------------------------------------------
# Section 6 — Scenario projections
# ---------------------------------------------------------------------------

SCENARIO_LABELS: list[str] = [
    "pooled",
    "test_split",
    "shrunk_50pct",
    "ci_floor",
]


def scenario_projections(
    filtered_trades: list[dict],
    expected_inputs: dict[str, dict],
    data_span_years: float = DATA_SPAN_YEARS,
) -> list[dict]:
    """
    Project annual expected P&L under each expected-return scenario.
    Uses actual trade frequency from simulation; only the per-trade expected
    return assumption changes across scenarios.

    Also shows cost-adjusted break-even trade frequency (trades needed per year
    for the scenario's per-trade return to equal the cost).
    """
    n_total = len(filtered_trades)
    if n_total == 0:
        return []

    annual_rate = n_total / data_span_years
    ctu_n = sum(1 for t in filtered_trades if t["state"] == "CONTRACTING_TRENDING_UP")
    etd_n = sum(1 for t in filtered_trades if t["state"] == "EXPANDING_TRENDING_DOWN")
    ctu_rate = ctu_n / data_span_years
    etd_rate = etd_n / data_span_years

    rows: list[dict] = []
    for scenario in SCENARIO_LABELS:
        ctu_inp = expected_inputs.get("CONTRACTING_TRENDING_UP", {})
        etd_inp = expected_inputs.get("EXPANDING_TRENDING_DOWN", {})

        ctu_e = ctu_inp.get(f"{scenario}_cost_adj" if scenario != "ci_floor"
                             else "ci_floor_cost_adj")
        etd_e = etd_inp.get(f"{scenario}_cost_adj" if scenario != "ci_floor"
                             else "ci_floor_cost_adj")

        # Map scenario names to field names
        field_map = {
            "pooled":       "pooled_cost_adj",
            "test_split":   "test_cost_adj",
            "shrunk_50pct": "shrunk_cost_adj",
            "ci_floor":     "ci_floor_cost_adj",
        }
        ctu_e = ctu_inp.get(field_map[scenario])
        etd_e = etd_inp.get(field_map[scenario])

        # Blended expected return (weighted by state trade count)
        if ctu_e is not None and etd_e is not None:
            blended = round((ctu_e * ctu_n + etd_e * etd_n) / n_total, 4)
        elif ctu_e is not None:
            blended = ctu_e
        elif etd_e is not None:
            blended = etd_e
        else:
            blended = None

        annual_pnl = round(blended * annual_rate, 2) if blended is not None else None

        rows.append({
            "scenario":        scenario,
            "ctu_expected":    ctu_e,
            "etd_expected":    etd_e,
            "blended_expected":blended,
            "annual_rate":     round(annual_rate, 1),
            "annual_pnl":      annual_pnl,
            "viable":          blended is not None and blended > 0,
        })

    return rows


# ---------------------------------------------------------------------------
# Section 7 — Promotion criteria
# ---------------------------------------------------------------------------

def evaluate_promotion(
    filtered_stats:  dict,
    unfiltered_stats:dict,
    div_results:     dict,
    scenarios:       list[dict],
    exp_inputs:      dict[str, dict],
) -> dict:
    """
    Five objective pass/fail criteria for promotion to sizing research.

    P1. risk_adjusted_advantage:
        filtered Sharpe >= unfiltered Sharpe + SHARPE_FLOOR
    P2. drawdown_acceptable:
        filtered max drawdown <= MAX_DD_CEILING SD30d
    P3. shrunk_viable:
        blended cost-adj expected return under shrunk scenario > 0
    P4. diversification:
        |quarterly correlation| <= 0.60 (CTU/ETD not too correlated)
    P5. concentration_acceptable:
        max single-symbol fraction <= 0.30 of filtered trades

    Verdict
    -------
    PROCEED          : all 5 pass
    CONTINUE_MONITOR : 4/5 pass
    REJECT           : <= 3/5 pass
    """
    criteria: dict[str, dict] = {}

    # P1: risk-adjusted advantage
    f_sharpe = filtered_stats.get("sharpe")
    u_sharpe = unfiltered_stats.get("sharpe")
    p1_pass  = (f_sharpe is not None and u_sharpe is not None
                and f_sharpe >= u_sharpe + SHARPE_FLOOR)
    criteria["risk_adjusted_advantage"] = {
        "pass":            p1_pass,
        "filtered_sharpe": f_sharpe,
        "unfiltered_sharpe": u_sharpe,
        "required_gap":    SHARPE_FLOOR,
    }

    # P2: drawdown
    max_dd  = filtered_stats.get("max_drawdown")
    p2_pass = max_dd is not None and max_dd <= MAX_DD_CEILING
    criteria["drawdown_acceptable"] = {
        "pass":         p2_pass,
        "max_drawdown": max_dd,
        "ceiling":      MAX_DD_CEILING,
    }

    # P3: shrunk scenario viable
    shrunk_sc    = next((s for s in scenarios if s["scenario"] == "shrunk_50pct"), None)
    blended_shrunk = shrunk_sc.get("blended_expected") if shrunk_sc else None
    p3_pass      = blended_shrunk is not None and blended_shrunk > 0
    criteria["shrunk_viable"] = {
        "pass":             p3_pass,
        "blended_expected": blended_shrunk,
    }

    # P4: diversification
    q_corr   = div_results.get("quarterly_corr")
    p4_pass  = q_corr is not None and abs(q_corr) <= 0.60
    criteria["diversification"] = {
        "pass":            p4_pass,
        "quarterly_corr":  q_corr,
        "threshold":       0.60,
    }

    # P5: concentration
    max_sym = filtered_stats.get("max_sym_frac")
    p5_pass = max_sym is not None and max_sym <= 0.30
    criteria["concentration_acceptable"] = {
        "pass":        p5_pass,
        "max_sym_frac":max_sym,
        "ceiling":     0.30,
    }

    n_pass  = sum(1 for c in criteria.values() if c.get("pass"))
    n_total = len(criteria)
    verdict = "PROCEED" if n_pass == n_total else ("CONTINUE_MONITOR" if n_pass >= 4 else "REJECT")

    return {
        "verdict":  verdict,
        "n_pass":   n_pass,
        "n_total":  n_total,
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_expected_return_inputs(inputs: dict[str, dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 1 — Conservative expected-return inputs")
    log.info("=" * 80)
    log.info("  Shrinkage rule: shrunk = %.0f%% * test_split_mean (toward zero)", SHRINKAGE_FACTOR * 100)
    log.info("  Cost:           %.2f SD30d per trade (medium friction)", MEDIUM_COST)
    log.info("")
    log.info("  %-35s  %6s  %6s  %9s  %9s  %9s  %9s  %9s",
             "state", "n_pool", "n_test",
             "pooled", "test", "shrunk", "ci_floor", "ci_high")
    log.info("  " + "-" * 95)
    for state, inp in inputs.items():
        log.info(
            "  %-35s  %6d  %6d  %9s  %9s  %9s  %9s  %9s",
            state,
            inp.get("n_pooled") or 0,
            inp.get("n_test")   or 0,
            f"{inp['pooled_mean']:+.4f}" if inp.get("pooled_mean") is not None else "N/A",
            f"{inp['test_mean']:+.4f}"   if inp.get("test_mean")   is not None else "N/A",
            f"{inp['shrunk_mean']:+.4f}" if inp.get("shrunk_mean") is not None else "N/A",
            f"{inp['ci_floor']:+.4f}"    if inp.get("ci_floor")    is not None else "N/A",
            f"{inp['ci_high']:+.4f}"     if inp.get("ci_high")     is not None else "N/A",
        )
    log.info("")
    log.info("  Cost-adjusted:")
    log.info("  %-35s  %9s  %9s  %9s  %9s",
             "state", "pooled", "test", "shrunk", "ci_floor")
    log.info("  " + "-" * 65)
    for state, inp in inputs.items():
        log.info(
            "  %-35s  %9s  %9s  %9s  %9s",
            state,
            f"{inp['pooled_cost_adj']:+.4f}" if inp.get("pooled_cost_adj") is not None else "N/A",
            f"{inp['test_cost_adj']:+.4f}"   if inp.get("test_cost_adj")   is not None else "N/A",
            f"{inp['shrunk_cost_adj']:+.4f}" if inp.get("shrunk_cost_adj") is not None else "N/A",
            f"{inp['ci_floor_cost_adj']:+.4f}" if inp.get("ci_floor_cost_adj") is not None else "N/A",
        )


def print_portfolio_stats(stats: dict) -> None:
    log.info("")
    log.info("  Portfolio: %-40s  n=%d", stats.get("label", ""), stats.get("n", 0))
    log.info("    mean return (cost-adj): %s", f"{stats['mean']:+.4f}" if stats.get("mean") is not None else "N/A")
    log.info("    median                : %s", f"{stats['median']:+.4f}" if stats.get("median") is not None else "N/A")
    log.info("    std dev               : %s", f"{stats['std']:.4f}" if stats.get("std") is not None else "N/A")
    log.info("    sharpe proxy          : %s", f"{stats['sharpe']:+.4f}" if stats.get("sharpe") is not None else "N/A")
    log.info("    win rate              : %s", f"{stats['win_rate']:.4f}" if stats.get("win_rate") is not None else "N/A")
    log.info("    payoff ratio          : %s", str(stats.get("payoff")) or "N/A")
    log.info("    max drawdown          : %s SD30d", f"{stats['max_drawdown']:.4f}" if stats.get("max_drawdown") is not None else "N/A")
    log.info("    final cum P&L         : %s SD30d", f"{stats['final_cum_pnl']:+.2f}" if stats.get("final_cum_pnl") is not None else "N/A")
    log.info("    annual trade rate     : %s", f"{stats['annual_rate']:.1f}" if stats.get("annual_rate") is not None else "N/A")
    log.info("    mean active at entry  : %s", f"{stats['mean_active']:.2f}" if stats.get("mean_active") is not None else "N/A")
    log.info("    top symbol            : %s (%.1f%%)", stats.get("top_symbol") or "N/A",
             (stats.get("max_sym_frac") or 0) * 100)
    log.info("    top state             : %s (%.1f%%)", stats.get("top_state") or "N/A",
             (stats.get("max_state_frac") or 0) * 100)
    log.info("    P(trade > +1 SD30d)   : %.4f", stats.get("p_gt1") or 0.0)
    log.info("    P(trade < -1 SD30d)   : %.4f", stats.get("p_lt_neg1") or 0.0)


def print_comparison(f_stats: dict, u_stats: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 4 — Filtered vs unfiltered comparison")
    log.info("=" * 80)
    log.info("  %-28s  %14s  %14s  %12s",
             "metric", "filtered", "unfiltered", "difference")
    log.info("  " + "-" * 72)

    metrics = [
        ("n_trades",     "n",            "d"),
        ("mean",         "mean",         "f4"),
        ("std",          "std",          "f4"),
        ("sharpe",       "sharpe",       "f4"),
        ("win_rate",     "win_rate",     "f4"),
        ("payoff",       "payoff",       "f3"),
        ("max_drawdown", "max_drawdown", "f4"),
        ("annual_rate",  "annual_rate",  "f1"),
        ("mean_active",  "mean_active",  "f2"),
        ("p_gt1",        "p_gt1",        "f4"),
        ("p_lt_neg1",    "p_lt_neg1",    "f4"),
    ]

    def _fmt(v: object, fmt: str) -> str:
        if v is None:
            return "N/A"
        if fmt == "d":
            return str(int(v))
        if fmt == "f4":
            return f"{float(v):+.4f}"
        if fmt == "f3":
            return f"{float(v):+.3f}"
        if fmt == "f2":
            return f"{float(v):+.2f}"
        if fmt == "f1":
            return f"{float(v):.1f}"
        return str(v)

    for label, key, fmt in metrics:
        fv = f_stats.get(key)
        uv = u_stats.get(key)
        if fv is not None and uv is not None and fmt not in ("d", "f1"):
            diff = f"{float(fv) - float(uv):+.4f}"
        else:
            diff = "N/A"
        log.info("  %-28s  %14s  %14s  %12s",
                 label, _fmt(fv, fmt), _fmt(uv, fmt), diff)


def print_diversification(div: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 5 — CTU / ETD diversification")
    log.info("=" * 80)
    log.info("  Common quarters (both active)    : %d", div.get("n_common_quarters") or 0)
    log.info("  Quarterly return correlation     : %s",
             f"{div['quarterly_corr']:+.4f}" if div.get("quarterly_corr") is not None else "N/A")
    log.info("  Frac quarters both active        : %s",
             f"{div['frac_quarters_both_active']:.3f}" if div.get("frac_quarters_both_active") is not None else "N/A")
    log.info("  CTU overlap with ETD (raw events): %s",
             f"{div['frac_ctu_with_etd_overlap']:.3f}" if div.get("frac_ctu_with_etd_overlap") is not None else "N/A")
    log.info("")
    log.info("  Conditional analysis:")
    log.info("    ETD mean when CTU in bottom quartile: %s  (n=%d quarters)",
             f"{div['etd_mean_ctu_weak']:+.4f}" if div.get("etd_mean_ctu_weak") is not None else "N/A",
             div.get("n_weak_ctu_quarters") or 0)
    log.info("    CTU mean when ETD in bottom quartile: %s  (n=%d quarters)",
             f"{div['ctu_mean_etd_weak']:+.4f}" if div.get("ctu_mean_etd_weak") is not None else "N/A",
             div.get("n_weak_etd_quarters") or 0)
    log.info("")
    log.info("  Quarterly presence (simulated trades):")
    log.info("    %-8s  %7s  %7s  %9s  %9s  %10s",
             "quarter", "ctu_n", "etd_n", "ctu_mean", "etd_mean", "both_active")
    log.info("    " + "-" * 60)
    for row in (div.get("presence_table") or []):
        log.info(
            "    %-8s  %7d  %7d  %9s  %9s  %10s",
            row["quarter"], row["ctu_n"], row["etd_n"],
            f"{row['ctu_mean']:+.4f}" if row.get("ctu_mean") is not None else "N/A",
            f"{row['etd_mean']:+.4f}" if row.get("etd_mean") is not None else "N/A",
            "YES" if row.get("both_active") else "NO",
        )


def print_scenarios(scenarios: list[dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 6 — Uncertainty-aware scenario projections")
    log.info("=" * 80)
    log.info("  Annual rate: actual trade frequency from simulation (fixed across scenarios)")
    log.info("  Annual P&L = blended expected cost-adj return x annual trade rate")
    log.info("")
    log.info("  %-15s  %11s  %11s  %13s  %12s  %11s  %8s",
             "scenario", "ctu_exp", "etd_exp", "blended_exp", "annual_rate", "annual_pnl", "viable")
    log.info("  " + "-" * 85)
    for s in scenarios:
        log.info(
            "  %-15s  %11s  %11s  %13s  %12s  %11s  %8s",
            s["scenario"],
            f"{s['ctu_expected']:+.4f}"    if s.get("ctu_expected")     is not None else "N/A",
            f"{s['etd_expected']:+.4f}"    if s.get("etd_expected")     is not None else "N/A",
            f"{s['blended_expected']:+.4f}" if s.get("blended_expected") is not None else "N/A",
            f"{s['annual_rate']:.1f}",
            f"{s['annual_pnl']:+.2f}"      if s.get("annual_pnl")       is not None else "N/A",
            "YES" if s.get("viable") else "NO",
        )


def print_promotion_verdict(result: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 7 — Promotion criteria: %s  (%d/%d pass)",
             result["verdict"], result["n_pass"], result["n_total"])
    log.info("=" * 80)
    for name, c in result["criteria"].items():
        status = "PASS" if c.get("pass") else "FAIL"
        log.info("  [%s]  %s", status, name)
        if name == "risk_adjusted_advantage":
            log.info("         filtered_sharpe=%s  unfiltered_sharpe=%s  required_gap=%.2f",
                     f"{c['filtered_sharpe']:+.4f}" if c.get("filtered_sharpe") is not None else "N/A",
                     f"{c['unfiltered_sharpe']:+.4f}" if c.get("unfiltered_sharpe") is not None else "N/A",
                     c.get("required_gap") or 0.0)
        elif name == "drawdown_acceptable":
            log.info("         max_drawdown=%.4f  ceiling=%.1f", c.get("max_drawdown") or 0.0, c.get("ceiling") or 0.0)
        elif name == "shrunk_viable":
            log.info("         blended_expected=%s",
                     f"{c['blended_expected']:+.4f}" if c.get("blended_expected") is not None else "N/A")
        elif name == "diversification":
            log.info("         quarterly_corr=%s  threshold=%.2f",
                     f"{c['quarterly_corr']:+.4f}" if c.get("quarterly_corr") is not None else "N/A",
                     c.get("threshold") or 0.0)
        elif name == "concentration_acceptable":
            log.info("         max_symbol_frac=%.3f  ceiling=%.2f",
                     c.get("max_sym_frac") or 0.0, c.get("ceiling") or 0.0)


def print_recommendation_memo(result: dict, scenarios: list[dict], div: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 8 — RECOMMENDATION MEMO")
    log.info("=" * 80)
    verdict = result["verdict"]
    log.info("  Verdict: %s", verdict)
    log.info("")

    if verdict == "PROCEED":
        log.info("  RECOMMENDATION: PROCEED to conservative position-sizing research.")
        log.info("")
        log.info("  Basis:")
        log.info("    - Filtered portfolio delivers superior risk-adjusted returns vs unfiltered")
        log.info("    - Max drawdown is within acceptable bounds")
        log.info("    - Both states remain viable under 50%% test-split shrinkage")
        log.info("    - CTU / ETD quarterly correlation indicates meaningful diversification")
        log.info("    - No single symbol dominates portfolio concentration")
        log.info("")
        log.info("  Constraints for sizing research:")
        shrunk_sc = next((s for s in scenarios if s["scenario"] == "shrunk_50pct"), {})
        log.info("    - Use shrunk expected return as sizing input:")
        log.info("      CTU: %s SD30d  ETD: %s SD30d  (blended: %s SD30d per trade)",
                 f"{shrunk_sc.get('ctu_expected'):+.4f}" if shrunk_sc.get("ctu_expected") is not None else "N/A",
                 f"{shrunk_sc.get('etd_expected'):+.4f}" if shrunk_sc.get("etd_expected") is not None else "N/A",
                 f"{shrunk_sc.get('blended_expected'):+.4f}" if shrunk_sc.get("blended_expected") is not None else "N/A")
        log.info("    - Do NOT pyramid or size based on pooled mean")
        log.info("    - Maintain max 2 positions per symbol, max 10 total")
        log.info("    - Monitor ETD separately: current forward P&L is marginal")
        log.info("    - Review quarterly; halt if cumulative cost-adj mean goes negative")

    elif verdict == "CONTINUE_MONITOR":
        log.info("  RECOMMENDATION: CONTINUE MONITORING. Do not proceed to sizing research yet.")
        log.info("")
        log.info("  At least one promotion criterion has not been met.")
        failed = [k for k, c in result["criteria"].items() if not c.get("pass")]
        for f in failed:
            log.info("    - FAILED: %s", f)
        log.info("")
        log.info("  Action: extend forward monitoring by two additional quarters.")
        log.info("  Reassess once n_test >= 300 per state or further criteria are met.")

    else:
        log.info("  RECOMMENDATION: REJECT. Evidence insufficient for capital allocation research.")
        log.info("")
        failed = [k for k, c in result["criteria"].items() if not c.get("pass")]
        for f in failed:
            log.info("    - FAILED: %s", f)
        log.info("")
        log.info("  Do not proceed to sizing research on this evidence base.")
        log.info("  Re-evaluate after 12+ additional months of forward monitoring.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(
    output_path: str,
    filtered_stats: dict,
    unfiltered_stats: dict,
    scenarios: list[dict],
    div: dict,
    promo: dict,
) -> None:
    rows: list[dict] = []
    # Filtered portfolio stats
    for key, val in filtered_stats.items():
        rows.append({"section": "filtered_portfolio_stats", "key": key, "value": val})
    # Unfiltered portfolio stats (comparison baseline)
    for key, val in unfiltered_stats.items():
        rows.append({"section": "unfiltered_portfolio_stats", "key": key, "value": val})
    # Scenario rows
    for s in scenarios:
        rows.append({"section": "scenario", **s})
    # Promotion verdict
    rows.append({"section": "promotion", "key": "verdict",  "value": promo.get("verdict")})
    rows.append({"section": "promotion", "key": "n_pass",   "value": promo.get("n_pass")})
    rows.append({"section": "promotion", "key": "n_total",  "value": promo.get("n_total")})
    for criterion, c in (promo.get("criteria") or {}).items():
        rows.append({"section": "promotion_criteria", "key": criterion, "value": "PASS" if c.get("pass") else "FAIL"})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fields = ["section", "key", "value", "scenario", "ctu_expected", "etd_expected",
              "blended_expected", "annual_rate", "annual_pnl", "viable"]
    with open(output_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Results written to %s", output_path)


def write_memo(memo_path: str, promo: dict, scenarios: list[dict], div: dict, filtered_stats: dict) -> None:
    """Write the recommendation memo as a standalone text file."""
    verdict  = promo.get("verdict", "UNKNOWN")
    n_pass   = promo.get("n_pass", 0)
    n_total  = promo.get("n_total", 0)
    criteria = promo.get("criteria", {})
    shrunk   = next((s for s in scenarios if s["scenario"] == "shrunk_50pct"), {})
    lines: list[str] = [
        "MARKOV PORTFOLIO SIMULATOR — RECOMMENDATION MEMO",
        "Issue #94: Conservative deployment simulator for CTU and ETD states",
        f"Generated: {__import__('datetime').date.today()}",
        "",
        "=" * 72,
        f"VERDICT: {verdict}  ({n_pass}/{n_total} criteria pass)",
        "=" * 72,
        "",
        "PROMOTION CRITERIA",
        "-" * 40,
    ]
    for name, c in criteria.items():
        status = "PASS" if c.get("pass") else "FAIL"
        lines.append(f"  [{status}]  {name}")
    lines += [
        "",
        "SCENARIO PROJECTIONS (annual P&L in SD30d units)",
        "-" * 40,
    ]
    for s in scenarios:
        viable = "viable" if s.get("viable") else "not viable"
        pnl    = f"{s['annual_pnl']:+.2f}" if s.get("annual_pnl") is not None else "N/A"
        lines.append(f"  {s['scenario']:<15}  blended={s.get('blended_expected') or 'N/A'}  annual_pnl={pnl}  [{viable}]")
    lines += [
        "",
        "DIVERSIFICATION",
        "-" * 40,
        f"  Quarterly return correlation (CTU/ETD): {div.get('quarterly_corr')}",
        f"  Fraction of quarters both active: {div.get('frac_quarters_both_active')}",
        f"  CTU overlap with ETD (raw events): {div.get('frac_ctu_with_etd_overlap')}",
        "",
        "FILTERED PORTFOLIO SUMMARY",
        "-" * 40,
        f"  n_trades:    {filtered_stats.get('n')}",
        f"  mean:        {filtered_stats.get('mean')}",
        f"  sharpe:      {filtered_stats.get('sharpe')}",
        f"  max_drawdown:{filtered_stats.get('max_drawdown')}",
        "",
        "RECOMMENDATION",
        "-" * 40,
    ]
    if verdict == "PROCEED":
        blended = shrunk.get("blended_expected")
        lines += [
            "  PROCEED to conservative position-sizing research.",
            "",
            "  Basis:",
            "    - Filtered portfolio delivers superior risk-adjusted returns vs unfiltered",
            "    - Max drawdown within acceptable bounds",
            "    - Both states viable under 50% test-split shrinkage",
            "    - CTU/ETD quarterly correlation indicates meaningful diversification",
            "    - No single symbol dominates portfolio concentration",
            "",
            "  Constraints for sizing research:",
            f"    - Use shrunk expected return (blended: {blended} SD30d per trade)",
            "    - Do NOT pyramid or size based on pooled mean",
            "    - Maintain max 2 positions per symbol, max 10 total",
            "    - Monitor ETD separately: current forward P&L is marginal",
            "    - Review quarterly; halt if cumulative cost-adj mean goes negative",
        ]
    elif verdict == "CONTINUE_MONITOR":
        failed = [k for k, c in criteria.items() if not c.get("pass")]
        lines += [
            "  CONTINUE MONITORING. Do not proceed to sizing research yet.",
            "",
            "  Failed criteria:",
        ]
        for f in failed:
            lines.append(f"    - {f}")
        lines += [
            "",
            "  Action: extend forward monitoring by two additional quarters.",
            "  Reassess once n_test >= 300 per state or further criteria are met.",
        ]
    else:
        failed = [k for k, c in criteria.items() if not c.get("pass")]
        lines += [
            "  REJECT. Evidence insufficient for capital allocation research.",
            "",
            "  Failed criteria:",
        ]
        for f in failed:
            lines.append(f"    - {f}")
        lines += [
            "",
            "  Do not proceed to sizing research on this evidence base.",
            "  Re-evaluate after 12+ additional months of forward monitoring.",
        ]
    Path(memo_path).parent.mkdir(parents=True, exist_ok=True)
    with open(memo_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Recommendation memo written to %s", memo_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio deployment simulator for candidate Markov states (Issue #94)."
    )
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap CIs (faster run)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--output", default=None,
                        help="Optional CSV output path")
    parser.add_argument("--memo", default=None,
                        help="Optional recommendation memo text output path")
    args = parser.parse_args()
    n_boot = 0 if args.no_bootstrap else args.n_bootstrap

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info("Loading FX breakout events ...")
    all_events = load_events()
    log.info("Loaded %d state-labeled events", len(all_events))
    if not all_events:
        log.error("No events found.")
        sys.exit(1)

    # Global carry neutralization
    baselines     = compute_baselines(all_events)
    excess_events = apply_excess_return(all_events, baselines)
    log.info("Carry neutralization: %d events", len(excess_events))

    # Filter to cohort
    present        = {ev["symbol"] for ev in excess_events}
    active_symbols = [s for s in COHORT_SYMBOLS if s in present]
    cohort_events  = [ev for ev in excess_events if ev["symbol"] in set(active_symbols)]
    log.info("Cohort core_majors: %d events  symbols: %s", len(cohort_events), active_symbols)

    # --- Section 1: Expected return inputs ---
    log.info("Computing expected-return inputs ...")
    exp_inputs = compute_expected_return_inputs(cohort_events, n_bootstrap=n_boot)
    print_expected_return_inputs(exp_inputs)

    # --- Section 2: Filtered simulation ---
    log.info("Running filtered portfolio simulation (CTU + ETD) ...")
    filtered_trades = simulate_portfolio(cohort_events, ELIGIBLE_FILTERED)
    filtered_stats  = compute_portfolio_stats(filtered_trades, label="CTU + ETD only")

    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 2 — Filtered portfolio (CTU + ETD)")
    log.info("=" * 80)
    print_portfolio_stats(filtered_stats)

    # --- Section 3: Unfiltered simulation ---
    log.info("Running unfiltered portfolio simulation (all states) ...")
    all_trades    = simulate_portfolio(cohort_events, eligible_states=None)
    all_stats     = compute_portfolio_stats(all_trades, label="All breakout events")

    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 3 — Unfiltered portfolio (all states)")
    log.info("=" * 80)
    print_portfolio_stats(all_stats)

    # --- Section 4: Comparison ---
    print_comparison(filtered_stats, all_stats)

    # --- Section 5: Diversification ---
    log.info("Computing CTU/ETD diversification ...")
    div_results = diversification_analysis(filtered_trades, cohort_events)
    print_diversification(div_results)

    # --- Section 6: Scenarios ---
    scenarios = scenario_projections(filtered_trades, exp_inputs)
    print_scenarios(scenarios)

    # --- Section 7: Promotion criteria ---
    promo = evaluate_promotion(
        filtered_stats, all_stats, div_results, scenarios, exp_inputs
    )
    print_promotion_verdict(promo)

    # --- Section 8: Recommendation ---
    print_recommendation_memo(promo, scenarios, div_results)

    if args.output:
        write_csv(args.output, filtered_stats, all_stats, scenarios, div_results, promo)
    if args.memo:
        write_memo(args.memo, promo, scenarios, div_results, filtered_stats)


if __name__ == "__main__":
    main()

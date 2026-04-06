"""
scripts/markov_pre_deployment.py

Pre-deployment validation of candidate Markov alpha states -- Issue #93.

Seven-section analysis pipeline:
  1. Threshold stability    -- re-test under 3 nearby bucket definitions
  2. Continuous model       -- OLS on (vol_ratio, efficiency, direction, interaction)
  3. Friction stress test   -- 3 cost levels applied to candidate-state returns
  4. Tail & lifecycle       -- return tails, excursion profile, winner/loser decomposition
  5. Probability calibration -- empirical P at multiple return thresholds with CIs
  6. Simplicity sanity      -- candidate edge vs simpler directional/regime rules
  7. Pre-sizing criteria     -- objective go / continue-monitoring / reject verdict

Frozen candidate states (no additions in this issue):
  CONTRACTING_TRENDING_UP
  EXPANDING_TRENDING_DOWN

Cohort: core_majors (7 G10 FX pairs, EUR/USD, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP)
All returns in SD30d units.  Excess return = realized - symbol+direction unconditional mean.

Usage
-----
    python scripts/markov_pre_deployment.py
    python scripts/markov_pre_deployment.py --no-bootstrap
    python scripts/markov_pre_deployment.py --output results/pre_deployment.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import sys
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

CANDIDATE_STATES: list[str] = [
    "CONTRACTING_TRENDING_UP",
    "EXPANDING_TRENDING_DOWN",
]

COHORT_SYMBOLS: list[str] = [
    "EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
]

# Three threshold sets for stability analysis
# Each set: vol split points (lo, hi) + efficiency split point (hi = 'trending' threshold)
THRESHOLD_SETS: dict[str, dict[str, float]] = {
    "baseline": {"vol_lo": 0.80, "vol_hi": 1.20, "eff_hi": 0.60},
    "narrow":   {"vol_lo": 0.85, "vol_hi": 1.15, "eff_hi": 0.65},
    "wide":     {"vol_lo": 0.75, "vol_hi": 1.25, "eff_hi": 0.55},
}

# Friction cost levels (SD30d per trade, round-trip inclusive)
COST_LEVELS: dict[str, float] = {
    "low":    0.03,   # tight G10 spread + minimal slippage
    "medium": 0.07,   # realistic with execution impact
    "high":   0.12,   # stressed spread + adverse fill
}
# Additional cost for trades that cross a weekend (proxy: event starts Thu/Fri)
WEEKEND_EXTRA_COST: float = 0.03

# Bootstrap config
N_BOOTSTRAP: int   = 1000
BLOCK_SIZE:  int   = 20
RANDOM_SEED: int   = 42

# Pass/fail thresholds for pre-sizing criteria
MIN_COUNT:              int   = 20
COST_PASS_LEVEL:        str   = "medium"   # must survive this cost level
THRESHOLD_PASS_FRAC:    float = 0.67       # 2/3 threshold sets must show positive excess
BOOTSTRAP_FRAC_THRESH:  float = 0.70


# ---------------------------------------------------------------------------
# State bucketing (baseline thresholds)
# ---------------------------------------------------------------------------

def _vol_bucket(vol_ratio: float, vol_lo: float = 0.80, vol_hi: float = 1.20) -> str:
    if vol_ratio < vol_lo:
        return "CONTRACTING"
    if vol_ratio <= vol_hi:
        return "NEUTRAL"
    return "EXPANDING"


def _eff_bucket(efficiency: float, eff_lo: float = 0.30, eff_hi: float = 0.60) -> str:
    if efficiency < eff_lo:
        return "CHOPPY"
    if efficiency <= eff_hi:
        return "MIXED"
    return "TRENDING"


def _median(values: list[float]) -> float:
    sv = sorted(values)
    n  = len(sv)
    m  = n // 2
    return sv[m] if n % 2 else (sv[m - 1] + sv[m]) / 2.0


# Candidate logical qualifiers (re-evaluated per threshold set)
def _qualifies_ctu(ev: dict, th: dict[str, float]) -> bool:
    return (float(ev["vol_ratio_20_100"]) < th["vol_lo"]
            and float(ev["efficiency_20"]) > th["eff_hi"]
            and ev["breakout_direction"] == "UP")


def _qualifies_etd(ev: dict, th: dict[str, float]) -> bool:
    return (float(ev["vol_ratio_20_100"]) > th["vol_hi"]
            and float(ev["efficiency_20"]) > th["eff_hi"]
            and ev["breakout_direction"] == "DOWN")


CANDIDATE_QUALIFIERS: dict[str, object] = {
    "CONTRACTING_TRENDING_UP":  _qualifies_ctu,
    "EXPANDING_TRENDING_DOWN":  _qualifies_etd,
}


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
    be.max_favorable_excursion_sd30d,
    be.max_adverse_excursion_sd30d,
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
        ev["state"] = (f"{_vol_bucket(ev['vol_ratio_20_100'])}"
                       f"_{_eff_bucket(ev['efficiency_20'])}"
                       f"_{d}")
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
    out, skipped = [], 0
    for ev in events:
        b = baselines.get((ev["symbol"], ev["breakout_direction"]))
        if b is None:
            skipped += 1
            continue
        new_ev = dict(ev)
        new_ev["excess_return"] = float(ev["realized_return_sd30d"]) - b
        out.append(new_ev)
    if skipped:
        log.warning("Dropped %d events with no baseline key.", skipped)
    return out


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _state_returns(
    events: list[dict],
    state_field: str,
    state_val: str,
    return_field: str = "excess_return",
) -> list[float]:
    return [
        float(ev[return_field])
        for ev in events
        if ev.get(state_field) == state_val and ev.get(return_field) is not None
    ]


def metrics(returns: list[float]) -> dict:
    n = len(returns)
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "win_rate": None, "payoff": None}
    wins    = [r for r in returns if r > 0]
    losses  = [r for r in returns if r <= 0]
    avg_win = sum(wins)   / len(wins)   if wins   else None
    avg_los = abs(sum(losses) / len(losses)) if losses else None
    return {
        "n":        n,
        "mean":     round(sum(returns) / n, 4),
        "median":   round(_median(returns), 4),
        "win_rate": round(len(wins) / n, 4),
        "payoff":   round(avg_win / avg_los, 3) if (avg_win and avg_los) else None,
    }


def cohort_baseline_mean(events: list[dict], return_field: str = "excess_return") -> float:
    vals = [float(ev[return_field]) for ev in events if ev.get(return_field) is not None]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    returns: list[float],
    block_size: int = BLOCK_SIZE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    n = len(returns)
    if n == 0 or n_bootstrap == 0:
        return {"ci_low": None, "ci_high": None, "frac_pos": None,
                "mean_obs": round(sum(returns)/n, 4) if n else None}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            s = rng.randint(0, n - 1)
            for k in range(block_size):
                sample.append(returns[(s + k) % n])
        sample = sample[:n]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[max(0, int(round(0.025 * n_bootstrap)))]
    hi = means[min(n_bootstrap - 1, int(round(0.975 * n_bootstrap)) - 1)]
    return {
        "ci_low":    round(lo, 4),
        "ci_high":   round(hi, 4),
        "frac_pos":  round(sum(1 for m in means if m > 0) / n_bootstrap, 3),
        "mean_obs":  round(sum(returns) / n, 4),
    }


# ---------------------------------------------------------------------------
# Section 1 — Threshold stability
# ---------------------------------------------------------------------------

def run_threshold_stability(
    events: list[dict],
    n_bootstrap: int = N_BOOTSTRAP,
    min_count: int = MIN_COUNT,
) -> dict[str, dict[str, dict]]:
    """
    Re-qualify candidate states under 3 threshold sets.
    Returns {state: {threshold_set: {n, mean, median, win_rate, payoff, ci_low, ci_high, frac_pos}}}
    """
    base_mean = cohort_baseline_mean(events)
    results: dict[str, dict[str, dict]] = {s: {} for s in CANDIDATE_STATES}

    for tset_name, th in THRESHOLD_SETS.items():
        for state in CANDIDATE_STATES:
            qualifier = CANDIDATE_QUALIFIERS[state]
            matched   = [ev for ev in events if qualifier(ev, th)]
            rets      = [float(ev["excess_return"]) for ev in matched
                         if ev.get("excess_return") is not None]
            m         = metrics(rets)
            boot      = block_bootstrap_ci(rets, n_bootstrap=n_bootstrap)
            results[state][tset_name] = {
                "n":         m["n"],
                "mean":      m["mean"],
                "median":    m["median"],
                "win_rate":  m["win_rate"],
                "payoff":    m["payoff"],
                "lift":      round(m["mean"] - base_mean, 4) if m["mean"] is not None else None,
                "ci_low":    boot["ci_low"],
                "ci_high":   boot["ci_high"],
                "frac_pos":  boot["frac_pos"],
                "positive":  (m["mean"] is not None and m["mean"] > 0 and m["n"] >= min_count),
            }
    return results


# ---------------------------------------------------------------------------
# Section 2 — Continuous model (OLS)
# ---------------------------------------------------------------------------

def _ols_solve(XTX: list[list[float]], XTy: list[float], ridge: float = 1e-4) -> Optional[list[float]]:
    """Solve XTX @ beta = XTy via Gaussian elimination (forward + back substitution)."""
    k = len(XTy)
    M = [[XTX[i][j] + (ridge if i == j else 0.0) for j in range(k)] + [XTy[i]]
         for i in range(k)]
    for col in range(k):
        prow = max(range(col, k), key=lambda r: abs(M[r][col]))
        M[col], M[prow] = M[prow], M[col]
        if abs(M[col][col]) < 1e-10:
            return None
        inv_p = 1.0 / M[col][col]
        for row in range(col + 1, k):
            f = M[row][col] * inv_p
            M[row] = [M[row][j] - f * M[col][j] for j in range(k + 1)]
    beta = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = M[i][k]
        for j in range(i + 1, k):
            s -= M[i][j] * beta[j]
        beta[i] = s / M[i][i]
    return beta


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num  = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx   = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy   = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return round(num / (dx * dy), 4) if dx > 1e-12 and dy > 1e-12 else 0.0


def run_continuous_model(events: list[dict]) -> dict:
    """
    OLS on excess_return ~ intercept + vol_ratio_c + efficiency_c + dir_bin + vol*eff_c.
    Reports coefficients, R^2, and predicted excess return at each candidate centroid.
    Also reports Pearson correlations of each feature with excess_return.
    """
    # Compute centering statistics over full cohort
    vols  = [ev["vol_ratio_20_100"] for ev in events]
    effs  = [ev["efficiency_20"]    for ev in events]
    mean_v = sum(vols) / len(vols)
    mean_e = sum(effs) / len(effs)

    # Build XTX, XTy incrementally (O(n*k^2), k=5)
    k     = 5    # intercept, vol_c, eff_c, dir_bin, vol_c*eff_c
    XTX   = [[0.0] * k for _ in range(k)]
    XTy   = [0.0] * k
    SS_tot = 0.0
    y_vals: list[float] = []

    raw_vols, raw_effs, raw_ints, raw_dirs, raw_y = [], [], [], [], []

    for ev in events:
        y = ev.get("excess_return")
        if y is None:
            continue
        y = float(y)
        vc  = ev["vol_ratio_20_100"] - mean_v
        ec  = ev["efficiency_20"]    - mean_e
        db  = 1.0 if ev["breakout_direction"] == "UP" else 0.0
        inter = vc * ec
        x = [1.0, vc, ec, db, inter]
        for i in range(k):
            for j in range(k):
                XTX[i][j] += x[i] * x[j]
            XTy[i] += x[i] * y
        y_vals.append(y)
        raw_vols.append(vc)
        raw_effs.append(ec)
        raw_ints.append(inter)
        raw_dirs.append(db)
        raw_y.append(y)

    n_total = len(y_vals)
    mean_y  = sum(y_vals) / n_total if n_total else 0.0
    SS_tot  = sum((yi - mean_y) ** 2 for yi in y_vals)

    beta = _ols_solve(XTX, XTy)
    r_sq: Optional[float] = None
    if beta is not None and SS_tot > 1e-12:
        SS_res = 0.0
        for i, ev in enumerate(events):
            y = ev.get("excess_return")
            if y is None:
                continue
            y = float(y)
            vc = ev["vol_ratio_20_100"] - mean_v
            ec = ev["efficiency_20"]    - mean_e
            db = 1.0 if ev["breakout_direction"] == "UP" else 0.0
            x  = [1.0, vc, ec, db, vc * ec]
            yhat = sum(b * xi for b, xi in zip(beta, x))
            SS_res += (y - yhat) ** 2
        r_sq = round(1.0 - SS_res / SS_tot, 5)

    # Pearson correlations
    corr_vol     = _pearson(raw_vols, raw_y)
    corr_eff     = _pearson(raw_effs, raw_y)
    corr_int     = _pearson(raw_ints, raw_y)
    corr_dir     = _pearson(raw_dirs, raw_y)

    # Predictions at candidate centroids
    predictions: dict[str, dict] = {}
    for state in CANDIDATE_STATES:
        state_events = [ev for ev in events if ev.get("state") == state]
        if not state_events:
            predictions[state] = {"n": 0, "predicted": None, "agrees": None}
            continue
        vc_c = sum(ev["vol_ratio_20_100"] for ev in state_events) / len(state_events) - mean_v
        ec_c = sum(ev["efficiency_20"]    for ev in state_events) / len(state_events) - mean_e
        db_c = sum(1.0 if ev["breakout_direction"] == "UP" else 0.0
                   for ev in state_events) / len(state_events)
        x_c  = [1.0, vc_c, ec_c, db_c, vc_c * ec_c]
        pred = round(sum(b * xi for b, xi in zip(beta, x_c)), 4) if beta else None
        obs  = round(sum(float(ev["excess_return"]) for ev in state_events
                         if ev.get("excess_return") is not None) / len(state_events), 4)
        predictions[state] = {
            "n":         len(state_events),
            "predicted": pred,
            "observed":  obs,
            "agrees":    (pred is not None and pred > 0 and obs > 0),
        }

    return {
        "n_events":     n_total,
        "mean_v":       round(mean_v, 4),
        "mean_e":       round(mean_e, 4),
        "beta":         [round(b, 5) for b in beta] if beta else None,
        "feature_names":["intercept", "vol_ratio_c", "efficiency_c", "dir_bin", "vol_c*eff_c"],
        "r_squared":    r_sq,
        "corr_vol":     corr_vol,
        "corr_eff":     corr_eff,
        "corr_interaction": corr_int,
        "corr_direction":   corr_dir,
        "predictions":  predictions,
    }


# ---------------------------------------------------------------------------
# Section 3 — Friction stress test
# ---------------------------------------------------------------------------

def _is_weekend_crossing(ev: dict) -> bool:
    """True if the trade likely crosses a weekend: starts Thu/Fri and holds >= 24h."""
    ts   = ev.get("event_hour_ts")
    bars = ev.get("bars_held") or 0
    if ts is None or bars < 24:
        return False
    if hasattr(ts, "weekday"):
        return ts.weekday() >= 3   # Thu=3, Fri=4
    return False


def run_friction_stress(
    events: list[dict],
    n_bootstrap: int = N_BOOTSTRAP,
    min_count: int = MIN_COUNT,
) -> dict[str, dict[str, dict]]:
    """
    Apply flat cost haircuts to candidate-state excess returns at 3 levels.
    Returns {state: {cost_level: {n, mean_raw, mean_adj, win_rate, payoff, ci_low, ci_high}}}
    """
    results: dict[str, dict[str, dict]] = {s: {} for s in CANDIDATE_STATES}

    for state in CANDIDATE_STATES:
        state_events = [ev for ev in events if ev.get("state") == state]

        for level_name, base_cost in COST_LEVELS.items():
            adj_returns: list[float] = []
            raw_returns: list[float] = []
            for ev in state_events:
                exc = ev.get("excess_return")
                if exc is None:
                    continue
                exc = float(exc)
                cost = base_cost
                if _is_weekend_crossing(ev):
                    cost += WEEKEND_EXTRA_COST
                adj_returns.append(exc - cost)
                raw_returns.append(exc)

            m_raw  = metrics(raw_returns)
            m_adj  = metrics(adj_returns)
            boot   = block_bootstrap_ci(adj_returns, n_bootstrap=n_bootstrap)

            results[state][level_name] = {
                "n":            len(adj_returns),
                "base_cost":    base_cost,
                "mean_raw":     m_raw["mean"],
                "mean_adj":     m_adj["mean"],
                "win_rate_adj": m_adj["win_rate"],
                "payoff_adj":   m_adj["payoff"],
                "ci_low":       boot["ci_low"],
                "ci_high":      boot["ci_high"],
                "frac_pos":     boot["frac_pos"],
                "survives":     (m_adj["mean"] is not None and m_adj["mean"] > 0),
            }

    return results


# ---------------------------------------------------------------------------
# Section 4 — Tail and lifecycle decomposition
# ---------------------------------------------------------------------------

def run_tail_lifecycle(events: list[dict]) -> dict[str, dict]:
    """
    For each candidate state, compute:
    - return tail proportions (> 0, > +1, > +2, > +4 SD30d; < -1, < -2)
    - max_favorable / max_adverse excursion means
    - bars held by winners vs losers
    - weekend-crossing frequency and mean return difference
    """
    results: dict[str, dict] = {}

    for state in CANDIDATE_STATES:
        state_events = [ev for ev in events if ev.get("state") == state]
        n = len(state_events)
        if n == 0:
            results[state] = {"n": 0}
            continue

        realized = [float(ev["realized_return_sd30d"])
                    for ev in state_events
                    if ev.get("realized_return_sd30d") is not None]
        excess   = [float(ev["excess_return"])
                    for ev in state_events
                    if ev.get("excess_return") is not None]
        fav_exc  = [float(ev["max_favorable_excursion_sd30d"])
                    for ev in state_events
                    if ev.get("max_favorable_excursion_sd30d") is not None]
        adv_exc  = [float(ev["max_adverse_excursion_sd30d"])
                    for ev in state_events
                    if ev.get("max_adverse_excursion_sd30d") is not None]
        bars_all = [int(ev["bars_held"])
                    for ev in state_events
                    if ev.get("bars_held") is not None]

        def _pct(vals: list[float], threshold: float, op: str = "gt") -> float:
            if not vals:
                return 0.0
            if op == "gt":
                return round(sum(1 for v in vals if v > threshold) / len(vals), 4)
            return round(sum(1 for v in vals if v < threshold) / len(vals), 4)

        def _mean(vals: list[float]) -> Optional[float]:
            return round(sum(vals) / len(vals), 4) if vals else None

        # Separate winners and losers (on realized return)
        winners = [ev for ev in state_events
                   if (ev.get("realized_return_sd30d") or 0.0) > 0]
        losers  = [ev for ev in state_events
                   if (ev.get("realized_return_sd30d") or 0.0) <= 0]
        bars_w  = [int(ev["bars_held"]) for ev in winners if ev.get("bars_held") is not None]
        bars_l  = [int(ev["bars_held"]) for ev in losers  if ev.get("bars_held") is not None]

        # Weekend crossing
        weekend_events = [ev for ev in state_events if _is_weekend_crossing(ev)]
        non_wknd       = [ev for ev in state_events if not _is_weekend_crossing(ev)]
        exc_wknd   = [float(ev["excess_return"]) for ev in weekend_events
                      if ev.get("excess_return") is not None]
        exc_nowknd = [float(ev["excess_return"]) for ev in non_wknd
                      if ev.get("excess_return") is not None]

        results[state] = {
            "n":                n,
            # Return tail proportions (on excess return)
            "pct_excess_gt0":   _pct(excess, 0.0),
            "pct_excess_gt1":   _pct(excess, 1.0),
            "pct_excess_gt2":   _pct(excess, 2.0),
            "pct_excess_gt4":   _pct(excess, 4.0),
            "pct_excess_lt_neg1": _pct(excess, -1.0, "lt"),
            "pct_excess_lt_neg2": _pct(excess, -2.0, "lt"),
            # Realized return tails
            "pct_realized_gt1":  _pct(realized, 1.0),
            "pct_realized_gt2":  _pct(realized, 2.0),
            "pct_realized_gt4":  _pct(realized, 4.0),
            # Excursion means
            "mean_fav_exc":      _mean(fav_exc),
            "mean_adv_exc":      _mean(adv_exc),
            "fav_to_adv_ratio":  round(sum(fav_exc) / sum(adv_exc), 3)
                                  if adv_exc and sum(adv_exc) > 0 else None,
            # Bars held
            "mean_bars_all":     _mean([float(b) for b in bars_all]),
            "mean_bars_winners": _mean([float(b) for b in bars_w]),
            "mean_bars_losers":  _mean([float(b) for b in bars_l]),
            "median_bars_all":   round(_median([float(b) for b in bars_all]), 1) if bars_all else None,
            # Weekend
            "n_weekend_crossing": len(weekend_events),
            "pct_weekend":        round(len(weekend_events) / n, 4),
            "mean_exc_weekend":   _mean(exc_wknd),
            "mean_exc_no_weekend":_mean(exc_nowknd),
        }

    return results


# ---------------------------------------------------------------------------
# Section 5 — Probability calibration
# ---------------------------------------------------------------------------

def run_probability_calibration(
    events: list[dict],
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, dict]:
    """
    For each candidate state, estimate empirical probabilities at multiple
    return thresholds with block bootstrap CIs.
    """
    results: dict[str, dict] = {}

    for state in CANDIDATE_STATES:
        excess = [
            float(ev["excess_return"])
            for ev in events
            if ev.get("state") == state and ev.get("excess_return") is not None
        ]
        n = len(excess)
        if n == 0:
            results[state] = {"n": 0}
            continue

        def _prob(threshold: float) -> float:
            return round(sum(1 for r in excess if r > threshold) / n, 4)

        # Bootstrap CI for P(excess > 0)
        indicator_pos = [1.0 if r > 0 else 0.0 for r in excess]
        boot_pos = block_bootstrap_ci(indicator_pos, n_bootstrap=n_bootstrap)

        # Bootstrap CI for mean excess return
        boot_mean = block_bootstrap_ci(excess, n_bootstrap=n_bootstrap)

        results[state] = {
            "n":                  n,
            "p_excess_gt_neg2":   _prob(-2.0),
            "p_excess_gt_neg1":   _prob(-1.0),
            "p_excess_gt_0":      _prob(0.0),
            "p_excess_gt_1":      _prob(1.0),
            "p_excess_gt_2":      _prob(2.0),
            "p_excess_gt_4":      _prob(4.0),
            "expected_excess":    round(sum(excess) / n, 4),
            "ci_low_mean":        boot_mean["ci_low"],
            "ci_high_mean":       boot_mean["ci_high"],
            "ci_low_p_pos":       boot_pos["ci_low"],
            "ci_high_p_pos":      boot_pos["ci_high"],
            "lower_bound_mean":   boot_mean["ci_low"],   # conservative estimate for sizing
        }

    return results


# ---------------------------------------------------------------------------
# Section 6 — Simplicity sanity check
# ---------------------------------------------------------------------------

SIMPLE_RULES: dict[str, object] = {
    "all_UP":            lambda ev: ev["breakout_direction"] == "UP",
    "all_DOWN":          lambda ev: ev["breakout_direction"] == "DOWN",
    "all_TRENDING_UP":   lambda ev: ev["efficiency_20"] > 0.60 and ev["breakout_direction"] == "UP",
    "all_TRENDING_DOWN": lambda ev: ev["efficiency_20"] > 0.60 and ev["breakout_direction"] == "DOWN",
    "all_CONTRACTING":   lambda ev: ev["vol_ratio_20_100"] < 0.80,
    "all_EXPANDING":     lambda ev: ev["vol_ratio_20_100"] > 1.20,
}


def run_simplicity_sanity(events: list[dict]) -> dict[str, dict]:
    """
    Compare candidate-state excess return vs simpler directional/regime rules.
    Primary question: does the joint state deliver materially more than the simpler rule?
    """
    base_mean = cohort_baseline_mean(events)
    results: dict[str, dict] = {}

    # Candidate state metrics (baseline for comparison)
    for state in CANDIDATE_STATES:
        exc = [float(ev["excess_return"]) for ev in events
               if ev.get("state") == state and ev.get("excess_return") is not None]
        m = metrics(exc)
        results[f"CANDIDATE_{state}"] = {
            "n":      m["n"],
            "mean":   m["mean"],
            "lift":   round(m["mean"] - base_mean, 4) if m["mean"] is not None else None,
            "win_rate": m["win_rate"],
        }

    # Simple rules
    for rule_name, qualifier in SIMPLE_RULES.items():
        exc = [float(ev["excess_return"]) for ev in events
               if qualifier(ev) and ev.get("excess_return") is not None]
        m = metrics(exc)
        results[rule_name] = {
            "n":       m["n"],
            "mean":    m["mean"],
            "lift":    round(m["mean"] - base_mean, 4) if m["mean"] is not None else None,
            "win_rate":m["win_rate"],
        }

    return results


# ---------------------------------------------------------------------------
# Section 7 — Pre-sizing promotion criteria and verdict
# ---------------------------------------------------------------------------

def evaluate_pre_sizing_criteria(
    state: str,
    threshold_stability: dict,
    continuous_model: dict,
    friction_stress: dict,
    probability_calib: dict,
    simplicity_sanity: dict,
) -> dict:
    """
    Apply pre-sizing promotion criteria and return structured verdict.

    Criteria
    --------
    C1. threshold_stability : positive excess in >= 2/3 threshold sets
    C2. continuous_model    : OLS prediction at candidate centroid is positive
    C3. friction_medium     : positive cost-adjusted mean at medium cost level
    C4. bootstrap_calibration: P(excess > 0) lower CI >= 0.35 (conservative)
    C5. simplicity_gap      : candidate mean > best simple-rule mean by >= 0.05 SD30d

    Verdict
    -------
    PROCEED          : all 5 pass -> advance to sizing research
    CONTINUE_MONITOR : 4/5 pass   -> extend forward monitoring, do not size
    REJECT           : <= 3/5     -> evidence too weak for capital allocation
    """
    criteria: dict[str, dict] = {}

    # C1: threshold stability
    ts = threshold_stability.get(state, {})
    n_positive = sum(1 for v in ts.values() if v.get("positive"))
    n_sets     = len(ts)
    criteria["threshold_stability"] = {
        "pass":       n_positive >= 2,
        "n_positive": n_positive,
        "n_sets":     n_sets,
        "detail":     {k: v.get("positive") for k, v in ts.items()},
    }

    # C2: continuous model
    pred = (continuous_model.get("predictions") or {}).get(state, {})
    criteria["continuous_model"] = {
        "pass":      pred.get("agrees", False) is True,
        "predicted": pred.get("predicted"),
        "observed":  pred.get("observed"),
    }

    # C3: friction medium level
    fs  = (friction_stress.get(state) or {}).get(COST_PASS_LEVEL, {})
    criteria["friction_medium"] = {
        "pass":      fs.get("survives", False),
        "mean_adj":  fs.get("mean_adj"),
        "cost_level":COST_PASS_LEVEL,
        "base_cost": fs.get("base_cost"),
    }

    # C4: bootstrap calibration (P(excess > 0) lower bound)
    pc    = probability_calib.get(state, {})
    ci_lo = pc.get("ci_low_p_pos")
    criteria["bootstrap_calibration"] = {
        "pass":       ci_lo is not None and ci_lo >= 0.35,
        "ci_low_p_pos": ci_lo,
        "p_excess_gt_0": pc.get("p_excess_gt_0"),
    }

    # C5: simplicity gap
    candidate_mean = (simplicity_sanity.get(f"CANDIDATE_{state}") or {}).get("mean")
    simple_means   = [v["mean"] for k, v in simplicity_sanity.items()
                      if not k.startswith("CANDIDATE_") and v.get("mean") is not None]
    best_simple    = max(simple_means) if simple_means else None
    gap            = round(candidate_mean - best_simple, 4) if (candidate_mean and best_simple) else None
    criteria["simplicity_gap"] = {
        "pass":           gap is not None and gap >= 0.05,
        "candidate_mean": candidate_mean,
        "best_simple":    best_simple,
        "gap":            gap,
    }

    n_pass  = sum(1 for c in criteria.values() if c.get("pass"))
    n_total = len(criteria)
    if n_pass == n_total:
        verdict = "PROCEED"
    elif n_pass >= 4:
        verdict = "CONTINUE_MONITOR"
    else:
        verdict = "REJECT"

    return {
        "state":    state,
        "verdict":  verdict,
        "n_pass":   n_pass,
        "n_total":  n_total,
        "criteria": criteria,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_threshold_stability(results: dict[str, dict[str, dict]]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 1 — Threshold stability")
    log.info("=" * 80)
    for state, tsets in results.items():
        log.info("")
        log.info("  %s", state)
        log.info("    %-10s  %7s  %9s  %9s  %9s  %9s  %10s  %10s  %10s  %8s",
                 "thresholds", "n", "mean", "median", "win_rate", "lift",
                 "ci_low", "ci_high", "frac_pos", "positive")
        log.info("    " + "-" * 110)
        for tset_name, r in tsets.items():
            log.info(
                "    %-10s  %7d  %9s  %9s  %9s  %9s  %10s  %10s  %10s  %8s",
                tset_name,
                r.get("n") or 0,
                f"{r['mean']:+.4f}"   if r.get("mean")    is not None else "N/A",
                f"{r['median']:+.4f}" if r.get("median")  is not None else "N/A",
                f"{r['win_rate']:.4f}" if r.get("win_rate") is not None else "N/A",
                f"{r['lift']:+.4f}"   if r.get("lift")    is not None else "N/A",
                f"{r['ci_low']:+.4f}" if r.get("ci_low")  is not None else "N/A",
                f"{r['ci_high']:+.4f}" if r.get("ci_high") is not None else "N/A",
                f"{r['frac_pos']:.3f}" if r.get("frac_pos") is not None else "N/A",
                "YES" if r.get("positive") else "NO",
            )


def print_continuous_model(result: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 2 — Continuous model (OLS)")
    log.info("=" * 80)
    log.info("  Features: %s", result.get("feature_names"))
    log.info("  Coefficients: %s", result.get("beta"))
    log.info("  R-squared   : %s", result.get("r_squared"))
    log.info("")
    log.info("  Pearson correlations with excess_return:")
    log.info("    vol_ratio_c     : %+.4f", result.get("corr_vol") or 0.0)
    log.info("    efficiency_c    : %+.4f", result.get("corr_eff") or 0.0)
    log.info("    interaction     : %+.4f", result.get("corr_interaction") or 0.0)
    log.info("    direction       : %+.4f", result.get("corr_direction") or 0.0)
    log.info("")
    log.info("  Candidate centroid predictions:")
    for state, pred in (result.get("predictions") or {}).items():
        log.info(
            "    %-35s  n=%d  predicted=%s  observed=%s  agrees=%s",
            state,
            pred.get("n") or 0,
            f"{pred['predicted']:+.4f}" if pred.get("predicted") is not None else "N/A",
            f"{pred['observed']:+.4f}"  if pred.get("observed")  is not None else "N/A",
            "YES" if pred.get("agrees") else "NO",
        )


def print_friction_stress(results: dict[str, dict[str, dict]]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 3 — Friction stress test")
    log.info("=" * 80)
    log.info("  Weekend extra cost: %.2f SD30d (trades starting Thu/Fri, held >= 24h)", WEEKEND_EXTRA_COST)
    for state, levels in results.items():
        log.info("")
        log.info("  %s", state)
        log.info("    %-8s  %6s  %9s  %9s  %9s  %9s  %10s  %10s  %9s  %8s",
                 "level", "cost", "mean_raw", "mean_adj", "win_adj", "payoff_adj",
                 "ci_low", "ci_high", "frac_pos", "survives")
        log.info("    " + "-" * 107)
        for level_name, r in levels.items():
            log.info(
                "    %-8s  %6.2f  %9s  %9s  %9s  %9s  %10s  %10s  %9s  %8s",
                level_name,
                r.get("base_cost") or 0.0,
                f"{r['mean_raw']:+.4f}"    if r.get("mean_raw")    is not None else "N/A",
                f"{r['mean_adj']:+.4f}"    if r.get("mean_adj")    is not None else "N/A",
                f"{r['win_rate_adj']:.4f}" if r.get("win_rate_adj") is not None else "N/A",
                str(r["payoff_adj"])        if r.get("payoff_adj")  is not None else "N/A",
                f"{r['ci_low']:+.4f}"      if r.get("ci_low")      is not None else "N/A",
                f"{r['ci_high']:+.4f}"     if r.get("ci_high")      is not None else "N/A",
                f"{r['frac_pos']:.3f}"     if r.get("frac_pos")    is not None else "N/A",
                "YES" if r.get("survives") else "NO",
            )


def print_tail_lifecycle(results: dict[str, dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 4 — Tail and lifecycle decomposition")
    log.info("=" * 80)
    for state, r in results.items():
        if r.get("n", 0) == 0:
            log.info("  %s: no data", state)
            continue
        log.info("")
        log.info("  %s  (n=%d)", state, r["n"])
        log.info("    Return tail (excess):")
        log.info("      P(excess > 0)   : %.4f", r.get("pct_excess_gt0") or 0.0)
        log.info("      P(excess > +1)  : %.4f  (tail winner, > 1 SD30d)", r.get("pct_excess_gt1") or 0.0)
        log.info("      P(excess > +2)  : %.4f", r.get("pct_excess_gt2") or 0.0)
        log.info("      P(excess > +4)  : %.4f", r.get("pct_excess_gt4") or 0.0)
        log.info("      P(excess < -1)  : %.4f  (significant loss)", r.get("pct_excess_lt_neg1") or 0.0)
        log.info("      P(excess < -2)  : %.4f", r.get("pct_excess_lt_neg2") or 0.0)
        log.info("    Return tail (realized):")
        log.info("      P(realized > +1): %.4f", r.get("pct_realized_gt1") or 0.0)
        log.info("      P(realized > +2): %.4f", r.get("pct_realized_gt2") or 0.0)
        log.info("      P(realized > +4): %.4f", r.get("pct_realized_gt4") or 0.0)
        log.info("    Excursion profile:")
        log.info("      Mean fav. exc.  : %s", f"{r['mean_fav_exc']:.4f}" if r.get("mean_fav_exc") is not None else "N/A")
        log.info("      Mean adv. exc.  : %s", f"{r['mean_adv_exc']:.4f}" if r.get("mean_adv_exc") is not None else "N/A")
        log.info("      Fav/Adv ratio   : %s", str(r.get("fav_to_adv_ratio")) or "N/A")
        log.info("    Duration (bars = hours):")
        log.info("      Mean all        : %s", f"{r['mean_bars_all']:.1f}" if r.get("mean_bars_all") is not None else "N/A")
        log.info("      Mean winners    : %s", f"{r['mean_bars_winners']:.1f}" if r.get("mean_bars_winners") is not None else "N/A")
        log.info("      Mean losers     : %s", f"{r['mean_bars_losers']:.1f}" if r.get("mean_bars_losers") is not None else "N/A")
        log.info("      Median all      : %s", f"{r['median_bars_all']:.1f}" if r.get("median_bars_all") is not None else "N/A")
        log.info("    Weekend crossing (proxy: starts Thu/Fri, >= 24h):")
        log.info("      Count           : %d  (%.1f%%)",
                 r.get("n_weekend_crossing") or 0,
                 (r.get("pct_weekend") or 0.0) * 100)
        log.info("      Mean exc (wknd) : %s", f"{r['mean_exc_weekend']:+.4f}" if r.get("mean_exc_weekend") is not None else "N/A")
        log.info("      Mean exc (else) : %s", f"{r['mean_exc_no_weekend']:+.4f}" if r.get("mean_exc_no_weekend") is not None else "N/A")


def print_probability_calibration(results: dict[str, dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 5 — Probability calibration")
    log.info("=" * 80)
    for state, r in results.items():
        if r.get("n", 0) == 0:
            log.info("  %s: no data", state)
            continue
        log.info("")
        log.info("  %s  (n=%d)", state, r["n"])
        log.info("    P(excess > -2)  : %.4f", r.get("p_excess_gt_neg2") or 0.0)
        log.info("    P(excess > -1)  : %.4f", r.get("p_excess_gt_neg1") or 0.0)
        log.info("    P(excess > 0)   : %.4f  [95%% CI: %s, %s]",
                 r.get("p_excess_gt_0") or 0.0,
                 f"{r['ci_low_p_pos']:+.4f}"  if r.get("ci_low_p_pos")  is not None else "N/A",
                 f"{r['ci_high_p_pos']:+.4f}" if r.get("ci_high_p_pos") is not None else "N/A")
        log.info("    P(excess > +1)  : %.4f", r.get("p_excess_gt_1") or 0.0)
        log.info("    P(excess > +2)  : %.4f", r.get("p_excess_gt_2") or 0.0)
        log.info("    P(excess > +4)  : %.4f", r.get("p_excess_gt_4") or 0.0)
        log.info("    Expected excess : %s  [95%% CI: %s, %s]",
                 f"{r['expected_excess']:+.4f}" if r.get("expected_excess") is not None else "N/A",
                 f"{r['ci_low_mean']:+.4f}"     if r.get("ci_low_mean")    is not None else "N/A",
                 f"{r['ci_high_mean']:+.4f}"    if r.get("ci_high_mean")   is not None else "N/A")
        log.info("    Conservative estimate (CI low): %s",
                 f"{r['lower_bound_mean']:+.4f}" if r.get("lower_bound_mean") is not None else "N/A")


def print_simplicity_sanity(results: dict[str, dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  SECTION 6 — Simplicity sanity check")
    log.info("=" * 80)
    log.info("  %-35s  %7s  %9s  %9s  %9s",
             "rule", "n", "mean_exc", "lift", "win_rate")
    log.info("  " + "-" * 75)
    for rule_name, r in sorted(results.items(), key=lambda x: x[1].get("mean") or -999, reverse=True):
        marker = "  <-- CANDIDATE" if rule_name.startswith("CANDIDATE_") else ""
        log.info(
            "  %-35s  %7d  %9s  %9s  %9s%s",
            rule_name,
            r.get("n") or 0,
            f"{r['mean']:+.4f}"    if r.get("mean")     is not None else "N/A",
            f"{r['lift']:+.4f}"    if r.get("lift")     is not None else "N/A",
            f"{r['win_rate']:.4f}" if r.get("win_rate") is not None else "N/A",
            marker,
        )


def print_criteria_verdict(result: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  VERDICT: %-35s  %s  (%d/%d criteria pass)",
             result["state"], result["verdict"], result["n_pass"], result["n_total"])
    log.info("=" * 80)
    for crit_name, c in result["criteria"].items():
        status = "PASS" if c.get("pass") else "FAIL"
        log.info("  [%s]  %s", status, crit_name)
        if crit_name == "threshold_stability":
            log.info("         %d/%d threshold sets positive", c["n_positive"], c["n_sets"])
            for k, v in c.get("detail", {}).items():
                log.info("         %s: %s", k, "ok" if v else "FAIL")
        elif crit_name == "continuous_model":
            log.info("         predicted=%s  observed=%s",
                     f"{c['predicted']:+.4f}" if c.get("predicted") is not None else "N/A",
                     f"{c['observed']:+.4f}"  if c.get("observed")  is not None else "N/A")
        elif crit_name == "friction_medium":
            log.info("         cost=%.2f SD30d  mean_adj=%s",
                     c.get("base_cost") or 0.0,
                     f"{c['mean_adj']:+.4f}" if c.get("mean_adj") is not None else "N/A")
        elif crit_name == "bootstrap_calibration":
            log.info("         P(excess>0)=%.4f  CI_low=%.4f  threshold=0.35",
                     c.get("p_excess_gt_0") or 0.0,
                     c.get("ci_low_p_pos") or 0.0)
        elif crit_name == "simplicity_gap":
            log.info("         candidate=%.4f  best_simple=%.4f  gap=%s  threshold=0.05",
                     c.get("candidate_mean") or 0.0,
                     c.get("best_simple") or 0.0,
                     f"{c['gap']:+.4f}" if c.get("gap") is not None else "N/A")


def print_final_recommendation(verdicts: list[dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  FINAL RECOMMENDATION")
    log.info("=" * 80)
    log.info("  %-35s  %-20s  %s", "state", "verdict", "criteria")
    log.info("  " + "-" * 70)
    for v in verdicts:
        log.info("  %-35s  %-20s  %d/%d", v["state"], v["verdict"], v["n_pass"], v["n_total"])
    log.info("")
    log.info("  Promotion criteria for PROCEED:")
    log.info("    C1. threshold_stability  : positive excess in >= 2/3 threshold sets")
    log.info("    C2. continuous_model     : OLS centroid prediction is positive")
    log.info("    C3. friction_medium      : positive cost-adjusted mean at medium cost (%.2f SD30d)", COST_LEVELS["medium"])
    log.info("    C4. bootstrap_calibration: P(excess > 0) bootstrap CI lower bound >= 0.35")
    log.info("    C5. simplicity_gap       : candidate mean > best simple rule by >= 0.05 SD30d")
    log.info("")
    log.info("  PROCEED = all 5.  CONTINUE_MONITOR = 4/5.  REJECT = <= 3/5.")
    log.info("")
    overall = [v["verdict"] for v in verdicts]
    if all(v == "PROCEED" for v in overall):
        log.info("  RECOMMENDATION: PROCEED to position sizing research.")
        log.info("  Both candidate states meet all pre-deployment criteria.")
        log.info("  Size conservatively against test-split mean, not pooled mean.")
    elif all(v in ("PROCEED", "CONTINUE_MONITOR") for v in overall):
        log.info("  RECOMMENDATION: CONTINUE MONITORING.")
        log.info("  At least one state has not cleared all pre-deployment criteria.")
        log.info("  Continue forward monitoring for one additional quarter.")
    else:
        log.info("  RECOMMENDATION: REJECT at least one state.")
        log.info("  Insufficient evidence for capital allocation research.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(output_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-deployment validation of candidate Markov states (Issue #93)."
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap (faster, no CIs)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help=f"Bootstrap iterations (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional CSV path for summary export",
    )
    args = parser.parse_args()
    n_boot = 0 if args.no_bootstrap else args.n_bootstrap

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info("Loading FX breakout events ...")
    all_events = load_events()
    log.info("Loaded %d state-labeled events", len(all_events))
    if not all_events:
        log.error("No events found. Run backfill_breakout_events.py first.")
        sys.exit(1)

    # Carry neutralization
    baselines     = compute_baselines(all_events)
    excess_events = apply_excess_return(all_events, baselines)
    log.info("Carry neutralization applied: %d events with excess_return", len(excess_events))

    # Filter to cohort
    present        = {ev["symbol"] for ev in excess_events}
    active_symbols = [s for s in COHORT_SYMBOLS if s in present]
    missing        = [s for s in COHORT_SYMBOLS if s not in present]
    if missing:
        log.warning("Symbols not in DB: %s", missing)
    cohort_events = [ev for ev in excess_events if ev["symbol"] in set(active_symbols)]
    log.info("Cohort core_majors: %d events  symbols: %s", len(cohort_events), active_symbols)

    # --- Section 1: Threshold stability ---
    log.info("Running Section 1: Threshold stability ...")
    thresh_results = run_threshold_stability(cohort_events, n_bootstrap=n_boot)
    print_threshold_stability(thresh_results)

    # --- Section 2: Continuous model ---
    log.info("Running Section 2: Continuous model (OLS) ...")
    ols_result = run_continuous_model(cohort_events)
    print_continuous_model(ols_result)

    # --- Section 3: Friction stress ---
    log.info("Running Section 3: Friction stress test ...")
    friction_results = run_friction_stress(cohort_events, n_bootstrap=n_boot)
    print_friction_stress(friction_results)

    # --- Section 4: Tail and lifecycle ---
    log.info("Running Section 4: Tail and lifecycle decomposition ...")
    tail_results = run_tail_lifecycle(cohort_events)
    print_tail_lifecycle(tail_results)

    # --- Section 5: Probability calibration ---
    log.info("Running Section 5: Probability calibration ...")
    calib_results = run_probability_calibration(cohort_events, n_bootstrap=n_boot)
    print_probability_calibration(calib_results)

    # --- Section 6: Simplicity sanity ---
    log.info("Running Section 6: Simplicity sanity check ...")
    sanity_results = run_simplicity_sanity(cohort_events)
    print_simplicity_sanity(sanity_results)

    # --- Section 7: Pre-sizing criteria ---
    log.info("Running Section 7: Pre-sizing promotion criteria ...")
    verdicts: list[dict] = []
    for state in CANDIDATE_STATES:
        v = evaluate_pre_sizing_criteria(
            state,
            thresh_results,
            ols_result,
            friction_results,
            calib_results,
            sanity_results,
        )
        print_criteria_verdict(v)
        verdicts.append(v)

    print_final_recommendation(verdicts)

    # CSV export: one row per (state, section, metric) not practical;
    # export a flattened summary of key metrics per state instead
    if args.output:
        rows = []
        for state in CANDIDATE_STATES:
            pc  = calib_results.get(state, {})
            frc = (friction_results.get(state) or {}).get("medium", {})
            tl  = tail_results.get(state, {})
            v   = next((x for x in verdicts if x["state"] == state), {})
            rows.append({
                "state":                state,
                "verdict":              v.get("verdict"),
                "n_pass":               v.get("n_pass"),
                "n":                    pc.get("n"),
                "expected_excess":      pc.get("expected_excess"),
                "ci_low_mean":          pc.get("ci_low_mean"),
                "ci_high_mean":         pc.get("ci_high_mean"),
                "p_excess_gt_0":        pc.get("p_excess_gt_0"),
                "p_excess_gt_1":        pc.get("p_excess_gt_1"),
                "p_excess_gt_2":        pc.get("p_excess_gt_2"),
                "mean_adj_medium":      frc.get("mean_adj"),
                "survives_medium_cost": frc.get("survives"),
                "fav_adv_ratio":        tl.get("fav_to_adv_ratio"),
                "mean_bars_winners":    tl.get("mean_bars_winners"),
                "mean_bars_losers":     tl.get("mean_bars_losers"),
                "pct_realized_gt2":     tl.get("pct_realized_gt2"),
            })
        write_csv(args.output, rows)


if __name__ == "__main__":
    main()

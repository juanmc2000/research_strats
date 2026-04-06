"""
scripts/markov_validation.py

Formal validation of candidate Markov alpha states -- Issue #92.

Moves from exploratory signal to formally validated candidate alpha by applying:

  1. Frozen candidate state registry (pre-test commitment, no post-hoc selection)
  2. Time-split validation  (train 2015-2020 / validation 2021-2023 / test 2024-2026)
  3. Block bootstrap uncertainty estimation (dependence-aware CIs)
  4. Leave-one-symbol-out robustness within each cohort
  5. Simpler model comparison (direction-only, dir+vol, dir+eff vs full 18-state)
  6. Multiple-testing context document (Bonferroni upper bound)
  7. Effect-size ranking table for all qualifying states
  8. Pass/fail verdict per candidate state

Frozen candidate states
-----------------------
  Primary:
    CONTRACTING_TRENDING_UP
    EXPANDING_TRENDING_DOWN  (core_majors focus)

  Secondary candidates remain exploratory and are NOT evaluated here.

All analysis is carry-neutralized: symbol+direction mean return subtracted before
computing any excess return statistic.  The baseline is computed globally over the
full universe and applied uniformly -- it is not cohort-local.

Excluded from scope
-------------------
  Position sizing, pyramiding, live execution, hidden Markov models.

Usage
-----
    python scripts/markov_validation.py
    python scripts/markov_validation.py --cohort core_majors
    python scripts/markov_validation.py --no-bootstrap
    python scripts/markov_validation.py --min-count 20 --n-bootstrap 2000
    python scripts/markov_validation.py --output results/markov_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
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
# DB connection (same pattern as carry_neutralized_analysis.py)
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
# Frozen candidate states (pre-registered, do not extend without new issue)
# ---------------------------------------------------------------------------
CANDIDATE_STATES: list[str] = [
    "CONTRACTING_TRENDING_UP",
    "EXPANDING_TRENDING_DOWN",
]

# Cohorts (mirrors carry_neutralized_analysis.py)
COHORTS: dict[str, list[str]] = {
    "core_majors": ["EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"],
    "jpy_crosses": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY", "CHF/JPY", "CAD/JPY"],
    "high_drift":  ["USD/TRY", "EUR/TRY"],
}

# Chronological time splits
TIME_SPLITS: dict[str, tuple[str, str]] = {
    "train":      ("2015-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2023-12-31"),
    "test":       ("2024-01-01", "2026-12-31"),
}

# Bootstrap configuration
N_BOOTSTRAP: int   = 1000
BLOCK_SIZE:  int   = 20      # bars; preserves short-range autocorrelation
RANDOM_SEED: int   = 42

# Pass/fail promotion thresholds (documented, objective)
PASS_MIN_COUNT:      int   = 20    # minimum events per state-split for it to count
PASS_BOOTSTRAP_FRAC: float = 0.70  # fraction of bootstrap means > 0
PASS_LOSO_FRAC:      float = 0.80  # fraction of LOSO iterations: state excess return > 0

# ---------------------------------------------------------------------------
# State space (mirrors markov_state_analysis.py)
# ---------------------------------------------------------------------------
ALL_STATES: list[str] = [
    f"{v}_{e}_{d}"
    for v in ("CONTRACTING", "NEUTRAL", "EXPANDING")
    for e in ("CHOPPY", "MIXED", "TRENDING")
    for d in ("UP", "DOWN")
]


def _vol_bucket(vol_ratio: float) -> str:
    if vol_ratio < 0.8:
        return "CONTRACTING"
    if vol_ratio <= 1.2:
        return "NEUTRAL"
    return "EXPANDING"


def _eff_bucket(efficiency: float) -> str:
    if efficiency < 0.3:
        return "CHOPPY"
    if efficiency <= 0.6:
        return "MIXED"
    return "TRENDING"


def _median(values: list[float]) -> float:
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def _ts_str(ts: object) -> str:
    """Convert a psycopg2 timestamp to a YYYY-MM-DD string for range comparisons."""
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


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
        ev = dict(row)
        vr  = ev.get("vol_ratio_20_100")
        eff = ev.get("efficiency_20")
        d   = ev.get("breakout_direction", "")
        if vr is None or eff is None or d not in ("UP", "DOWN"):
            continue
        vb = _vol_bucket(float(vr))
        eb = _eff_bucket(float(eff))
        ev["state"]        = f"{vb}_{eb}_{d}"
        ev["model0_state"] = d
        ev["model1_state"] = f"{vb}_{d}"
        ev["model2_state"] = f"{eb}_{d}"
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Carry neutralization (global baseline, not cohort-local)
# ---------------------------------------------------------------------------

def compute_baselines(events: list[dict]) -> dict[tuple[str, str], float]:
    """Return mean realized_return_sd30d per (symbol, direction) over the full universe."""
    groups: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        key = (ev["symbol"], ev["breakout_direction"])
        groups.setdefault(key, []).append(float(ev["realized_return_sd30d"]))
    return {k: sum(v) / len(v) for k, v in groups.items()}


def apply_excess_return(
    events: list[dict],
    baselines: dict[tuple[str, str], float],
) -> list[dict]:
    """Return events with excess_return field appended.  Events missing a baseline are dropped."""
    out: list[dict] = []
    skipped = 0
    for ev in events:
        key = (ev["symbol"], ev["breakout_direction"])
        b = baselines.get(key)
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
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_by_time(events: list[dict], start: str, end: str) -> list[dict]:
    """Filter events to [start, end] (YYYY-MM-DD strings, inclusive)."""
    return [ev for ev in events if start <= _ts_str(ev["event_hour_ts"]) <= end]


def filter_by_symbols(events: list[dict], symbols: list[str]) -> list[dict]:
    sym_set = set(symbols)
    return [ev for ev in events if ev["symbol"] in sym_set]


# ---------------------------------------------------------------------------
# State and cohort metric helpers
# ---------------------------------------------------------------------------

def _state_returns(
    events: list[dict],
    state_field: str,
    state_value: str,
    return_field: str = "excess_return",
) -> list[float]:
    return [
        float(ev[return_field])
        for ev in events
        if ev.get(state_field) == state_value and ev.get(return_field) is not None
    ]


def compute_state_metrics(
    events: list[dict],
    state_field: str,
    state_value: str,
    return_field: str = "excess_return",
) -> dict:
    returns = _state_returns(events, state_field, state_value, return_field)
    n = len(returns)
    if n == 0:
        return {"count": 0, "mean": None, "median": None, "win_rate": None, "payoff": None}
    wins   = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    mean_r = sum(returns) / n
    med_r  = _median(returns)
    win_rate = len(wins) / n
    avg_win  = sum(wins) / len(wins) if wins else None
    avg_loss = abs(sum(losses) / len(losses)) if losses else None
    payoff   = round(avg_win / avg_loss, 3) if (avg_win and avg_loss) else None
    return {
        "count":    n,
        "mean":     round(mean_r, 4),
        "median":   round(med_r, 4),
        "win_rate": round(win_rate, 4),
        "payoff":   payoff,
    }


def compute_cohort_baseline(
    events: list[dict],
    return_field: str = "excess_return",
) -> dict:
    returns = [float(ev[return_field]) for ev in events if ev.get(return_field) is not None]
    n = len(returns)
    if n == 0:
        return {"count": 0, "mean": None, "win_rate": None}
    wins = [r for r in returns if r > 0]
    return {
        "count":    n,
        "mean":     round(sum(returns) / n, 4),
        "win_rate": round(len(wins) / n, 4),
    }


def _lift(
    state_mean: Optional[float],
    baseline_mean: Optional[float],
) -> Optional[float]:
    if state_mean is None or baseline_mean is None:
        return None
    return round(state_mean - baseline_mean, 4)


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    returns: list[float],
    block_size: int = BLOCK_SIZE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Circular block bootstrap for the mean of a dependent sample.

    Samples overlapping blocks of length block_size with replacement,
    concatenates until n observations are reached, trims to n.

    Returns ci_low, ci_high (95%), frac_positive, mean_obs.
    Returns all None if n == 0 or n_bootstrap == 0.
    """
    n = len(returns)
    if n == 0 or n_bootstrap == 0:
        mean_obs = round(sum(returns) / n, 4) if n > 0 else None
        return {"ci_low": None, "ci_high": None, "frac_positive": None, "mean_obs": mean_obs}

    mean_obs = sum(returns) / n
    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(n_bootstrap):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randint(0, n - 1)
            for k in range(block_size):
                sample.append(returns[(start + k) % n])
        sample = sample[:n]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    lo_idx  = max(0, int(round(0.025 * n_bootstrap)))
    hi_idx  = min(n_bootstrap - 1, int(round(0.975 * n_bootstrap)) - 1)
    frac_pos = sum(1 for m in bootstrap_means if m > 0) / n_bootstrap

    return {
        "ci_low":        round(bootstrap_means[lo_idx], 4),
        "ci_high":       round(bootstrap_means[hi_idx], 4),
        "frac_positive": round(frac_pos, 3),
        "mean_obs":      round(mean_obs, 4),
    }


# ---------------------------------------------------------------------------
# Task 2 — Time-split validation
# ---------------------------------------------------------------------------

def run_time_split_analysis(
    events: list[dict],
    candidate_state: str,
    min_count: int = PASS_MIN_COUNT,
) -> dict[str, dict]:
    """
    For each chronological split, compute excess return metrics and lift.
    Returns {split_name: {..., sign_ok: bool}}.
    sign_ok is True when n >= min_count AND lift > 0.
    """
    results: dict[str, dict] = {}
    for split_name, (start, end) in TIME_SPLITS.items():
        split_ev = filter_by_time(events, start, end)
        base     = compute_cohort_baseline(split_ev)
        metrics  = compute_state_metrics(split_ev, "state", candidate_state)
        lift     = _lift(metrics.get("mean"), base.get("mean"))
        n        = metrics["count"]
        results[split_name] = {
            "count":         n,
            "mean":          metrics.get("mean"),
            "median":        metrics.get("median"),
            "win_rate":      metrics.get("win_rate"),
            "payoff":        metrics.get("payoff"),
            "baseline_mean": base.get("mean"),
            "baseline_n":    base.get("count"),
            "lift":          lift,
            "sign_ok":       (lift is not None and lift > 0 and n >= min_count),
        }
    return results


# ---------------------------------------------------------------------------
# Task 4 — Leave-one-symbol-out robustness
# ---------------------------------------------------------------------------

def run_loso(
    events: list[dict],
    candidate_state: str,
    symbols: list[str],
    min_count: int = PASS_MIN_COUNT,
) -> dict:
    """
    Drop one symbol at a time and check whether candidate_state remains
    positive and well-ranked among qualifying states.

    Returns frac_positive, frac_top_half, and per-iteration details.
    """
    iterations: list[dict] = []
    for dropped in symbols:
        remaining = [s for s in symbols if s != dropped]
        subset    = filter_by_symbols(events, remaining)
        base      = compute_cohort_baseline(subset)
        metrics   = compute_state_metrics(subset, "state", candidate_state)
        n         = metrics["count"]
        mean_val  = metrics.get("mean")

        # Rank: gather all state means with sufficient count
        ranked_means: list[float] = []
        for s in ALL_STATES:
            m = compute_state_metrics(subset, "state", s)
            if m["count"] >= min_count and m.get("mean") is not None:
                ranked_means.append(m["mean"])
        ranked_means.sort(reverse=True)

        positive: bool = False
        rank:     Optional[int] = None
        top_half: bool = False

        if mean_val is not None and n >= min_count:
            positive = mean_val > 0
            # Rank = 1-based position; ties go to worst rank among ties
            rank = sum(1 for m in ranked_means if m >= mean_val)
            top_half = rank <= max(1, len(ranked_means) // 2)

        iterations.append({
            "dropped":       dropped,
            "n":             n,
            "mean":          mean_val,
            "baseline_mean": base.get("mean"),
            "lift":          _lift(mean_val, base.get("mean")),
            "positive":      positive,
            "rank":          rank,
            "top_half":      top_half,
        })

    n_iter = len(iterations)
    if n_iter == 0:
        return {"frac_positive": None, "frac_top_half": None, "iterations": []}

    frac_pos  = sum(1 for it in iterations if it["positive"])  / n_iter
    frac_half = sum(1 for it in iterations if it["top_half"])  / n_iter
    return {
        "frac_positive": round(frac_pos, 3),
        "frac_top_half": round(frac_half, 3),
        "iterations":    iterations,
    }


# ---------------------------------------------------------------------------
# Task 5 — Simpler model comparison
# ---------------------------------------------------------------------------

# Simpler models: field name -> human label
SIMPLER_MODELS: dict[str, str] = {
    "direction_only": "model0_state",
    "dir_plus_vol":   "model1_state",
    "dir_plus_eff":   "model2_state",
    "full_18state":   "state",
}


def run_model_comparison(
    events: list[dict],
    min_count: int = PASS_MIN_COUNT,
) -> dict[str, dict]:
    """
    For each model complexity level, find the best state's excess return and coverage.
    Primary question: does 18-state granularity add lift over simpler bucketing?
    """
    base          = compute_cohort_baseline(events)
    baseline_mean = base.get("mean") or 0.0
    total_n       = len(events)
    results: dict[str, dict] = {}

    for model_name, state_field in SIMPLER_MODELS.items():
        model_states = sorted({ev[state_field] for ev in events if ev.get(state_field)})
        best_state:  Optional[str]   = None
        best_mean:   Optional[float] = None
        n_above      = 0
        events_above = 0

        for s in model_states:
            m = compute_state_metrics(events, state_field, s)
            if m["count"] < min_count or m.get("mean") is None:
                continue
            if m["mean"] > baseline_mean:
                n_above      += 1
                events_above += m["count"]
            if best_mean is None or m["mean"] > best_mean:
                best_mean  = m["mean"]
                best_state = s

        results[model_name] = {
            "best_state":       best_state,
            "best_mean":        best_mean,
            "best_lift":        _lift(best_mean, baseline_mean),
            "n_above_baseline": n_above,
            "coverage_frac":    round(events_above / total_n, 3) if total_n > 0 else 0.0,
        }
    return results


# ---------------------------------------------------------------------------
# Task 6 — Multiple-testing documentation
# ---------------------------------------------------------------------------

def compute_bonferroni_context(
    n_candidate_states: int,
    n_cohorts: int,
    n_splits: int,
    alpha: float = 0.05,
) -> dict:
    """
    Document the effective upper bound on hypothesis comparisons and the
    implied Bonferroni-corrected significance threshold.

    This is not a strict cutoff but a floor for skepticism.
    """
    n_effective = n_candidate_states * n_cohorts * n_splits
    return {
        "n_candidate_states":  n_candidate_states,
        "n_cohorts":           n_cohorts,
        "n_splits":            n_splits,
        "n_effective_tests":   n_effective,
        "alpha_family":        alpha,
        "bonferroni_alpha":    round(alpha / n_effective, 5),
        "exploratory_n_implicit": "~108 (18 states x 3 cohorts x 2 directions)",
        "note": (
            "Effective N assumes independence -- correlations between cohorts and "
            "splits reduce the true N.  Use as a floor for skepticism, not a hard cutoff."
        ),
    }


# ---------------------------------------------------------------------------
# Task 7 — Effect-size ranking
# ---------------------------------------------------------------------------

def compute_effect_size_ranking(
    events: list[dict],
    min_count: int = PASS_MIN_COUNT,
    n_bootstrap: int = N_BOOTSTRAP,
) -> list[dict]:
    """
    Full effect-size profile for all states with n >= min_count.
    Sorted by mean_excess descending.  Candidate states flagged with is_candidate.
    """
    base          = compute_cohort_baseline(events)
    baseline_mean = base.get("mean") or 0.0
    rows: list[dict] = []

    for state in ALL_STATES:
        m = compute_state_metrics(events, "state", state)
        if m["count"] < min_count or m.get("mean") is None:
            continue
        returns = _state_returns(events, "state", state)
        boot    = block_bootstrap_ci(returns, n_bootstrap=n_bootstrap)
        rows.append({
            "state":              state,
            "count":              m["count"],
            "mean_excess":        m["mean"],
            "median_excess":      m["median"],
            "win_rate":           m["win_rate"],
            "payoff":             m["payoff"],
            "lift":               _lift(m["mean"], baseline_mean),
            "ci_low":             boot["ci_low"],
            "ci_high":            boot["ci_high"],
            "frac_positive_boot": boot["frac_positive"],
            "is_candidate":       state in CANDIDATE_STATES,
        })

    rows.sort(key=lambda r: r["mean_excess"] or 0.0, reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1
    return rows


# ---------------------------------------------------------------------------
# Task 8 — Pass/fail verdict
# ---------------------------------------------------------------------------

def evaluate_pass_fail(
    state: str,
    time_split_results: dict[str, dict],
    bootstrap_result: dict,
    loso_result: dict,
    model_comparison: dict,
    min_count: int = PASS_MIN_COUNT,
) -> dict:
    """
    Apply the four promotion criteria and return a structured verdict.

    Criteria
    --------
    1. time_splits   : positive excess return in ALL splits that have n >= min_count
                       and at least 2 splits have sufficient data
    2. bootstrap     : fraction of bootstrap means > 0 >= PASS_BOOTSTRAP_FRAC (0.70)
    3. loso          : fraction of LOSO iterations with positive excess return >= PASS_LOSO_FRAC (0.80)
    4. simpler_model : full 18-state best lift >= best lift of any simpler model

    Verdict
    -------
    PASS        : all 4 criteria pass
    CONDITIONAL : 3/4 criteria pass
    FAIL        : <= 2 criteria pass
    """
    criteria: dict[str, dict] = {}

    # Criterion 1
    splits_with_data = [
        (name, res) for name, res in time_split_results.items()
        if res["count"] >= min_count
    ]
    all_positive = all(res.get("sign_ok", False) for _, res in splits_with_data)
    criteria["time_splits"] = {
        "pass":               all_positive and len(splits_with_data) >= 2,
        "n_splits_with_data": len(splits_with_data),
        "detail":             {name: res.get("sign_ok") for name, res in splits_with_data},
    }

    # Criterion 2
    frac_pos = bootstrap_result.get("frac_positive")
    criteria["bootstrap"] = {
        "pass":          frac_pos is not None and frac_pos >= PASS_BOOTSTRAP_FRAC,
        "frac_positive": frac_pos,
        "ci_low":        bootstrap_result.get("ci_low"),
        "ci_high":       bootstrap_result.get("ci_high"),
        "threshold":     PASS_BOOTSTRAP_FRAC,
    }

    # Criterion 3
    loso_frac = loso_result.get("frac_positive")
    criteria["loso"] = {
        "pass":          loso_frac is not None and loso_frac >= PASS_LOSO_FRAC,
        "frac_positive": loso_frac,
        "frac_top_half": loso_result.get("frac_top_half"),
        "threshold":     PASS_LOSO_FRAC,
    }

    # Criterion 4
    full_lift    = (model_comparison.get("full_18state") or {}).get("best_lift")
    best_simpler = None
    for model_name in ("direction_only", "dir_plus_vol", "dir_plus_eff"):
        m_lift = (model_comparison.get(model_name) or {}).get("best_lift")
        if m_lift is not None and (best_simpler is None or m_lift > best_simpler):
            best_simpler = m_lift
    criteria["simpler_model"] = {
        "pass":              (full_lift is not None and best_simpler is not None
                              and full_lift >= best_simpler),
        "full_18state_lift": full_lift,
        "best_simpler_lift": best_simpler,
    }

    n_pass  = sum(1 for c in criteria.values() if c.get("pass"))
    n_total = len(criteria)
    if n_pass == n_total:
        verdict = "PASS"
    elif n_pass >= 3:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"

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

def print_bonferroni_context(ctx: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  MULTIPLE-TESTING CONTEXT")
    log.info("=" * 80)
    log.info("  Frozen candidate states      : %s", CANDIDATE_STATES)
    log.info("  Exploratory implicit N       : %s", ctx["exploratory_n_implicit"])
    log.info("  Pre-registered candidates    : %d state(s)", ctx["n_candidate_states"])
    log.info("  Conservative effective N     : %d  (candidates x cohorts x splits = %d x %d x %d)",
             ctx["n_effective_tests"],
             ctx["n_candidate_states"], ctx["n_cohorts"], ctx["n_splits"])
    log.info("  Bonferroni alpha at %.2f     : %.5f", ctx["alpha_family"], ctx["bonferroni_alpha"])
    log.info("  Note: %s", ctx["note"])


def print_time_split_table(state: str, results: dict[str, dict]) -> None:
    log.info("")
    log.info("  Time-split validation  [%s]", state)
    log.info(
        "    %-12s  %7s  %9s  %9s  %9s  %9s  %9s  %9s  %8s",
        "split", "n_state", "mean", "median", "win_rate", "payoff",
        "baseline", "lift", "sign_ok",
    )
    log.info("    " + "-" * 108)
    for split_name, res in results.items():
        log.info(
            "    %-12s  %7d  %9s  %9s  %9s  %9s  %9s  %9s  %8s",
            split_name,
            res["count"],
            f"{res['mean']:+.4f}"          if res.get("mean")         is not None else "N/A",
            f"{res['median']:+.4f}"        if res.get("median")       is not None else "N/A",
            f"{res['win_rate']:.4f}"       if res.get("win_rate")     is not None else "N/A",
            str(res["payoff"])             if res.get("payoff")       is not None else "N/A",
            f"{res['baseline_mean']:+.4f}" if res.get("baseline_mean") is not None else "N/A",
            f"{res['lift']:+.4f}"          if res.get("lift")         is not None else "N/A",
            "YES" if res.get("sign_ok") else "NO",
        )


def print_bootstrap_result(state: str, result: dict) -> None:
    log.info("")
    log.info("  Block bootstrap CI  [%s]  (block_size=%d, n=%d)",
             state, BLOCK_SIZE, N_BOOTSTRAP)
    log.info("    observed mean  : %s",
             f"{result['mean_obs']:+.4f}" if result.get("mean_obs") is not None else "N/A")
    log.info("    95%% CI         : [%s, %s]",
             f"{result['ci_low']:+.4f}"  if result.get("ci_low")  is not None else "N/A",
             f"{result['ci_high']:+.4f}" if result.get("ci_high") is not None else "N/A")
    log.info("    frac > 0       : %s  (threshold: %.2f)",
             f"{result['frac_positive']:.3f}" if result.get("frac_positive") is not None else "N/A",
             PASS_BOOTSTRAP_FRAC)


def print_loso_table(state: str, result: dict) -> None:
    log.info("")
    log.info("  Leave-one-symbol-out  [%s]", state)
    log.info("    frac_positive : %.3f  (threshold: %.2f)",
             result.get("frac_positive") or 0.0, PASS_LOSO_FRAC)
    log.info("    frac_top_half : %.3f", result.get("frac_top_half") or 0.0)
    log.info("    %-14s  %7s  %9s  %9s  %9s  %8s  %8s",
             "dropped", "n", "mean", "baseline", "lift", "positive", "top_half")
    log.info("    " + "-" * 80)
    for it in result.get("iterations", []):
        log.info(
            "    %-14s  %7d  %9s  %9s  %9s  %8s  %8s",
            it["dropped"],
            it.get("n") or 0,
            f"{it['mean']:+.4f}"          if it.get("mean")         is not None else "N/A",
            f"{it['baseline_mean']:+.4f}" if it.get("baseline_mean") is not None else "N/A",
            f"{it['lift']:+.4f}"          if it.get("lift")         is not None else "N/A",
            "YES" if it.get("positive") else "NO",
            "YES" if it.get("top_half") else "NO",
        )


def print_model_comparison(results: dict[str, dict]) -> None:
    log.info("")
    log.info("  Simpler model comparison:")
    log.info("    %-18s  %-35s  %9s  %9s  %8s  %10s",
             "model", "best_state", "best_mean", "best_lift", "n_above", "coverage")
    log.info("    " + "-" * 102)
    for model_name, res in results.items():
        log.info(
            "    %-18s  %-35s  %9s  %9s  %8s  %10s",
            model_name,
            res.get("best_state") or "N/A",
            f"{res['best_mean']:+.4f}"  if res.get("best_mean")  is not None else "N/A",
            f"{res['best_lift']:+.4f}"  if res.get("best_lift")  is not None else "N/A",
            res.get("n_above_baseline") or 0,
            f"{res['coverage_frac']:.3f}" if res.get("coverage_frac") is not None else "N/A",
        )
    log.info("")
    log.info("    Primary question: does full_18state best_lift exceed best_simpler best_lift?")


def print_effect_size_ranking(rows: list[dict]) -> None:
    log.info("")
    log.info("  Effect-size ranking  (all states with n >= min_count, sorted by mean_excess):")
    log.info(
        "    %-4s  %-35s  %7s  %9s  %9s  %9s  %9s  %10s  %10s  %13s",
        "rank", "state", "count",
        "mean_exc", "median", "win_rate", "lift",
        "ci_low", "ci_high", "frac_pos_boot",
    )
    log.info("    " + "-" * 130)
    for row in rows:
        marker = "  <-- CANDIDATE" if row.get("is_candidate") else ""
        log.info(
            "    %-4d  %-35s  %7d  %9s  %9s  %9s  %9s  %10s  %10s  %13s%s",
            row["rank"],
            row["state"],
            row["count"],
            f"{row['mean_excess']:+.4f}"    if row.get("mean_excess")        is not None else "N/A",
            f"{row['median_excess']:+.4f}"  if row.get("median_excess")      is not None else "N/A",
            f"{row['win_rate']:.4f}"        if row.get("win_rate")           is not None else "N/A",
            f"{row['lift']:+.4f}"           if row.get("lift")               is not None else "N/A",
            f"{row['ci_low']:+.4f}"         if row.get("ci_low")             is not None else "N/A",
            f"{row['ci_high']:+.4f}"        if row.get("ci_high")            is not None else "N/A",
            f"{row['frac_positive_boot']:.3f}" if row.get("frac_positive_boot") is not None else "N/A",
            marker,
        )


def print_verdict(result: dict) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  VERDICT: %-35s  %s  (%d/%d criteria pass)",
             result["state"], result["verdict"], result["n_pass"], result["n_total"])
    log.info("=" * 80)
    for crit_name, crit in result["criteria"].items():
        status = "PASS" if crit.get("pass") else "FAIL"
        log.info("  [%s]  %s", status, crit_name)
        if crit_name == "time_splits":
            for split_name, ok in crit.get("detail", {}).items():
                log.info("         %s: %s", split_name, "ok" if ok else "FAIL")
            log.info("         splits_with_data: %d", crit.get("n_splits_with_data", 0))
        elif crit_name == "bootstrap":
            log.info(
                "         frac_positive=%.3f  threshold=%.2f  CI=[%s, %s]",
                crit.get("frac_positive") or 0.0,
                crit.get("threshold") or 0.0,
                f"{crit['ci_low']:+.4f}"  if crit.get("ci_low")  is not None else "N/A",
                f"{crit['ci_high']:+.4f}" if crit.get("ci_high") is not None else "N/A",
            )
        elif crit_name == "loso":
            log.info(
                "         frac_positive=%.3f  threshold=%.2f  frac_top_half=%.3f",
                crit.get("frac_positive") or 0.0,
                crit.get("threshold")     or 0.0,
                crit.get("frac_top_half") or 0.0,
            )
        elif crit_name == "simpler_model":
            log.info(
                "         full_18state_lift=%s  best_simpler_lift=%s",
                f"{crit['full_18state_lift']:+.4f}" if crit.get("full_18state_lift") is not None else "N/A",
                f"{crit['best_simpler_lift']:+.4f}" if crit.get("best_simpler_lift") is not None else "N/A",
            )


def print_final_summary(verdicts: list[dict]) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  FINAL SUMMARY")
    log.info("=" * 80)
    log.info(
        "  %-35s  %-13s  %8s  %s",
        "state", "verdict", "criteria", "cohort",
    )
    log.info("  " + "-" * 75)
    for v in verdicts:
        log.info(
            "  %-35s  %-13s  %d/%d      %s",
            v["state"], v["verdict"], v["n_pass"], v["n_total"], v["cohort"],
        )
    log.info("")
    log.info("  Promotion criteria:")
    log.info("    1. time_splits   : positive excess return in all splits with n >= %d", PASS_MIN_COUNT)
    log.info("    2. bootstrap     : frac(bootstrap mean > 0) >= %.2f", PASS_BOOTSTRAP_FRAC)
    log.info("    3. loso          : frac(positive after symbol drop) >= %.2f", PASS_LOSO_FRAC)
    log.info("    4. simpler_model : full 18-state best lift >= best simpler-model lift")
    log.info("")
    log.info("  PASS = all 4.  CONDITIONAL = 3/4.  FAIL = <= 2/4.")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_results_csv(output_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal validation of candidate Markov alpha states (Issue #92)."
    )
    parser.add_argument(
        "--cohort", default="core_majors",
        choices=list(COHORTS.keys()) + ["all"],
        help="Cohort(s) to validate against (default: core_majors)",
    )
    parser.add_argument(
        "--min-count", type=int, default=20,
        help="Min events per state-split to count as valid (default: 20)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=N_BOOTSTRAP,
        help=f"Block bootstrap iterations (default: {N_BOOTSTRAP})",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip bootstrap -- faster run, no CIs or frac_positive",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional CSV path for effect-size ranking export",
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

    # Global carry neutralization
    baselines = compute_baselines(all_events)
    log.info("Computed baselines for %d (symbol, direction) groups.", len(baselines))
    excess_events = apply_excess_return(all_events, baselines)
    log.info("Excess return applied to %d events.", len(excess_events))

    cohorts_to_run: dict[str, list[str]] = (
        COHORTS if args.cohort == "all" else {args.cohort: COHORTS[args.cohort]}
    )

    # Multiple-testing context (printed once)
    bonferroni_ctx = compute_bonferroni_context(
        n_candidate_states=len(CANDIDATE_STATES),
        n_cohorts=len(cohorts_to_run),
        n_splits=len(TIME_SPLITS),
    )
    print_bonferroni_context(bonferroni_ctx)

    all_csv_rows: list[dict] = []
    final_verdicts: list[dict] = []

    for cohort_name, cohort_symbols in cohorts_to_run.items():
        present        = {ev["symbol"] for ev in excess_events}
        active_symbols = [s for s in cohort_symbols if s in present]
        missing        = [s for s in cohort_symbols if s not in present]
        if missing:
            log.warning("Cohort '%s': symbols not in DB: %s", cohort_name, missing)
        if not active_symbols:
            log.warning("Cohort '%s': no symbols found -- skipping.", cohort_name)
            continue

        cohort_events = filter_by_symbols(excess_events, active_symbols)

        log.info("")
        log.info("=" * 80)
        log.info("  COHORT: %s  (%d events,  symbols: %s)",
                 cohort_name, len(cohort_events), active_symbols)
        log.info("=" * 80)

        # Effect-size ranking (full cohort, all qualifying states)
        ranking_rows = compute_effect_size_ranking(
            cohort_events,
            min_count=args.min_count,
            n_bootstrap=n_boot,
        )
        print_effect_size_ranking(ranking_rows)

        # Simpler model comparison (full cohort)
        model_results = run_model_comparison(cohort_events, min_count=args.min_count)
        print_model_comparison(model_results)

        # Per-candidate validation
        for candidate_state in CANDIDATE_STATES:
            log.info("")
            log.info("=" * 80)
            log.info("  CANDIDATE STATE: %-35s  [cohort: %s]",
                     candidate_state, cohort_name)
            log.info("=" * 80)

            # Task 2: time splits
            split_results = run_time_split_analysis(
                cohort_events, candidate_state, min_count=args.min_count
            )
            print_time_split_table(candidate_state, split_results)

            # Task 3: block bootstrap CI
            candidate_returns = _state_returns(cohort_events, "state", candidate_state)
            boot_result = block_bootstrap_ci(candidate_returns, n_bootstrap=n_boot)
            print_bootstrap_result(candidate_state, boot_result)

            # Task 4: LOSO
            loso_result = run_loso(
                cohort_events, candidate_state, active_symbols, min_count=args.min_count
            )
            print_loso_table(candidate_state, loso_result)

            # Task 8: pass/fail verdict
            verdict = evaluate_pass_fail(
                candidate_state,
                split_results,
                boot_result,
                loso_result,
                model_results,
                min_count=args.min_count,
            )
            print_verdict(verdict)

            final_verdicts.append(dict(verdict, cohort=cohort_name))

            # CSV rows
            for row in ranking_rows:
                if row["state"] == candidate_state:
                    csv_row = dict(row)
                    csv_row["cohort"]  = cohort_name
                    csv_row["verdict"] = verdict["verdict"]
                    csv_row["n_pass"]  = verdict["n_pass"]
                    all_csv_rows.append(csv_row)

    print_final_summary(final_verdicts)

    if args.output and all_csv_rows:
        write_results_csv(args.output, all_csv_rows)


if __name__ == "__main__":
    main()

"""
scripts/markov_forward_monitor.py

Forward-only monitoring for frozen candidate Markov states -- Issue #93.

Tracks CONTRACTING_TRENDING_UP and EXPANDING_TRENDING_DOWN on post-test
data (2024-01-01 onward) with NO state discovery.

The monitoring window is divided into quarters to show evolution over time.
All analysis uses the global carry baseline established from the full
historical universe.  No new baselines are fitted within the monitoring window.

Run this script periodically as new breakout events are labeled.

Usage
-----
    python scripts/markov_forward_monitor.py
    python scripts/markov_forward_monitor.py --from-date 2024-01-01
    python scripts/markov_forward_monitor.py --output results/forward_monitor.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
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
# Constants — frozen, no modifications permitted
# ---------------------------------------------------------------------------

CANDIDATE_STATES: list[str] = [
    "CONTRACTING_TRENDING_UP",
    "EXPANDING_TRENDING_DOWN",
]

COHORT_SYMBOLS: list[str] = [
    "EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
]

# Monitoring cost assumption (medium level from pre-deployment analysis)
MONITORING_COST: float = 0.07


# ---------------------------------------------------------------------------
# Bucketing (baseline thresholds only -- no threshold tuning in monitoring)
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_SQL_FULL = """
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
            cur.execute(_SQL_FULL)
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
# Carry neutralization (uses full historical baseline -- never re-fit in window)
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
        ne["excess_return"]      = float(ev["realized_return_sd30d"]) - b
        ne["cost_adj_excess"]    = ne["excess_return"] - MONITORING_COST
        out.append(ne)
    return out


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _quarter_label(ts: object) -> str:
    if hasattr(ts, "year") and hasattr(ts, "month"):
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}Q{q}"
    s = str(ts)[:7]   # YYYY-MM
    month = int(s[5:7])
    q = (month - 1) // 3 + 1
    return f"{s[:4]}Q{q}"


def _ts_date_str(ts: object) -> str:
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d")
    return str(ts)[:10]


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def _state_metrics(events: list[dict], state: str, return_field: str = "excess_return") -> dict:
    matching = [ev for ev in events if ev.get("state") == state]
    n        = len(matching)
    if n == 0:
        return {"n": 0, "mean": None, "win_rate": None, "payoff": None,
                "cost_adj_mean": None, "p_gt1": None, "p_lt_neg1": None}

    returns   = [float(ev[return_field]) for ev in matching if ev.get(return_field) is not None]
    cost_adj  = [float(ev["cost_adj_excess"]) for ev in matching if ev.get("cost_adj_excess") is not None]
    wins      = [r for r in returns if r > 0]
    losses    = [r for r in returns if r <= 0]
    avg_win   = sum(wins)   / len(wins)   if wins   else None
    avg_los   = abs(sum(losses) / len(losses)) if losses else None

    return {
        "n":            n,
        "mean":         round(sum(returns) / len(returns), 4) if returns else None,
        "median":       round(_median(returns), 4) if returns else None,
        "win_rate":     round(len(wins) / n, 4),
        "payoff":       round(avg_win / avg_los, 3) if (avg_win and avg_los) else None,
        "cost_adj_mean":round(sum(cost_adj) / len(cost_adj), 4) if cost_adj else None,
        "p_gt1":        round(sum(1 for r in returns if r > 1.0) / n, 4),
        "p_lt_neg1":    round(sum(1 for r in returns if r < -1.0) / n, 4),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_forward_monitor(
    events: list[dict],
    from_date: str,
) -> dict[str, list[dict]]:
    """
    For each candidate state, produce quarterly breakdown from from_date onward.
    Returns {state: [quarterly rows]}.
    No discovery -- only the two frozen states are monitored.
    """
    # Filter to cohort and monitoring window
    monitor_events = [
        ev for ev in events
        if ev["symbol"] in set(COHORT_SYMBOLS)
        and _ts_date_str(ev["event_hour_ts"]) >= from_date
    ]

    # Group by quarter
    quarters: dict[str, list[dict]] = {}
    for ev in monitor_events:
        q = _quarter_label(ev["event_hour_ts"])
        quarters.setdefault(q, []).append(ev)

    sorted_quarters = sorted(quarters.keys())
    results: dict[str, list[dict]] = {s: [] for s in CANDIDATE_STATES}

    for state in CANDIDATE_STATES:
        # Cumulative totals
        cum_returns: list[float] = []
        cum_cost_adj: list[float] = []

        for q in sorted_quarters:
            q_events  = quarters[q]
            q_metrics = _state_metrics(q_events, state)
            m         = _state_metrics(monitor_events[:monitor_events.index(q_events[-1]) + 1]
                                       if q_events else [], state)

            # Gather returns for this quarter only
            q_returns = [float(ev["excess_return"]) for ev in q_events
                         if ev.get("state") == state and ev.get("excess_return") is not None]
            q_cost    = [float(ev["cost_adj_excess"]) for ev in q_events
                         if ev.get("state") == state and ev.get("cost_adj_excess") is not None]
            cum_returns.extend(q_returns)
            cum_cost_adj.extend(q_cost)

            cum_mean     = round(sum(cum_returns)  / len(cum_returns),  4) if cum_returns  else None
            cum_cost_adj_mean = round(sum(cum_cost_adj) / len(cum_cost_adj), 4) if cum_cost_adj else None

            results[state].append({
                "quarter":          q,
                "state":            state,
                "n_quarter":        q_metrics["n"],
                "mean_quarter":     q_metrics["mean"],
                "win_rate_quarter": q_metrics["win_rate"],
                "payoff_quarter":   q_metrics["payoff"],
                "cost_adj_quarter": q_metrics["cost_adj_mean"],
                "p_gt1_quarter":    q_metrics["p_gt1"],
                "p_lt_neg1_quarter":q_metrics["p_lt_neg1"],
                "n_cumulative":     len(cum_returns),
                "mean_cumulative":  cum_mean,
                "cost_adj_cumulative": cum_cost_adj_mean,
            })

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_monitor_report(
    results: dict[str, list[dict]],
    from_date: str,
) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  FORWARD MONITORING REPORT")
    log.info("  Monitoring window: %s onward", from_date)
    log.info("  Cost assumption: %.2f SD30d per trade (medium level)", MONITORING_COST)
    log.info("  Carry baseline: computed from FULL historical universe (frozen, not re-fit)")
    log.info("  States: %s", CANDIDATE_STATES)
    log.info("=" * 80)

    for state, rows in results.items():
        if not rows:
            log.info("  %s: no events in monitoring window", state)
            continue

        total_n    = rows[-1]["n_cumulative"]
        final_mean = rows[-1]["mean_cumulative"]
        final_cost = rows[-1]["cost_adj_cumulative"]

        log.info("")
        log.info("  %s", state)
        log.info("    Total events: %d   Cumulative mean: %s   Cost-adj: %s",
                 total_n,
                 f"{final_mean:+.4f}" if final_mean is not None else "N/A",
                 f"{final_cost:+.4f}" if final_cost is not None else "N/A")
        log.info("")
        log.info("    %-8s  %9s  %9s  %9s  %9s  %9s  %9s  %9s  %10s  %10s",
                 "quarter", "n_qtr", "mean_qtr", "win_qtr", "payoff", "cost_adj",
                 "p_gt+1", "p_lt-1", "n_cum", "mean_cum")
        log.info("    " + "-" * 105)
        for row in rows:
            log.info(
                "    %-8s  %9d  %9s  %9s  %9s  %9s  %9s  %9s  %10d  %10s",
                row["quarter"],
                row["n_quarter"],
                f"{row['mean_quarter']:+.4f}"     if row.get("mean_quarter")     is not None else "N/A",
                f"{row['win_rate_quarter']:.4f}"  if row.get("win_rate_quarter") is not None else "N/A",
                str(row.get("payoff_quarter"))     if row.get("payoff_quarter")   is not None else "N/A",
                f"{row['cost_adj_quarter']:+.4f}" if row.get("cost_adj_quarter") is not None else "N/A",
                f"{row['p_gt1_quarter']:.4f}"     if row.get("p_gt1_quarter")    is not None else "N/A",
                f"{row['p_lt_neg1_quarter']:.4f}" if row.get("p_lt_neg1_quarter") is not None else "N/A",
                row["n_cumulative"],
                f"{row['mean_cumulative']:+.4f}"  if row.get("mean_cumulative")  is not None else "N/A",
            )

        # Simple trend assessment
        positive_qtrs = sum(1 for row in rows if row.get("mean_quarter") is not None and row["mean_quarter"] > 0)
        n_qtrs        = sum(1 for row in rows if row.get("n_quarter", 0) > 0)
        log.info("")
        log.info("    Quarters with positive mean: %d/%d", positive_qtrs, n_qtrs)
        if final_cost is not None:
            if final_cost > 0.05:
                log.info("    Assessment: POSITIVE  -- cumulative cost-adj mean > 0.05 SD30d")
            elif final_cost > 0.0:
                log.info("    Assessment: MARGINAL  -- cumulative cost-adj mean barely positive")
            else:
                log.info("    Assessment: NEGATIVE  -- cumulative cost-adj mean <= 0")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(output_path: str, results: dict[str, list[dict]]) -> None:
    rows = [row for state_rows in results.values() for row in state_rows]
    if not rows:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("Monitor results written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-only monitoring for frozen Markov candidate states (Issue #93)."
    )
    parser.add_argument(
        "--from-date", default="2024-01-01",
        help="Start of monitoring window (YYYY-MM-DD, default: 2024-01-01)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional CSV path for quarterly monitoring export",
    )
    args = parser.parse_args()

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info("Loading FX breakout events ...")
    all_events = load_events()
    log.info("Loaded %d state-labeled events", len(all_events))
    if not all_events:
        log.error("No events found.")
        sys.exit(1)

    # Carry baseline computed from FULL history -- frozen, never re-fit in monitoring window
    baselines = compute_baselines(all_events)
    log.info("Carry baselines computed from %d (symbol, direction) groups (full history).",
             len(baselines))

    excess_events = apply_excess_return(all_events, baselines)
    log.info("Excess return applied to %d events.", len(excess_events))

    log.info("Monitoring window: %s onward (cohort: core_majors)", args.from_date)
    results = run_forward_monitor(excess_events, from_date=args.from_date)

    print_monitor_report(results, from_date=args.from_date)

    if args.output:
        write_csv(args.output, results)


if __name__ == "__main__":
    main()

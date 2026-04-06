"""
scripts/carry_neutralized_analysis.py

Isolate carry/drift effects and re-evaluate Markov state edge — Issue #77.

Tests whether the 18-state breakout entry-quality edge survives after
removing symbol-level directional bias (carry, structural depreciation,
inflation differentials).

Neutralization method
---------------------
For each (symbol, breakout_direction) group:
    mean_return = mean(realized_return_sd30d) over all events in group
    excess_return_sd30d = realized_return_sd30d - mean_return

The excess return is purely in-memory and never written to the database.

Cohorts
-------
core_majors : EUR/USD, GBP/USD, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP
jpy_crosses : USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY, NZD/JPY, CHF/JPY, CAD/JPY
high_drift  : USD/TRY, EUR/TRY

Hypothesis states tested for survival
--------------------------------------
    EXPANDING_TRENDING_UP
    CONTRACTING_TRENDING_UP
    NEUTRAL_TRENDING_UP

No lookahead: the baseline is computed from historical realized outcomes only.
All events used for neutralization have already exited — no future information.

Usage
-----
    python scripts/carry_neutralized_analysis.py
    python scripts/carry_neutralized_analysis.py --cohort core_majors
    python scripts/carry_neutralized_analysis.py --direction UP
    python scripts/carry_neutralized_analysis.py --min-count 20
    python scripts/carry_neutralized_analysis.py --output results/carry_neutralized.csv
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
# Cohort definitions
# ---------------------------------------------------------------------------
COHORTS: dict[str, list[str]] = {
    "core_majors": ["EUR/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"],
    "jpy_crosses": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY", "CHF/JPY", "CAD/JPY"],
    "high_drift":  ["USD/TRY", "EUR/TRY"],
}

# States to explicitly test for survival of neutralization
HYPOTHESIS_STATES: list[str] = [
    "EXPANDING_TRENDING_UP",
    "CONTRACTING_TRENDING_UP",
    "NEUTRAL_TRENDING_UP",
]

# ---------------------------------------------------------------------------
# State space (mirrors markov_state_analysis.py)
# ---------------------------------------------------------------------------
ALL_STATES: list[str] = [
    f"{v}_{e}_{d}"
    for v in ("CONTRACTING", "NEUTRAL", "EXPANDING")
    for e in ("CHOPPY", "MIXED", "TRENDING")
    for d in ("UP", "DOWN")
]
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(ALL_STATES)}
N_STATES: int = len(ALL_STATES)


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
        ev = dict(row)
        vr  = ev.get("vol_ratio_20_100")
        eff = ev.get("efficiency_20")
        d   = ev.get("breakout_direction", "")
        if vr is None or eff is None or d not in ("UP", "DOWN"):
            continue
        ev["state"] = f"{_vol_bucket(float(vr))}_{_eff_bucket(float(eff))}_{d}"
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Symbol+direction baselines
# ---------------------------------------------------------------------------

def compute_symbol_direction_baselines(events: list[dict]) -> dict[tuple[str, str], dict]:
    """
    For each (symbol, breakout_direction) group compute unconditional stats.
    These are the carry/drift baselines to be removed via excess return.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for ev in events:
        key = (ev["symbol"], ev["breakout_direction"])
        groups.setdefault(key, []).append(float(ev["realized_return_sd30d"]))

    baselines: dict[tuple[str, str], dict] = {}
    for key, returns in groups.items():
        n       = len(returns)
        wins    = [r for r in returns if r > 0]
        losses  = [r for r in returns if r <= 0]
        avg_win = sum(wins) / len(wins) if wins else None
        avg_loss = abs(sum(losses) / len(losses)) if losses else None
        payoff  = round(avg_win / avg_loss, 3) if (avg_win and avg_loss) else None
        mean_r  = sum(returns) / n
        baselines[key] = {
            "count":        n,
            "mean_return":  round(mean_r, 4),
            "median_return":round(_median(returns), 4),
            "win_rate":     round(len(wins) / n, 4),
            "payoff_ratio": payoff,
        }
        if n < 10:
            log.warning("Small group (%s, %s): only %d events — baseline may be unreliable.",
                        key[0], key[1], n)
    return baselines


def apply_excess_return(events: list[dict], baselines: dict) -> list[dict]:
    """
    Return a new list with excess_return_sd30d added to each event.
    Events without a baseline key are dropped (logged).
    """
    out: list[dict] = []
    skipped = 0
    for ev in events:
        key = (ev["symbol"], ev["breakout_direction"])
        b = baselines.get(key)
        if b is None:
            skipped += 1
            continue
        new_ev = dict(ev)
        new_ev["excess_return_sd30d"] = float(ev["realized_return_sd30d"]) - b["mean_return"]
        out.append(new_ev)
    if skipped:
        log.warning("Dropped %d events with no baseline key.", skipped)
    return out


# ---------------------------------------------------------------------------
# State stats (generic over return field)
# ---------------------------------------------------------------------------

def compute_state_stats(
    events: list[dict],
    return_field: str = "realized_return_sd30d",
) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {s: [] for s in ALL_STATES}
    for ev in events:
        state = ev.get("state")
        if state in buckets:
            buckets[state].append(ev)

    stats: dict[str, dict] = {}
    for state, rows in buckets.items():
        n = len(rows)
        if n == 0:
            stats[state] = {"count": 0}
            continue
        returns = [float(r[return_field]) for r in rows if r.get(return_field) is not None]
        wins    = [r for r in rows if (r.get("realized_return_sd30d") or 0.0) > 0
                   and r.get("exit_reason") == "trailing_stop"]
        avg_r   = sum(returns) / len(returns) if returns else None
        med_r   = _median(returns) if returns else None
        stats[state] = {
            "count":        n,
            "win_pct":      round(len(wins) / n, 4),
            "avg_return":   round(avg_r, 4) if avg_r is not None else None,
            "median_return":round(med_r, 4) if med_r is not None else None,
        }
    return stats


def compute_baseline(
    events: list[dict],
    return_field: str = "realized_return_sd30d",
) -> dict:
    returns = [float(e[return_field]) for e in events if e.get(return_field) is not None]
    wins    = [e for e in events if (e.get("realized_return_sd30d") or 0.0) > 0
               and e.get("exit_reason") == "trailing_stop"]
    n = len(events)
    if n == 0:
        return {"count": 0, "win_pct": None, "avg_return": None, "median_return": None}
    return {
        "count":         n,
        "win_pct":       round(len(wins) / n, 4),
        "avg_return":    round(sum(returns) / len(returns), 4) if returns else None,
        "median_return": round(_median(returns), 4) if returns else None,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_symbol_baselines(baselines: dict, symbols: Optional[list[str]] = None) -> None:
    keys = sorted(
        (k for k in baselines if (symbols is None or k[0] in symbols)),
        key=lambda k: abs(baselines[k]["mean_return"]),
        reverse=True,
    )
    log.info("  Symbol+direction baselines (sorted by abs drift):")
    log.info(
        "    %-12s %-5s %7s %11s %11s %13s %8s",
        "symbol", "dir", "count", "mean_ret", "median_ret", "win_rate", "payoff",
    )
    log.info("    " + "-" * 74)
    for key in keys:
        b = baselines[key]
        log.info(
            "    %-12s %-5s %7d %11.4f %11.4f %13.4f %8s",
            key[0], key[1],
            b["count"], b["mean_return"], b["median_return"],
            b["win_rate"],
            str(b["payoff_ratio"]) if b["payoff_ratio"] is not None else "N/A",
        )


def print_state_comparison(
    raw_stats: dict[str, dict],
    neu_stats: dict[str, dict],
    raw_baseline: dict,
    neu_baseline: dict,
    min_count: int,
    label: str,
) -> None:
    log.info("")
    log.info("  State comparison — raw vs neutralized  [%s]", label)
    log.info(
        "    %-35s %7s %9s %9s %9s %8s %8s",
        "state", "count",
        "raw_avg", "neu_avg", "delta",
        "raw_win", "neu_win",
    )
    log.info("    " + "-" * 98)

    rb = raw_baseline.get("avg_return") or 0.0
    nb = neu_baseline.get("avg_return") or 0.0

    for state in ALL_STATES:
        rs = raw_stats[state]
        ns = neu_stats[state]
        if rs.get("count", 0) < min_count:
            continue
        raw_avg = rs.get("avg_return") or 0.0
        neu_avg = ns.get("avg_return") or 0.0
        delta   = neu_avg - raw_avg
        log.info(
            "    %-35s %7d %9.4f %9.4f %9.4f %8.4f %8.4f",
            state,
            rs["count"],
            raw_avg,
            neu_avg,
            delta,
            rs.get("win_pct") or 0.0,
            ns.get("win_pct") or 0.0,
        )

    log.info("    " + "-" * 98)
    log.info(
        "    %-35s %7d %9.4f %9.4f %9s %8.4f",
        "BASELINE",
        raw_baseline.get("count", 0),
        rb, nb, "",
        raw_baseline.get("win_pct") or 0.0,
    )


def check_hypothesis_states(
    raw_stats: dict[str, dict],
    neu_stats: dict[str, dict],
    raw_baseline: dict,
    neu_baseline: dict,
    min_count: int,
    label: str,
) -> dict[str, bool]:
    """
    For each hypothesis state, test whether it beats the baseline under both
    raw and neutralized returns. Print a clear SURVIVES / DOES NOT SURVIVE tag.
    Returns a dict of {state: survives_bool}.
    """
    rb = raw_baseline.get("avg_return") or 0.0
    nb = neu_baseline.get("avg_return") or 0.0
    rw = raw_baseline.get("win_pct") or 0.0
    nw = neu_baseline.get("win_pct") or 0.0

    log.info("")
    log.info("  Hypothesis state survival  [%s]:", label)
    log.info(
        "    %-35s %-10s %-10s %-10s %-10s  %s",
        "state", "raw_avg", "neu_avg", "raw_win", "neu_win", "verdict",
    )
    log.info("    " + "-" * 90)

    survival: dict[str, bool] = {}
    for state in HYPOTHESIS_STATES:
        rs = raw_stats.get(state, {})
        ns = neu_stats.get(state, {})
        n  = rs.get("count", 0)
        if n < min_count:
            log.info("    %-35s  (insufficient data: n=%d)", state, n)
            survival[state] = False
            continue

        raw_avg = rs.get("avg_return") or 0.0
        neu_avg = ns.get("avg_return") or 0.0
        raw_win = rs.get("win_pct")    or 0.0
        neu_win = ns.get("win_pct")    or 0.0

        raw_above = (raw_avg > rb) and (raw_win > rw)
        neu_above = (neu_avg > nb) and (neu_win > nw)
        survives  = raw_above and neu_above

        verdict = "SURVIVES" if survives else ("DOES NOT SURVIVE" if raw_above else "NOT ABOVE BASELINE RAW")
        survival[state] = survives

        log.info(
            "    %-35s %+10.4f %+10.4f %10.4f %10.4f  %s",
            state, raw_avg, neu_avg, raw_win, neu_win, verdict,
        )
    return survival


# ---------------------------------------------------------------------------
# Cohort analysis runner
# ---------------------------------------------------------------------------

def run_analysis(
    label: str,
    events: list[dict],
    excess_events: list[dict],
    baselines: dict,
    min_count: int,
    show_transitions: bool,
    symbols: Optional[list[str]] = None,
) -> dict:
    """
    Run full raw + neutralized comparison for a given cohort+direction slice.
    Returns result dict for CSV export.
    """
    n = len(events)
    if n < min_count:
        log.warning("Skipping [%s]: only %d events (< min_count=%d)", label, n, min_count)
        return {}

    log.info("")
    log.info("=" * 80)
    log.info("  COHORT: %s  (%d events)", label, n)
    log.info("=" * 80)

    if symbols:
        print_symbol_baselines(baselines, symbols=symbols)

    raw_stats = compute_state_stats(events, return_field="realized_return_sd30d")
    neu_stats = compute_state_stats(excess_events, return_field="excess_return_sd30d")
    raw_base  = compute_baseline(events, return_field="realized_return_sd30d")
    neu_base  = compute_baseline(excess_events, return_field="excess_return_sd30d")

    print_state_comparison(raw_stats, neu_stats, raw_base, neu_base, min_count, label)
    survival = check_hypothesis_states(raw_stats, neu_stats, raw_base, neu_base, min_count, label)

    # High-EV states (above baseline on both raw AND neutralized)
    rb = raw_base.get("avg_return") or 0.0
    nb = neu_base.get("avg_return") or 0.0
    rw = raw_base.get("win_pct")    or 0.0
    nw = neu_base.get("win_pct")    or 0.0

    robust_states = []
    raw_only_states = []
    for state in ALL_STATES:
        rs = raw_stats[state]
        ns = neu_stats[state]
        if rs.get("count", 0) < min_count:
            continue
        raw_above = (rs.get("avg_return") or 0.0) > rb and (rs.get("win_pct") or 0.0) > rw
        neu_above = (ns.get("avg_return") or 0.0) > nb and (ns.get("win_pct") or 0.0) > nw
        if raw_above and neu_above:
            robust_states.append(state)
        elif raw_above:
            raw_only_states.append(state)

    log.info("")
    log.info("  States above baseline on BOTH raw and neutralized (robust edge):")
    if robust_states:
        for s in robust_states:
            rs = raw_stats[s]
            ns = neu_stats[s]
            log.info("    %-35s  raw_avg=%+.4f  neu_avg=%+.4f  win=%+.4f",
                     s,
                     rs.get("avg_return") or 0.0,
                     ns.get("avg_return") or 0.0,
                     rs.get("win_pct") or 0.0)
    else:
        log.info("    (none)")

    log.info("  States above baseline on raw only (carry-driven, disappear after neutralization):")
    if raw_only_states:
        for s in raw_only_states:
            rs = raw_stats[s]
            ns = neu_stats[s]
            log.info("    %-35s  raw_avg=%+.4f  neu_avg=%+.4f",
                     s,
                     rs.get("avg_return") or 0.0,
                     ns.get("avg_return") or 0.0)
    else:
        log.info("    (none)")

    # Build result dict for CSV
    results = {}
    for state in ALL_STATES:
        rs = raw_stats[state]
        ns = neu_stats[state]
        results[state] = {
            "label":           label,
            "state":           state,
            "count":           rs.get("count", 0),
            "raw_avg_return":  rs.get("avg_return"),
            "raw_win_pct":     rs.get("win_pct"),
            "neu_avg_return":  ns.get("avg_return"),
            "neu_win_pct":     ns.get("win_pct"),
            "survives":        survival.get(state),
        }
    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(output_path: str, all_results: list[dict]) -> None:
    fields = [
        "label", "state", "count",
        "raw_avg_return", "raw_win_pct",
        "neu_avg_return", "neu_win_pct",
        "survives",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    log.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# Cross-cohort summary
# ---------------------------------------------------------------------------

def print_cross_cohort_summary(
    cohort_results: dict[str, dict],
    min_count: int,
) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("  CROSS-COHORT SUMMARY — Hypothesis state survival")
    log.info("=" * 80)
    col_labels = list(cohort_results.keys())
    col_w = 12

    header = f"  {'state':<35}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in col_labels)
    log.info(header)
    log.info("  " + "-" * (35 + col_w * len(col_labels)))

    for state in HYPOTHESIS_STATES:
        row = f"  {state:<35}"
        for lbl in col_labels:
            result = cohort_results[lbl].get(state)
            if result is None or result.get("count", 0) < min_count:
                cell = "N/A"
            elif result.get("survives") is True:
                cell = "SURVIVES"
            elif result.get("survives") is False:
                cell = "NO"
            else:
                cell = "-"
            row += f"{cell:>{col_w}}"
        log.info(row)

    log.info("")
    log.info("  SURVIVES = above baseline on avg_return AND win_pct under both raw and neutralized.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Carry-neutralized Markov state analysis (Issue #77)."
    )
    parser.add_argument("--cohort",    default="all",
                        choices=list(COHORTS.keys()) + ["all"],
                        help="Cohort to analyse (default: all)")
    parser.add_argument("--direction", default="BOTH",
                        choices=["UP", "DOWN", "BOTH"],
                        help="Direction filter (default: BOTH)")
    parser.add_argument("--min-count", type=int, default=30,
                        help="Min events per state to include (default 30)")
    parser.add_argument("--no-transitions", action="store_true",
                        help="Skip transition matrix printing")
    parser.add_argument("--output",    default=None,
                        help="Optional CSV path for full results export")
    args = parser.parse_args()

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    log.info("Loading FX breakout events ...")
    all_events = load_events()
    log.info("Loaded %d state-labeled events", len(all_events))

    if not all_events:
        log.error("No events found. Run backfill_breakout_events.py first.")
        sys.exit(1)

    # Compute full-universe symbol+direction baselines
    baselines = compute_symbol_direction_baselines(all_events)
    log.info("Computed baselines for %d (symbol, direction) groups.", len(baselines))

    # Apply excess return over full universe (baseline is global, not cohort-local)
    excess_all = apply_excess_return(all_events, baselines)
    log.info("Excess return applied to %d events.", len(excess_all))

    # Identify high-drift symbols (abs mean_return > 0.5 SD30d in either direction)
    log.info("")
    log.info("High-drift symbol+direction pairs (|mean_return| > 0.5 SD30d):")
    flagged = {k: v for k, v in baselines.items() if abs(v["mean_return"]) > 0.5}
    if flagged:
        for k, v in sorted(flagged.items(), key=lambda x: abs(x[1]["mean_return"]), reverse=True):
            log.info("  %-12s %-5s  mean=%-8.4f  win=%.4f  n=%d",
                     k[0], k[1], v["mean_return"], v["win_rate"], v["count"])
    else:
        log.info("  (none above threshold)")

    # Determine which cohorts and directions to run
    cohorts_to_run: dict[str, list[str]] = (
        COHORTS if args.cohort == "all" else {args.cohort: COHORTS[args.cohort]}
    )
    directions: list[str] = (
        ["UP", "DOWN", "BOTH"] if args.direction == "BOTH" else [args.direction]
    )

    all_results: list[dict] = []
    cohort_summary_results: dict[str, dict] = {}

    # Run full-universe analysis first
    log.info("")
    log.info("Running full-universe analysis ...")
    for direction in directions:
        evs   = [e for e in all_events  if direction == "BOTH" or e["breakout_direction"] == direction]
        ex_evs= [e for e in excess_all  if direction == "BOTH" or e["breakout_direction"] == direction]
        label = f"ALL_FX | {direction}"
        res = run_analysis(label, evs, ex_evs, baselines,
                           args.min_count, not args.no_transitions, symbols=None)
        for state_res in res.values():
            all_results.append(state_res)
        cohort_summary_results[label] = res

    # Per-cohort analysis
    for cohort_name, symbols in cohorts_to_run.items():
        # Check which symbols are actually present in the loaded data
        present = {e["symbol"] for e in all_events}
        missing = [s for s in symbols if s not in present]
        if missing:
            log.warning("Cohort '%s': symbols not in DB: %s", cohort_name, missing)
        active_symbols = [s for s in symbols if s in present]
        if not active_symbols:
            log.warning("Cohort '%s': no symbols found — skipping.", cohort_name)
            continue

        for direction in directions:
            evs    = [e for e in all_events if e["symbol"] in active_symbols
                      and (direction == "BOTH" or e["breakout_direction"] == direction)]
            ex_evs = [e for e in excess_all if e["symbol"] in active_symbols
                      and (direction == "BOTH" or e["breakout_direction"] == direction)]
            label  = f"{cohort_name} | {direction}"
            res = run_analysis(label, evs, ex_evs, baselines,
                               args.min_count, not args.no_transitions, symbols=active_symbols)
            for state_res in res.values():
                all_results.append(state_res)
            cohort_summary_results[label] = res

    # Cross-cohort summary
    if len(cohort_summary_results) > 1:
        print_cross_cohort_summary(cohort_summary_results, args.min_count)

    # CSV export
    if args.output and all_results:
        write_csv(args.output, all_results)


if __name__ == "__main__":
    main()

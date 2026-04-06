"""
scripts/markov_state_analysis.py

Markov state model for breakout entry quality — Issue #76.

Builds an 18-state discrete Markov model over features.breakout_events,
computes per-state outcome statistics, and prints a transition matrix with
Laplace smoothing.

State space (3 x 3 x 2 = 18 states):

    Dimension 1 — Volatility regime (vol_ratio_20_100):
        contracting  : vol_ratio < 0.8
        neutral      : 0.8 <= vol_ratio <= 1.2
        expanding    : vol_ratio > 1.2

    Dimension 2 — Trend quality (efficiency_20):
        choppy       : efficiency < 0.3
        mixed        : 0.3 <= efficiency <= 0.6
        trending     : efficiency > 0.6

    Dimension 3 — Breakout direction:
        UP / DOWN

State label format:  {vol_bucket}_{eff_bucket}_{direction}
Example:             EXPANDING_TRENDING_UP

Input:
    features.breakout_events   (TimescaleDB)

Filters applied:
    - FX symbols only (JOIN market_data.symbols WHERE type = 'forex')
    - exit_reason IS NOT NULL  (fully labeled trades only)
    - entry_range_sd_30d > 0   (valid SD30d risk unit)

Transitions:
    Consecutive breakout events per symbol ordered by event_hour_ts.
    Laplace smoothing: add 1 count to every cell of the raw count matrix
    before normalising (add-1 / add-k smoothing across 18 states).

Outcome statistics per state:
    count                    : number of events in state
    win_pct                  : fraction where exit_reason = 'trailing_stop'
                               AND realized_return_sd30d > 0
    avg_return_sd30d         : mean realized_return_sd30d
    median_return_sd30d      : median realized_return_sd30d
    avg_favorable_excursion  : mean max_favorable_excursion_sd30d
    avg_adverse_excursion    : mean max_adverse_excursion_sd30d
    avg_bars_held            : mean bars_held

Usage
-----
    python scripts/markov_state_analysis.py
    python scripts/markov_state_analysis.py --symbol "USD/JPY"
    python scripts/markov_state_analysis.py --min-count 30
    python scripts/markov_state_analysis.py --no-transitions
    python scripts/markov_state_analysis.py --output results/markov_states.csv
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
# State buckets
# ---------------------------------------------------------------------------

VOL_BUCKETS = [
    ("contracting", None,  0.8),
    ("neutral",      0.8,  1.2),
    ("expanding",    1.2, None),
]

EFF_BUCKETS = [
    ("choppy",   None,  0.3),
    ("mixed",     0.3,  0.6),
    ("trending",  0.6, None),
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


# All 18 canonical state labels (order is deterministic for matrix indexing)
ALL_STATES: list[str] = [
    f"{v}_{e}_{d}"
    for v in ("CONTRACTING", "NEUTRAL", "EXPANDING")
    for e in ("CHOPPY", "MIXED", "TRENDING")
    for d in ("UP", "DOWN")
]

STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(ALL_STATES)}
N_STATES: int = len(ALL_STATES)


# ---------------------------------------------------------------------------
# SQL: fetch labeled events with state features
# ---------------------------------------------------------------------------

_EVENTS_SQL = """
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
WHERE {type_filter}
  AND be.exit_reason IS NOT NULL
  AND be.entry_range_sd_30d > 0
  AND be.vol_ratio_20_100   IS NOT NULL
  AND be.efficiency_20      IS NOT NULL
  AND be.realized_return_sd30d IS NOT NULL
{symbol_filter}
ORDER BY be.symbol, be.event_hour_ts
"""


# ---------------------------------------------------------------------------
# Markov transition matrix
# ---------------------------------------------------------------------------

def build_transition_matrix(
    events: list[dict],
    laplace: float = 1.0,
) -> list[list[float]]:
    """
    Count state->state transitions for consecutive events within a symbol,
    apply Laplace smoothing, return row-normalised probability matrix.

    Row i  = current state
    Col j  = next state
    Value  = P(next=j | current=i)
    """
    counts: list[list[float]] = [
        [laplace] * N_STATES for _ in range(N_STATES)
    ]

    prev_symbol: Optional[str] = None
    prev_state: Optional[int] = None

    for ev in events:
        cur_state = STATE_INDEX.get(ev["state"])
        if cur_state is None:
            prev_symbol = None
            prev_state = None
            continue

        if ev["symbol"] == prev_symbol and prev_state is not None:
            counts[prev_state][cur_state] += 1.0

        prev_symbol = ev["symbol"]
        prev_state  = cur_state

    # Row-normalise
    probs: list[list[float]] = []
    for row in counts:
        row_sum = sum(row)
        probs.append([c / row_sum for c in row])

    return probs


# ---------------------------------------------------------------------------
# Per-state outcome statistics
# ---------------------------------------------------------------------------

def compute_state_stats(events: list[dict]) -> dict[str, dict]:
    """
    Aggregate outcome metrics per state label.

    A trade is a "win" when exit_reason = 'trailing_stop' AND
    realized_return_sd30d > 0 (i.e. the stop was hit in profit).
    """
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

        returns      = [r["realized_return_sd30d"] for r in rows if r["realized_return_sd30d"] is not None]
        fav_exc      = [r["max_favorable_excursion_sd30d"] for r in rows if r["max_favorable_excursion_sd30d"] is not None]
        adv_exc      = [r["max_adverse_excursion_sd30d"] for r in rows if r["max_adverse_excursion_sd30d"] is not None]
        bars         = [r["bars_held"] for r in rows if r["bars_held"] is not None]
        wins         = [r for r in rows if r["exit_reason"] == "trailing_stop" and (r["realized_return_sd30d"] or 0.0) > 0]

        avg_ret      = sum(returns) / len(returns) if returns else None
        median_ret   = _median(returns) if returns else None
        avg_fav      = sum(fav_exc) / len(fav_exc) if fav_exc else None
        avg_adv      = sum(adv_exc) / len(adv_exc) if adv_exc else None
        avg_bars     = sum(bars) / len(bars) if bars else None
        win_pct      = len(wins) / n

        stats[state] = {
            "count":              n,
            "win_pct":            round(win_pct, 4),
            "avg_return_sd30d":   round(avg_ret, 4) if avg_ret is not None else None,
            "median_return_sd30d":round(median_ret, 4) if median_ret is not None else None,
            "avg_favorable_exc":  round(avg_fav, 4) if avg_fav is not None else None,
            "avg_adverse_exc":    round(avg_adv, 4) if avg_adv is not None else None,
            "avg_bars_held":      round(avg_bars, 1) if avg_bars is not None else None,
        }

    return stats


def _median(values: list[float]) -> float:
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    if n % 2 == 1:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


# ---------------------------------------------------------------------------
# Baseline (pooled over all states)
# ---------------------------------------------------------------------------

def compute_baseline(events: list[dict]) -> dict:
    returns = [e["realized_return_sd30d"] for e in events if e["realized_return_sd30d"] is not None]
    wins    = [e for e in events if e["exit_reason"] == "trailing_stop" and (e["realized_return_sd30d"] or 0.0) > 0]
    n       = len(events)
    if n == 0:
        return {"count": 0}
    return {
        "count":               n,
        "win_pct":             round(len(wins) / n, 4),
        "avg_return_sd30d":    round(sum(returns) / len(returns), 4) if returns else None,
        "median_return_sd30d": round(_median(returns), 4) if returns else None,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_state_stats(stats: dict[str, dict], baseline: dict, min_count: int) -> None:
    log.info("")
    log.info("Per-state outcome statistics (min_count=%d)", min_count)
    log.info(
        "  %-35s  %7s  %8s  %11s  %13s  %12s  %12s  %11s",
        "state", "count", "win_pct",
        "avg_ret_sd", "med_ret_sd", "avg_fav", "avg_adv", "avg_bars",
    )
    log.info("  " + "-" * 120)

    for state in ALL_STATES:
        s = stats[state]
        if s.get("count", 0) < min_count:
            continue
        log.info(
            "  %-35s  %7d  %8.4f  %11.4f  %13.4f  %12.4f  %12.4f  %11.1f",
            state,
            s["count"],
            s["win_pct"],
            s["avg_return_sd30d"] or 0.0,
            s["median_return_sd30d"] or 0.0,
            s["avg_favorable_exc"] or 0.0,
            s["avg_adverse_exc"] or 0.0,
            s["avg_bars_held"] or 0.0,
        )

    log.info("  " + "-" * 120)
    log.info(
        "  %-35s  %7d  %8.4f  %11.4f  %13.4f",
        "BASELINE (all states)",
        baseline["count"],
        baseline["win_pct"],
        baseline["avg_return_sd30d"] or 0.0,
        baseline["median_return_sd30d"] or 0.0,
    )


def print_transition_matrix(matrix: list[list[float]], min_count: int, stats: dict[str, dict]) -> None:
    """Print a condensed transition matrix for states with sufficient count."""
    active = [s for s in ALL_STATES if stats[s].get("count", 0) >= min_count]
    if not active:
        log.info("No states meet min_count threshold for transition matrix.")
        return

    col_w = 8
    header_indent = " " * 38
    log.info("")
    log.info("Transition matrix (row=current state, col=next state)")
    log.info("(Laplace smoothed, row-normalised probabilities)")
    log.info("")

    # Column headers (abbreviated)
    abbrev = {s: s[:col_w] for s in active}
    header = header_indent + "  ".join(f"{abbrev[s]:>{col_w}}" for s in active)
    log.info(header)
    log.info(" " * 38 + "-" * (len(active) * (col_w + 2)))

    for row_state in active:
        ri = STATE_INDEX[row_state]
        row_vals = "  ".join(
            f"{matrix[ri][STATE_INDEX[col_state]]:>{col_w}.4f}"
            for col_state in active
        )
        log.info("  %-35s  %s", row_state, row_vals)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(output_path: str, stats: dict[str, dict]) -> None:
    fields = [
        "state", "count", "win_pct",
        "avg_return_sd30d", "median_return_sd30d",
        "avg_favorable_exc", "avg_adverse_exc", "avg_bars_held",
    ]
    rows = []
    for state in ALL_STATES:
        s = stats[state]
        rows.append({
            "state":               state,
            "count":               s.get("count", 0),
            "win_pct":             s.get("win_pct"),
            "avg_return_sd30d":    s.get("avg_return_sd30d"),
            "median_return_sd30d": s.get("median_return_sd30d"),
            "avg_favorable_exc":   s.get("avg_favorable_exc"),
            "avg_adverse_exc":     s.get("avg_adverse_exc"),
            "avg_bars_held":       s.get("avg_bars_held"),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info("State statistics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Markov state analysis of FX breakout events (Issue #76)."
    )
    parser.add_argument("--symbol",         default=None,  help='Single symbol, e.g. "USD/JPY"')
    parser.add_argument("--non-fx",         action="store_true",
                        help="Analyse non-FX instruments instead of FX symbols")
    parser.add_argument("--min-count",      type=int, default=30,
                        help="Minimum events per state to include in output (default 30)")
    parser.add_argument("--no-transitions", action="store_true",
                        help="Skip printing the transition matrix")
    parser.add_argument("--output",         default=None,
                        help="Optional CSV path for state statistics export")
    args = parser.parse_args()

    # Build filters
    symbol_filter = ""
    if args.symbol is not None:
        if len(args.symbol) > 20 or not all(c.isalnum() or c in ("/_-. ") for c in args.symbol):
            raise ValueError(f"Symbol looks unsafe: {args.symbol!r}")
        symbol_filter = f"AND be.symbol = '{args.symbol}'"

    type_filter = "s.type != 'forex'" if args.non_fx else "s.type = 'forex'"
    sql = _EVENTS_SQL.format(symbol_filter=symbol_filter, type_filter=type_filter)

    log.info("Connecting to %s", TIMESCALE_DSN or f"{PGHOST}:{PGPORT}/{PGDATABASE}")
    scope = args.symbol or ("all non-FX instruments" if args.non_fx else "all FX symbols")
    log.info("Loading breakout events for %s", scope)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            raw_rows = cur.fetchall()
    finally:
        conn.close()

    log.info("Loaded %d fully-labeled events", len(raw_rows))
    if not raw_rows:
        log.warning("No events found — check that backfill_breakout_events.py has been run.")
        sys.exit(1)

    # Assign state labels
    events: list[dict] = []
    for row in raw_rows:
        ev = dict(row)
        vr  = ev.get("vol_ratio_20_100")
        eff = ev.get("efficiency_20")
        d   = ev.get("breakout_direction", "")
        if vr is None or eff is None or d not in ("UP", "DOWN"):
            continue
        ev["state"] = f"{_vol_bucket(float(vr))}_{_eff_bucket(float(eff))}_{d}"
        events.append(ev)

    log.info("State-labeled events: %d  (skipped %d with NULL features)",
             len(events), len(raw_rows) - len(events))

    # Baseline
    baseline = compute_baseline(events)
    log.info(
        "Baseline: count=%d  win_pct=%.4f  avg_return=%.4f  median_return=%.4f",
        baseline["count"],
        baseline["win_pct"],
        baseline["avg_return_sd30d"] or 0.0,
        baseline["median_return_sd30d"] or 0.0,
    )

    # State distribution
    state_counts: dict[str, int] = {}
    for ev in events:
        state_counts[ev["state"]] = state_counts.get(ev["state"], 0) + 1

    log.info("")
    log.info("State distribution:")
    for state in ALL_STATES:
        cnt = state_counts.get(state, 0)
        pct = cnt / len(events) * 100 if events else 0.0
        log.info("  %-35s  %6d  (%5.2f%%)", state, cnt, pct)

    # Per-state outcome statistics
    stats = compute_state_stats(events)
    print_state_stats(stats, baseline, args.min_count)

    # Transition matrix
    if not args.no_transitions:
        matrix = build_transition_matrix(events)
        print_transition_matrix(matrix, args.min_count, stats)

    # Identify high-EV states (above-baseline avg_return and win_pct)
    baseline_ret  = baseline.get("avg_return_sd30d") or 0.0
    baseline_win  = baseline.get("win_pct") or 0.0

    log.info("")
    log.info("High-EV states (avg_return > baseline AND win_pct > baseline, count >= %d):", args.min_count)
    found_any = False
    for state in ALL_STATES:
        s = stats[state]
        if s.get("count", 0) < args.min_count:
            continue
        ret = s.get("avg_return_sd30d") or 0.0
        win = s.get("win_pct") or 0.0
        if ret > baseline_ret and win > baseline_win:
            log.info(
                "  %-35s  count=%d  win_pct=%.4f  avg_return=%.4f",
                state, s["count"], win, ret,
            )
            found_any = True
    if not found_any:
        log.info("  (none above baseline on both metrics)")

    log.info("")
    log.info("Low-EV states (avg_return < baseline AND win_pct < baseline, count >= %d):", args.min_count)
    for state in ALL_STATES:
        s = stats[state]
        if s.get("count", 0) < args.min_count:
            continue
        ret = s.get("avg_return_sd30d") or 0.0
        win = s.get("win_pct") or 0.0
        if ret < baseline_ret and win < baseline_win:
            log.info(
                "  %-35s  count=%d  win_pct=%.4f  avg_return=%.4f",
                state, s["count"], win, ret,
            )

    # Optional CSV export
    if args.output:
        write_csv(args.output, stats)


if __name__ == "__main__":
    main()

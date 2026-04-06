"""
CTU portfolio simulation.

Simulates equal-weight and capped portfolio performance across splits.
Writes results to features.ctu_portfolio_simulation_results.

Usage:
    python scripts/ctu/markov_portfolio_simulator.py
"""
import sys
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from scripts.shared.db import get_engine
from scripts.shared.stats import sharpe, max_drawdown
from scripts.ctu.constants import (
    CTU_EVENT_QUERY,
    MEDIUM_COST,
    MAX_PER_SYMBOL,
    MAX_PER_STATE,
    MAX_TOTAL,
    TRAIN_CUTOFF_YEAR,
    VALIDATION_CUTOFF_YEAR,
    THRESHOLD_SETS,
    CTU_SYMBOLS,
)

CANDIDATE_STATE = "CONTRACTING_TRENDING_UP"


def load_events(engine) -> pd.DataFrame:
    ts = THRESHOLD_SETS["baseline"]
    params = {"vol_lo": ts["vol_lo"], "eff_hi": ts["eff_hi"]}
    with engine.connect() as conn:
        df = pd.read_sql(text(CTU_EVENT_QUERY), conn, params=params)
    df["year"] = df["year"].astype(int)
    df["excess_return"] = df["realized_return_sd30d"] - MEDIUM_COST
    return df


def label_split(year: int) -> str:
    if year < TRAIN_CUTOFF_YEAR:
        return "train"
    elif year < VALIDATION_CUTOFF_YEAR:
        return "validation"
    else:
        return "forward"


def simulate_equal_weight(df: pd.DataFrame, split: str) -> dict:
    if split != "all":
        subset = df[df["split"] == split]
    else:
        subset = df
    if len(subset) < 5:
        return {}

    # Simple equal-weight: average across all events
    r = subset.groupby("event_hour_ts")["excess_return"].mean().values
    cum = np.cumsum(r)
    concentration = subset.groupby("symbol")["excess_return"].count()
    hhi = ((concentration / concentration.sum()) ** 2).sum()

    verdict = "STOP"
    if r.mean() > 0 and sharpe(r) > 0.3:
        verdict = "PASS"
    elif r.mean() > 0:
        verdict = "MONITOR"

    return {
        "portfolio_name": "equal_weight",
        "split_name": split,
        "n_trades": int(len(r)),
        "mean_return": float(r.mean()),
        "sharpe": float(sharpe(r)),
        "max_drawdown": float(max_drawdown(cum)),
        "win_rate": float((r > 0).mean()),
        "payoff": float(r[r > 0].mean() / abs(r[r < 0].mean())) if (r < 0).any() and (r > 0).any() else None,
        "p_lt_neg1": float((r < -1).mean()),
        "p_gt_pos1": float((r > 1).mean()),
        "concentration_metric": float(hhi),
        "verdict": verdict,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    df = load_events(engine)
    df["split"] = df["year"].apply(label_split)

    results = []
    print("\n=== CTU Portfolio Simulation ===")

    for split in ["train", "validation", "forward", "all"]:
        res = simulate_equal_weight(df, split)
        if not res:
            continue
        results.append(res)
        print(f"  [{res['portfolio_name']}] [{split:12s}]  n={res['n_trades']}  "
              f"mean={res['mean_return']:+.4f}  sharpe={res['sharpe']:.3f}  "
              f"mdd={res['max_drawdown']:.3f}  hhi={res['concentration_metric']:.3f}  "
              f"verdict={res['verdict']}")

    if not args.no_write and results:
        pd.DataFrame(results).to_sql(
            "ctu_portfolio_simulation_results", engine, schema="features",
            if_exists="append", index=False
        )
        print(f"\nWrote {len(results)} rows to features.ctu_portfolio_simulation_results")


if __name__ == "__main__":
    main()

"""
CTU forward monitoring snapshot.

For use only after CTU is frozen. Currently logs forward period performance
snapshots to features.ctu_forward_monitor.

Usage:
    python scripts/ctu/markov_forward_monitor.py
"""
import sys
from datetime import datetime, date, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from scripts.shared.db import get_engine
from scripts.shared.stats import sharpe
from scripts.ctu.constants import (
    CTU_EVENT_QUERY,
    MEDIUM_COST,
    VALIDATION_CUTOFF_YEAR,
    THRESHOLD_SETS,
)

CANDIDATE_STATE = "CONTRACTING_TRENDING_UP"


def load_forward_events(engine) -> pd.DataFrame:
    ts = THRESHOLD_SETS["baseline"]
    params = {"vol_lo": ts["vol_lo"], "eff_hi": ts["eff_hi"]}
    with engine.connect() as conn:
        df = pd.read_sql(text(CTU_EVENT_QUERY), conn, params=params)
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= VALIDATION_CUTOFF_YEAR].copy()
    df["excess_return"] = df["realized_return_sd30d"] - MEDIUM_COST
    return df


def snapshot(df: pd.DataFrame) -> list[dict]:
    snapshot_date = date.today()
    rows = []

    # Overall forward
    r = df["excess_return"].values
    if len(r) >= 5:
        rows.append({
            "snapshot_date": snapshot_date,
            "candidate_state": CANDIDATE_STATE,
            "symbol": None,
            "split_name": "forward",
            "n_events": int(len(r)),
            "mean_return": float(r.mean()),
            "sharpe": float(sharpe(r)),
            "cumulative_return": float(r.sum()),
            "win_rate": float((r > 0).mean()),
            "verdict": "MONITOR" if r.mean() > 0 else "WATCH",
            "note": "forward monitoring snapshot",
            "created_at": datetime.now(timezone.utc),
        })

    # Per-symbol
    for sym, grp in df.groupby("symbol"):
        r_sym = grp["excess_return"].values
        if len(r_sym) < 3:
            continue
        rows.append({
            "snapshot_date": snapshot_date,
            "candidate_state": CANDIDATE_STATE,
            "symbol": sym,
            "split_name": "forward",
            "n_events": int(len(r_sym)),
            "mean_return": float(r_sym.mean()),
            "sharpe": float(sharpe(r_sym)),
            "cumulative_return": float(r_sym.sum()),
            "win_rate": float((r_sym > 0).mean()),
            "verdict": "MONITOR" if r_sym.mean() > 0 else "WATCH",
            "note": None,
            "created_at": datetime.now(timezone.utc),
        })

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    df = load_forward_events(engine)

    print(f"\n=== CTU Forward Monitor Snapshot ===")
    print(f"Forward events loaded: {len(df)}")

    rows = snapshot(df)
    for row in rows:
        sym_label = row["symbol"] or "ALL"
        print(f"  {sym_label:10s}  n={row['n_events']}  mean={row['mean_return']:+.4f}  "
              f"sharpe={row['sharpe']:.3f}  cum={row['cumulative_return']:+.4f}  "
              f"verdict={row['verdict']}")

    if not args.no_write:
        pd.DataFrame(rows).to_sql(
            "ctu_forward_monitor", engine, schema="features",
            if_exists="append", index=False
        )
        print(f"\nWrote {len(rows)} rows to features.ctu_forward_monitor")


if __name__ == "__main__":
    main()

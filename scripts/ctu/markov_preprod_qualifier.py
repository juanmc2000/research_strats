"""
CTU per-symbol pre-production qualification.

Qualifies each symbol as KEEP / MONITOR / EXCLUDE based on pooled and forward performance.
Writes results to features.ctu_symbol_qualification.

Usage:
    python scripts/ctu/markov_preprod_qualifier.py
"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from scripts.shared.db import get_engine
from scripts.shared.stats import sharpe
from scripts.ctu.constants import (
    CTU_EVENT_QUERY,
    MEDIUM_COST,
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


def qualify_symbol(sym_df: pd.DataFrame, symbol: str) -> dict:
    pooled = sym_df["excess_return"].values
    forward_mask = sym_df["year"] >= VALIDATION_CUTOFF_YEAR
    forward = sym_df.loc[forward_mask, "excess_return"].values

    pooled_mean = pooled.mean() if len(pooled) > 0 else None
    pooled_sharpe = sharpe(pooled) if len(pooled) > 1 else None
    forward_mean = forward.mean() if len(forward) > 0 else None
    forward_sharpe = sharpe(forward) if len(forward) > 1 else None
    cost_adj_mean = float(pooled_mean) if pooled_mean is not None else None
    win_rate = float((pooled > 0).mean()) if len(pooled) > 0 else None
    median_return = float(np.median(pooled)) if len(pooled) > 0 else None

    status = "EXCLUDE"
    reason = "insufficient data"
    if len(pooled) >= 10 and pooled_mean is not None:
        if pooled_mean > 0 and (forward_mean is None or forward_mean > -0.02):
            if pooled_sharpe is not None and pooled_sharpe > 0.3:
                status = "KEEP"
                reason = "positive pooled Sharpe and acceptable forward"
            else:
                status = "MONITOR"
                reason = "positive mean but low Sharpe"
        else:
            status = "EXCLUDE"
            reason = "negative pooled mean or significant forward degradation"

    return {
        "symbol": symbol,
        "status": status,
        "pooled_mean": float(pooled_mean) if pooled_mean is not None else None,
        "pooled_sharpe": float(pooled_sharpe) if pooled_sharpe is not None else None,
        "forward_mean": float(forward_mean) if forward_mean is not None else None,
        "forward_sharpe": float(forward_sharpe) if forward_sharpe is not None else None,
        "cost_adjusted_mean": cost_adj_mean,
        "win_rate": win_rate,
        "median_return": median_return,
        "ruling_reason": reason,
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

    print(f"\n=== CTU Symbol Qualification ===")
    results = []
    for sym in CTU_SYMBOLS:
        sym_df = df[df["symbol"] == sym]
        result = qualify_symbol(sym_df, sym)
        results.append(result)
        print(f"  {sym:10s}  n={len(sym_df):4d}  pooled_mean={result['pooled_mean'] or 'N/A':>8}  "
              f"fwd_mean={result['forward_mean'] or 'N/A':>8}  "
              f"status={result['status']:8s}  reason={result['ruling_reason']}")

    if not args.no_write:
        pd.DataFrame(results).to_sql(
            "ctu_symbol_qualification", engine, schema="features",
            if_exists="append", index=False
        )
        print(f"\nWrote {len(results)} rows to features.ctu_symbol_qualification")


if __name__ == "__main__":
    main()

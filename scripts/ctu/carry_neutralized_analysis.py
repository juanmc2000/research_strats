"""
CTU carry-neutralized analysis.

Loads CTU events and checks whether the edge persists after subtracting
a carry/drift baseline. Exploratory only — no production claims.

Usage:
    python scripts/ctu/carry_neutralized_analysis.py
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from scripts.shared.db import get_engine
from scripts.shared.stats import bootstrap_ci, loso_mean, sharpe
from scripts.ctu.constants import (
    CTU_EVENT_QUERY,
    MEDIUM_COST,
    N_BOOTSTRAP,
    BLOCK_SIZE,
    TRAIN_CUTOFF_YEAR,
    VALIDATION_CUTOFF_YEAR,
    THRESHOLD_SETS,
)

CANDIDATE_STATE = "CONTRACTING_TRENDING_UP"


def load_events(engine, vol_lo: float, eff_hi: float) -> pd.DataFrame:
    params = {"vol_lo": vol_lo, "eff_hi": eff_hi}
    with engine.connect() as conn:
        df = pd.read_sql(text(CTU_EVENT_QUERY), conn, params=params)
    df["excess_return"] = df["realized_return_sd30d"] - MEDIUM_COST
    df["year"] = df["year"].astype(int)
    return df


def label_split(year: int) -> str:
    if year < TRAIN_CUTOFF_YEAR:
        return "train"
    elif year < VALIDATION_CUTOFF_YEAR:
        return "validation"
    else:
        return "forward"


def analyze(df: pd.DataFrame, split_name: str) -> dict:
    subset = df[df["split"] == split_name]
    if len(subset) == 0:
        return {}
    r = subset["excess_return"].values
    ci_low, ci_high, frac_pos = bootstrap_ci(r, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE)
    loso_fp, loso_th = loso_mean(r, subset["symbol"].values)
    return {
        "split": split_name,
        "n": len(r),
        "mean": r.mean(),
        "median": np.median(r),
        "win_rate": (r > 0).mean(),
        "sharpe": sharpe(r),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "frac_pos_boot": frac_pos,
        "loso_frac_pos": loso_fp,
        "loso_frac_top_half": loso_th,
    }


def main():
    engine = get_engine()
    ts = THRESHOLD_SETS["baseline"]
    df = load_events(engine, ts["vol_lo"], ts["eff_hi"])
    df["split"] = df["year"].apply(label_split)

    print(f"\n=== CTU Carry-Neutralized Analysis ===")
    print(f"State: {CANDIDATE_STATE}")
    print(f"Cost assumption: MEDIUM_COST={MEDIUM_COST}")
    print(f"Total events: {len(df)}")
    print()

    for split in ["train", "validation", "forward"]:
        res = analyze(df, split)
        if not res:
            print(f"[{split}] No events.")
            continue
        print(f"[{split}]  n={res['n']}  mean={res['mean']:.4f}  median={res['median']:.4f}  "
              f"win_rate={res['win_rate']:.3f}  sharpe={res['sharpe']:.3f}")
        print(f"         CI=[{res['ci_low']:.4f}, {res['ci_high']:.4f}]  "
              f"frac_pos_boot={res['frac_pos_boot']:.3f}")
        print(f"         loso_frac_pos={res['loso_frac_pos']:.3f}  "
              f"loso_frac_top_half={res['loso_frac_top_half']:.3f}")
        print()

    # Symbol breakdown
    print("=== Symbol breakdown (all splits) ===")
    for sym, grp in df.groupby("symbol"):
        r = grp["excess_return"].values
        print(f"  {sym:10s}  n={len(r):4d}  mean={r.mean():.4f}  win_rate={(r>0).mean():.3f}")


if __name__ == "__main__":
    main()

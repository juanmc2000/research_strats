"""
CTU formal validation script.

Runs train / validation / forward analysis with bootstrap uncertainty
and LOSO portability check. Writes results to features.ctu_validation_results.

Usage:
    python scripts/ctu/markov_validation.py [--threshold-set baseline|narrow|wide]
"""
import argparse
import sys
from datetime import datetime, timezone

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


def run_validation(df: pd.DataFrame, cohort_name: str, threshold_set_name: str) -> list[dict]:
    results = []
    all_returns = df["excess_return"].values
    baseline_mean = all_returns.mean() if len(all_returns) > 0 else 0.0

    for split in ["train", "validation", "forward", "all"]:
        if split == "all":
            subset = df
        else:
            subset = df[df["split"] == split]
        if len(subset) < 5:
            continue

        r = subset["excess_return"].values
        ci_low, ci_high, frac_pos_boot = bootstrap_ci(r, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE)
        loso_fp, loso_th = loso_mean(r, subset["symbol"].values)
        mean_r = r.mean()
        lift = mean_r - baseline_mean

        verdict = "FAIL"
        if mean_r > 0 and frac_pos_boot > 0.80 and loso_fp > 0.60:
            verdict = "PASS"
        elif mean_r > 0 and frac_pos_boot > 0.60:
            verdict = "CONDITIONAL"

        results.append({
            "candidate_state": CANDIDATE_STATE,
            "cohort_name": cohort_name,
            "split_name": split,
            "n_events": int(len(r)),
            "mean_excess_return": float(mean_r),
            "median_excess_return": float(np.median(r)),
            "win_rate": float((r > 0).mean()),
            "payoff": float(r[r > 0].mean() / abs(r[r < 0].mean())) if (r < 0).any() and (r > 0).any() else None,
            "baseline_mean": float(baseline_mean),
            "lift": float(lift),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "frac_positive_boot": float(frac_pos_boot),
            "loso_frac_positive": float(loso_fp),
            "loso_frac_top_half": float(loso_th),
            "verdict": verdict,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        })
    return results


def write_results(engine, results: list[dict]):
    df = pd.DataFrame(results)
    df.to_sql("ctu_validation_results", engine, schema="features", if_exists="append", index=False)
    print(f"Wrote {len(df)} rows to features.ctu_validation_results")


def print_summary(results: list[dict]):
    print(f"\n=== CTU Formal Validation ===")
    print(f"State: {CANDIDATE_STATE}\n")
    for r in results:
        print(f"[{r['cohort_name']}] [{r['split_name']:12s}]  "
              f"n={r['n_events']:4d}  mean={r['mean_excess_return']:+.4f}  "
              f"win={r['win_rate']:.3f}  frac_pos_boot={r['frac_positive_boot']:.3f}  "
              f"loso_fp={r['loso_frac_positive']:.3f}  verdict={r['verdict']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-set", default="baseline",
                        choices=list(THRESHOLD_SETS.keys()))
    parser.add_argument("--no-write", action="store_true",
                        help="Print only, do not write to Timescale")
    args = parser.parse_args()

    ts = THRESHOLD_SETS[args.threshold_set]
    engine = get_engine()
    df = load_events(engine, ts["vol_lo"], ts["eff_hi"])
    df["split"] = df["year"].apply(label_split)

    cohort_name = f"ctu_baseline_{args.threshold_set}"
    results = run_validation(df, cohort_name, args.threshold_set)
    print_summary(results)

    if not args.no_write:
        write_results(engine, results)


if __name__ == "__main__":
    main()

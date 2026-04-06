"""
CTU pre-deployment qualification.

Runs threshold stability, friction stress, and tail analysis.
Writes results to features.ctu_predeployment_results.

Usage:
    python scripts/ctu/markov_pre_deployment.py
"""
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, ".")
from scripts.shared.db import get_engine
from scripts.shared.stats import bootstrap_ci, sharpe, max_drawdown
from scripts.ctu.constants import (
    CTU_EVENT_QUERY,
    MEDIUM_COST,
    WEEKEND_EXTRA_COST,
    N_BOOTSTRAP,
    BLOCK_SIZE,
    TRAIN_CUTOFF_YEAR,
    VALIDATION_CUTOFF_YEAR,
    THRESHOLD_SETS,
)

CANDIDATE_STATE = "CONTRACTING_TRENDING_UP"
FRICTION_VARIANTS = {
    "zero_cost":    0.00,
    "low_cost":     0.03,
    "medium_cost":  MEDIUM_COST,
    "high_cost":    0.12,
    "weekend_cost": MEDIUM_COST + WEEKEND_EXTRA_COST,
}


def load_events(engine, vol_lo: float, eff_hi: float) -> pd.DataFrame:
    params = {"vol_lo": vol_lo, "eff_hi": eff_hi}
    with engine.connect() as conn:
        df = pd.read_sql(text(CTU_EVENT_QUERY), conn, params=params)
    df["year"] = df["year"].astype(int)
    return df


def label_split(year: int) -> str:
    if year < TRAIN_CUTOFF_YEAR:
        return "train"
    elif year < VALIDATION_CUTOFF_YEAR:
        return "validation"
    else:
        return "forward"


def analyze_config(df: pd.DataFrame, cost: float, split: str) -> dict:
    if split != "all":
        subset = df[df["split"] == split]
    else:
        subset = df
    if len(subset) < 5:
        return {}

    r = (subset["realized_return_sd30d"] - cost).values
    cum = np.cumsum(r)
    ci_low, ci_high, _ = bootstrap_ci(r, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE)

    verdict = "FAIL"
    if r.mean() > 0 and sharpe(r) > 0.3:
        verdict = "PASS"
    elif r.mean() > 0:
        verdict = "MONITOR"

    return {
        "candidate_state": CANDIDATE_STATE,
        "section_name": "friction_stress",
        "config_name": f"cost={cost}",
        "split_name": split,
        "n_events": int(len(r)),
        "mean_return": float(r.mean()),
        "sharpe": float(sharpe(r)),
        "max_drawdown": float(max_drawdown(cum)),
        "win_rate": float((r > 0).mean()),
        "payoff": float(r[r > 0].mean() / abs(r[r < 0].mean())) if (r < 0).any() and (r > 0).any() else None,
        "p_lt_neg1": float((r < -1).mean()),
        "p_gt_pos1": float((r > 1).mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "verdict": verdict,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def run_threshold_stability(engine) -> list[dict]:
    results = []
    for ts_name, ts in THRESHOLD_SETS.items():
        df = load_events(engine, ts["vol_lo"], ts["eff_hi"])
        df["split"] = df["year"].apply(label_split)
        for split in ["train", "validation", "forward"]:
            subset = df[df["split"] == split] if split != "all" else df
            if len(subset) < 5:
                continue
            r = (subset["realized_return_sd30d"] - MEDIUM_COST).values
            cum = np.cumsum(r)
            ci_low, ci_high, _ = bootstrap_ci(r, n_boot=N_BOOTSTRAP, block_size=BLOCK_SIZE)
            verdict = "FAIL"
            if r.mean() > 0 and sharpe(r) > 0.3:
                verdict = "PASS"
            elif r.mean() > 0:
                verdict = "MONITOR"
            results.append({
                "candidate_state": CANDIDATE_STATE,
                "section_name": "threshold_stability",
                "config_name": ts_name,
                "split_name": split,
                "n_events": int(len(r)),
                "mean_return": float(r.mean()),
                "sharpe": float(sharpe(r)),
                "max_drawdown": float(max_drawdown(cum)),
                "win_rate": float((r > 0).mean()),
                "payoff": float(r[r > 0].mean() / abs(r[r < 0].mean())) if (r < 0).any() and (r > 0).any() else None,
                "p_lt_neg1": float((r < -1).mean()),
                "p_gt_pos1": float((r > 1).mean()),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "verdict": verdict,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            })
    return results


def write_results(engine, results: list[dict]):
    df = pd.DataFrame([r for r in results if r])
    df.to_sql("ctu_predeployment_results", engine, schema="features", if_exists="append", index=False)
    print(f"Wrote {len(df)} rows to features.ctu_predeployment_results")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    ts = THRESHOLD_SETS["baseline"]
    df = load_events(engine, ts["vol_lo"], ts["eff_hi"])
    df["split"] = df["year"].apply(label_split)

    results = []

    # Friction stress
    print("\n=== Friction Stress ===")
    for cost_name, cost in FRICTION_VARIANTS.items():
        for split in ["train", "validation", "forward"]:
            res = analyze_config(df, cost, split)
            if res:
                res["config_name"] = cost_name
                res["section_name"] = "friction_stress"
                results.append(res)
                print(f"  [{cost_name}] [{split:12s}]  n={res['n_events']}  "
                      f"mean={res['mean_return']:+.4f}  sharpe={res['sharpe']:.3f}  "
                      f"verdict={res['verdict']}")

    # Threshold stability
    print("\n=== Threshold Stability ===")
    ts_results = run_threshold_stability(engine)
    for r in ts_results:
        results.append(r)
        print(f"  [{r['config_name']}] [{r['split_name']:12s}]  n={r['n_events']}  "
              f"mean={r['mean_return']:+.4f}  sharpe={r['sharpe']:.3f}  verdict={r['verdict']}")

    if not args.no_write:
        write_results(engine, results)


if __name__ == "__main__":
    main()

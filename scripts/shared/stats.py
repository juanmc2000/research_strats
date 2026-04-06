"""
Statistical helpers for CTU research: bootstrap, block bootstrap, LOSO.
"""
import numpy as np
from typing import Callable, Optional


def block_bootstrap(
    returns: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    block_size: int = 20,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Stationary block bootstrap.
    Returns array of bootstrapped statistics.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    results = np.empty(n_boot)
    for i in range(n_boot):
        sample = []
        while len(sample) < n:
            start = rng.integers(0, n)
            block = returns[start : start + block_size]
            if start + block_size > n:
                block = np.concatenate([returns[start:], returns[: (start + block_size - n)]])
            sample.extend(block.tolist())
        results[i] = stat_fn(np.array(sample[:n]))
    return results


def loso_mean(
    returns: np.ndarray,
    symbols: np.ndarray,
) -> tuple[float, float]:
    """
    Leave-one-symbol-out mean.
    Returns (frac_positive, frac_top_half) across LOSO folds.
    """
    unique_syms = np.unique(symbols)
    fold_means = []
    for sym in unique_syms:
        mask = symbols != sym
        if mask.sum() == 0:
            continue
        fold_means.append(returns[mask].mean())
    fold_means = np.array(fold_means)
    pooled_mean = returns.mean()
    frac_positive = (fold_means > 0).mean()
    frac_top_half = (fold_means > np.median(fold_means)).mean()
    return float(frac_positive), float(frac_top_half)


def bootstrap_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    block_size: int = 20,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> tuple[float, float, float]:
    """
    Block bootstrap confidence interval for the mean.
    Returns (ci_low, ci_high, frac_positive_boot).
    """
    boot = block_bootstrap(returns, np.mean, n_boot=n_boot, block_size=block_size, seed=seed)
    ci_low = float(np.percentile(boot, 100 * alpha / 2))
    ci_high = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    frac_positive = float((boot > 0).mean())
    return ci_low, ci_high, frac_positive


def sharpe(returns: np.ndarray, annualize: int = 1) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualize))


def max_drawdown(cumulative: np.ndarray) -> float:
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / np.where(peak != 0, peak, 1)
    return float(dd.min())

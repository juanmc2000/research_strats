"""
Common feature calculations for CTU research.
These match the definitions used in TimescaleDB features tables.
"""
import numpy as np
import pandas as pd


def vol_ratio(close: pd.Series, short_window: int = 20, long_window: int = 100) -> pd.Series:
    """vol_ratio_20_100: ratio of short-window to long-window realised vol."""
    short_vol = close.pct_change().rolling(short_window).std()
    long_vol = close.pct_change().rolling(long_window).std()
    return short_vol / long_vol.replace(0, np.nan)


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """efficiency_20: directional efficiency = net move / sum of absolute moves."""
    net_move = (close - close.shift(window)).abs()
    total_path = close.diff().abs().rolling(window).sum()
    return net_move / total_path.replace(0, np.nan)

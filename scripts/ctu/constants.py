"""
CTU research constants. Do not change without a new issue and versioned research note.
"""

# Signal thresholds — baseline
VOL_CONTRACTING_MAX = 0.80
EFF_TRENDING_MIN = 0.60

# Friction
MEDIUM_COST = 0.07
WEEKEND_EXTRA_COST = 0.03

# Portfolio constraints
SHRINKAGE_FACTOR = 0.50
MAX_PER_SYMBOL = 2
MAX_PER_STATE = 5
MAX_TOTAL = 10

# Bootstrap
N_BOOTSTRAP = 1000
BLOCK_SIZE = 20

# Threshold stability sets
THRESHOLD_SETS = {
    "baseline": {"vol_lo": 0.80, "eff_hi": 0.60},
    "narrow":   {"vol_lo": 0.85, "eff_hi": 0.65},
    "wide":     {"vol_lo": 0.75, "eff_hi": 0.55},
}

# Time splits
TRAIN_CUTOFF_YEAR = 2022
VALIDATION_CUTOFF_YEAR = 2024

# Universe
CTU_SYMBOLS = [
    "EUR/USD",
    "GBP/USD",
    "USD/CHF",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD",
    "EUR/GBP",
]

# Core SQL for loading CTU events
CTU_EVENT_QUERY = """
SELECT
    be.symbol,
    be.event_hour_ts,
    be.breakout_direction,
    be.vol_ratio_20_100,
    be.efficiency_20,
    be.realized_return_sd30d,
    be.bars_held,
    be.exit_reason,
    be.max_favorable_excursion_sd30d,
    be.max_adverse_excursion_sd30d,
    EXTRACT(YEAR FROM be.event_hour_ts) AS year
FROM features.breakout_events be
WHERE be.exit_reason IS NOT NULL
  AND be.entry_range_sd_30d > 0
  AND be.vol_ratio_20_100 < %(vol_lo)s
  AND be.efficiency_20 > %(eff_hi)s
  AND be.breakout_direction = 'UP'
  AND be.realized_return_sd30d IS NOT NULL
ORDER BY be.symbol, be.event_hour_ts;
"""

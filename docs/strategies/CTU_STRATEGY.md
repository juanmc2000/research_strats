# CTU Strategy

## Status

CTU is an active research strategy.

It is not frozen.
It is not approved for live trading.
It is not approved for advanced sizing.
It is not approved for integration with ETD.

CTU stays in research until a new issue explicitly reopens promotion work.

---

## Strategy identity

CTU stands for:

**CONTRACTING_TRENDING_UP**

This is a breakout-quality state.

It is designed to identify long breakout entries that happen while volatility is contracting, but directional efficiency is already high.

The intuition is simple:

- volatility has compressed
- price action is still moving cleanly in one direction
- the breakout may therefore have better payoff than a generic breakout

---

## Exact signal definition

A CTU trade is an event where all of the following are true:

- `vol_ratio_20_100 < 0.80`
- `efficiency_20 > 0.60`
- `breakout_direction = 'UP'`

These conditions define the state:

**CONTRACTING_TRENDING_UP**

Do not change these thresholds without a new issue and a versioned research note.

---

## Return definition

The current research return unit is:

`realized_return_sd30d - 0.07`

Where:

- `realized_return_sd30d` is the realized trade return in 30-day SD range units
- `0.07` is the medium friction assumption

This means all CTU performance work is currently evaluated on a medium-cost basis.

---

## Current constants

These are the active CTU research constants and should remain explicit in code.

- `VOL_CONTRACTING_MAX = 0.80`
- `EFF_TRENDING_MIN = 0.60`
- `MEDIUM_COST = 0.07`
- `WEEKEND_EXTRA_COST = 0.03`
- `SHRINKAGE_FACTOR = 0.50`
- `MAX_PER_SYMBOL = 2`
- `MAX_PER_STATE = 5`
- `MAX_TOTAL = 10`
- `N_BOOTSTRAP = 1000`
- `BLOCK_SIZE = 20`

These belong to research configuration, not to live execution logic.

---

## Threshold stability sets

Three threshold sets have already been used for robustness work.

### Baseline
- `vol_lo = 0.80`
- `eff_hi = 0.60`

### Narrow
- `vol_lo = 0.85`
- `eff_hi = 0.65`

### Wide
- `vol_lo = 0.75`
- `eff_hi = 0.55`

These are stability checks only.

They are not permission to keep moving thresholds until results improve.

---

## Time splits

All CTU work should preserve the current split structure unless a new issue explicitly changes it.

- **Train:** `year < 2022`
- **Validation:** `2022 <= year < 2024`
- **Forward:** `year >= 2024`

This split structure is part of the research discipline and should not drift casually.

---

## Current universe

The core CTU FX universe is:

- EUR/USD
- GBP/USD
- USD/CHF
- AUD/USD
- USD/CAD
- NZD/USD
- EUR/GBP

This is the base universe for research.

Any pruning, qualification, or sleeve construction must be documented explicitly and persisted to TimescaleDB.

---

## Data source

CTU research uses TimescaleDB as the source of truth.

Primary inputs currently come from:

- `features.breakout_events`
- `features.hourly_market_features`
- `features.daily_market_metrics`

No parquet files.
No prod trading DB writes.
No alternate local truth datasets.

---

## Core SQL shape

The core CTU event load follows this pattern:

```sql
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
    be.max_adverse_excursion_sd30d
FROM features.breakout_events be
WHERE be.exit_reason IS NOT NULL
  AND be.entry_range_sd_30d > 0
  AND be.vol_ratio_20_100 < 0.80
  AND be.efficiency_20 > 0.60
  AND be.breakout_direction = 'UP'
  AND be.realized_return_sd30d IS NOT NULL
ORDER BY be.symbol, be.event_hour_ts;
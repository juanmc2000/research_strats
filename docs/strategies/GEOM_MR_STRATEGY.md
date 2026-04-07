# Baseline Geometric Mean Reversion Strategy

## Status

GEOM_MR is an active exploratory research strategy (Stage A).

It is not frozen.
It is not approved for live trading.
It is not approved for advanced sizing.
It is completely isolated from CTU, ETD, STAT_MR, and any live trading infrastructure.

GEOM_MR stays in research until a new issue explicitly promotes it.

---

## Strategy identity

GEOM_MR stands for:

**Baseline Geometric Mean Reversion**

This is a path-based baseline research object.

The goal is to determine whether raw path displacement from a local manifold contains edge, before any volatility normalization or statistical blending.

The key distinction from STAT_MR:

- STAT_MR normalizes displacement by rolling volatility (z-score)
- GEOM_MR uses raw price displacement in instrument-native units
- No z-score conversion
- No volatility scaling of any kind

The question is whether the raw geometric distance from a local mean has predictive power independent of volatility normalization.

---

## Exact signal definition

Entry long: `d_t < -threshold`
Entry short: `d_t > +threshold`

Where:

`d_t = close_t - mean_N`

- `mean_N` is the rolling mean of mid close over `N` bars
- `d_t` is raw price displacement from the local manifold — no normalization

Exit: `d_t` crosses 0

- long exits when `d_t >= 0`
- short exits when `d_t <= 0`

No stop loss beyond this rule.
No filters.
No volatility normalization.
No z-score conversion.
No pyramiding.

---

## Price definition

`mid_close = (bid_close + ask_close) / 2`

---

## Parameter grid

These are the pre-declared research parameters. Do not add new values after seeing results.

- `lookback ∈ {10, 20, 30}`
- `distance_threshold`: instrument-native, expressed as multiples of tick size

```
threshold_multipliers = [5, 10, 20]
threshold = multiplier × tick_size
```

Where `tick_size` comes from `market_data.symbols.tick_size`.

Total combinations: 9 per symbol × timeframe.

---

## Timeframes

`5min, 10min, 15min, 30min, 1h, 4h, 6h`

Sub-hourly timeframes use `market_data.minute_prices`.
Hourly+ timeframes use `market_data.hourly_prices`.

---

## Research split

- **In-sample:** `date < 2025-06-01`
- **Unseen:** `date >= 2025-06-01`

All outputs carry `period_label ∈ {in_sample, unseen}`.

The unseen split must never be touched during model selection or parameter tuning.

---

## Universe

Symbols with full historical data in TimescaleDB:

- AUD/CAD, AUD/CHF, AUD/JPY, AUD/NZD, AUD/USD (forex)
- EUR/USD, GBP/USD, USD/JPY (forex)
- AUS200 (global index)
- BTC/USD, BCH/USD (crypto)
- NGAS (energy)
- XAU/USD (metal)

---

## Data source

`market_data.minute_prices` — for 5min, 10min, 15min, 30min
`market_data.hourly_prices` — for 1h, 4h, 6h
`market_data.symbols` — for tick_size (threshold scaling)

All outputs written to `strategy_research` schema only.

---

## Output tables

All under `strategy_research`:

- `experiment_runs` — shared run registry
- `geom_mr_features` — per-bar displacement features at entry bars
- `geom_mr_signals` — all entry signal events
- `geom_mr_trades` — all completed trades
- `geom_mr_analysis_summary` — aggregate metrics per (symbol, timeframe, params)
- `geom_mr_period_comparison` — IS vs unseen comparison with degradation ratios
- `geom_mr_timeframe_comparison` — ranked timeframe performance per symbol
- `geom_mr_param_robustness` — parameter grid stability metrics
- `geom_mr_analysis_by_period` — subperiod performance (calendar year splits)
- `geom_mr_analysis_by_displacement_bucket` — return distribution by displacement bucket at entry

---

## Current constants

- `OOS_CUTOFF = '2025-06-01'`
- `LOOKBACKS = [10, 20, 30]`
- `THRESHOLD_MULTIPLIERS = [5, 10, 20]`
- `TIMEFRAMES = ['5min', '10min', '15min', '30min', '1h', '4h', '6h']`

---

## Comparison with STAT_MR

Both strategies share the same entry/exit logic structure but differ critically:

| | STAT_MR | GEOM_MR |
|---|---------|---------|
| Entry signal | z-score (normalized) | raw displacement |
| Threshold | dimensionless (1.5/2.0/2.5) | instrument-native (5/10/20 × tick) |
| Vol adjustment | yes (rolling std) | none |
| Manifold deviation type | statistical | geometric/path |

The two strategies are designed to be run and compared independently. Do not blend or mix their logic.

---

## Research process position

Stage A — Exploratory discovery.

Goal: determine whether raw path displacement from a local manifold has predictive power without normalization.

The strategy is not frozen.
No filters are permitted at this stage.
No advanced sizing.
No comparisons to CTU or ETD.

Decision to be made after full analysis:

- **CONTINUE** to Stage B if edge exists without normalization
- **MONITOR** if results are ambiguous or thin
- **REJECT** if no credible signal survives unseen evaluation

---

## Isolation requirements

This strategy MUST NOT:

- share code with CTU, ETD, or STAT_MR
- write to `features` schema or any production schema
- use CTU/ETD/STAT_MR parameters or logic
- share DAGs or pipelines with live systems

All outputs go to `strategy_research` schema only.

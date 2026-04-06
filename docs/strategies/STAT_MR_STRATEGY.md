# Baseline Statistical Mean Reversion Strategy

## Status

STAT_MR is an active exploratory research strategy (Stage A).

It is not frozen.
It is not approved for live trading.
It is not approved for advanced sizing.
It is completely isolated from CTU, ETD, and any live trading infrastructure.

STAT_MR stays in research until a new issue explicitly promotes it.

---

## Strategy identity

STAT_MR stands for:

**Baseline Statistical Mean Reversion**

This is a pure baseline research object.

The goal is to determine whether raw statistical mean reversion exists across timeframes and instruments before introducing any filters, regimes, or hybrid logic.

The intuition is simple:

- prices deviate from their rolling mean
- normalised by local volatility (z-score), large deviations tend to revert
- the simplest possible entry/exit captures this if the edge exists

---

## Exact signal definition

Entry long: `z_t < -entry_threshold`
Entry short: `z_t > +entry_threshold`

Where:

`z_t = (close_t - mean_N) / std_N`

- `mean_N` is the rolling mean of mid close over `N` bars
- `std_N` is the rolling standard deviation over `N` bars

Exit: `z_t` crosses 0

- long exits when `z_t >= 0`
- short exits when `z_t <= 0`

No stop loss beyond this rule.
No filters.
No volatility regime conditioning.
No pyramiding.

---

## Price definition

`mid_close = (bid_close + ask_close) / 2`

---

## Parameter grid

These are the pre-declared research parameters. Do not add new values after seeing results.

- `lookback ∈ {10, 20, 30}`
- `entry_threshold ∈ {1.5, 2.0, 2.5}`

Total combinations: 9 per symbol × timeframe.

---

## Timeframes

`5m, 10m, 15m, 30m, 1h, 4h, 6h`

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

Sub-hourly timeframes are restricted to this set (symbols with full minute history).

---

## Data source

`market_data.minute_prices` — for 5m, 10m, 15m, 30m
`market_data.hourly_prices` — for 1h, 4h, 6h

All outputs written to `strategy_research` schema only.

No writes to `features`, `trading_metrics_dev`, or any production schema.

---

## Output tables

All under `strategy_research`:

- `experiment_runs` — run registry with metadata
- `stat_mr_features` — per-bar z-score features at signal bars
- `stat_mr_signals` — all entry signal events
- `stat_mr_trades` — all completed trades
- `stat_mr_analysis_summary` — aggregate metrics per (symbol, timeframe, params)
- `stat_mr_period_comparison` — IS vs unseen comparison with degradation ratios
- `stat_mr_timeframe_comparison` — ranked timeframe performance per symbol
- `stat_mr_param_robustness` — parameter grid stability metrics
- `stat_mr_analysis_by_period` — subperiod performance (calendar splits)
- `stat_mr_analysis_by_entry_bucket` — return distribution by z-score bucket at entry

---

## Current constants

- `OOS_CUTOFF = '2025-06-01'`
- `LOOKBACKS = [10, 20, 30]`
- `THRESHOLDS = [1.5, 2.0, 2.5]`
- `TIMEFRAMES = ['5min', '10min', '15min', '30min', '1h', '4h', '6h']`

These belong to research configuration and must not drift without a new issue.

---

## Research process position

Stage A — Exploratory discovery.

Goal: determine whether raw statistical mean reversion exists as a signal.

The strategy is not frozen.
No filters are permitted at this stage.
No advanced sizing.
No comparisons to CTU or ETD.

Decision to be made after full analysis:

- **CONTINUE** to Stage B (candidate freeze) if edge exists robustly
- **MONITOR** if edge is conditional or thin
- **REJECT** if no credible signal survives unseen evaluation

---

## Isolation requirements

This strategy MUST NOT:

- share code with CTU or ETD
- write to `features` schema or any production schema
- use CTU/ETD parameters, thresholds, or logic
- share DAGs or pipelines with live systems

All outputs go to `strategy_research` schema only.

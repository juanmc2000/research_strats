# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Development

## Running scripts

All scripts are Python CLI scripts runnable directly:

```bash
python scripts/ctu/some_script.py
```

DB credentials are loaded from `.env` in the repo root. The expected variables are:

```
TIMESCALE_HOST=...
TIMESCALE_PORT=...
TIMESCALE_DB=...
TIMESCALE_USER=...
TIMESCALE_PASSWORD=...
```

Use `python-dotenv` or equivalent to load `.env` at the top of each script.

## Directory layout

```
scripts/
  shared/       # reusable utilities: DB connection, stats helpers, bootstrap
  ctu/          # active CTU research scripts
  etd_archive/  # archived ETD reference — do not modify without a new issue
db/
  features/     # SQL migrations for TimescaleDB features schema tables
docs/
  strategies/   # per-strategy docs (CTU_STRATEGY.md, etc.)
  issues/       # per-issue research notes
results/        # lightweight local artifacts only — not the durable record
```

## Shared utilities pattern

Scripts in `scripts/shared/` are imported by CTU scripts. Expected modules when built out:

- `db.py` — connection helper (returns a `psycopg2` or `sqlalchemy` connection from env)
- `stats.py` — bootstrap, block bootstrap, LOSO helpers
- `features.py` — common feature calculations (vol ratio, efficiency, etc.)

## DB write pattern

Durable outputs go to `features` schema only:

```python
# Example: write validation results
df.to_sql("ctu_validation_results", engine, schema="features", if_exists="append", index=False)
```

---

# research_strats — original CLAUDE

This repository is for **systematic strategy research only**.

It is not a live trading repo.
It is not an orchestration repo.
It is not an execution repo.

Its purpose is to support disciplined, reproducible quantitative strategy research in a style closer to a serious institutional / Renaissance-like process.

---

# 1. Mission of this repo

This repo exists to:

- research candidate trading strategies
- determine whether an apparent edge is real
- reject weak or fragile ideas early
- preserve full research lineage
- freeze strong ideas cleanly when warranted
- avoid structural drift while working on multiple ideas

At the moment:

- **CTU is the active research object**
- **ETD is archived methodology / reference unless explicitly reopened**

---

# 2. Hard boundaries

## Never add
- IG execution logic
- Redis state management
- Airflow DAGs
- live trading engine code
- production trading DB writes
- ad hoc parquet research pipelines
- silent threshold tuning
- hidden strategy changes inside refactors

## Data rules
- TimescaleDB is the source of truth
- research outputs may be written only to the `features` schema in TimescaleDB
- do not write to trading/prod DB
- do not create alternate local “truth” datasets

---

# 3. Current active object: CTU

## Current CTU signal definition
State: `CONTRACTING_TRENDING_UP`

Signal:
- `vol_ratio_20_100 < 0.80`
- `efficiency_20 > 0.60`
- `breakout_direction = 'UP'`

Return unit:
- `realized_return_sd30d - 0.07`

## Current CTU universe
Core G10 FX:
- EUR/USD
- GBP/USD
- USD/CHF
- AUD/USD
- USD/CAD
- NZD/USD
- EUR/GBP

## Current CTU time splits
- Train: `year < 2022`
- Validation: `2022 <= year < 2024`
- Forward: `year >= 2024`

## Current CTU constants
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

## Threshold stability sets
- baseline: `vol_lo=0.80`, `eff_hi=0.60`
- narrow: `vol_lo=0.85`, `eff_hi=0.65`
- wide: `vol_lo=0.75`, `eff_hi=0.55`

## Current status
CTU is **not frozen**.

It is still an active research object.

CTU integration is blocked unless explicitly reopened by a dedicated issue. Do not silently resume integration work.

---

# 4. ETD status in this repo

ETD is here as:
- methodological reference
- archived research lineage
- example of disciplined freeze / qualify / sleeve / pre-sizing workflow

ETD is **not** the active development target in this repo unless explicitly reopened.

Do not modify archived ETD logic in-place unless a new issue explicitly requires it.

---

# 5. Research philosophy

This repo follows a strong quant research discipline:

1. Start with a clear hypothesis
2. Test the simple version first
3. Separate exploration from validation
4. Freeze candidates before formal testing
5. Prefer robustness over backtest aesthetics
6. Favor portability, stability, and honest failure analysis
7. Keep filters sparse
8. Avoid rescuing weak ideas with complexity
9. Do not move to sizing too early
10. Once frozen, do not silently alter the object

The core question is:
> Is this a real, portable, risk-adjusted edge that survives honest out-of-sample scrutiny?

Not:
> Can we improve this backtest?

---

# 6. Required research lifecycle

Every strategy idea should move through these stages.

## Stage A — exploratory discovery
Goal:
- determine whether there is a plausible signal

Allowed:
- descriptive analysis
- state analysis
- symbol decomposition
- carry/drift neutralization
- failure-mode analysis
- exploratory plots/tables

Not allowed:
- production claims
- advanced sizing
- pyramiding
- threshold rescue games

Deliverable:
- exploratory memo with candidate hypotheses

## Stage B — candidate freeze
Goal:
- define a small set of candidate states or rules before formal validation

Rules:
- candidate definitions must be documented before formal testing
- do not add new candidates after seeing validation results unless a new issue is opened

Deliverable:
- frozen candidate registry in code + docs

## Stage C — formal validation
Required where relevant:
- train / validation / forward split
- bootstrap uncertainty
- LOSO portability or equivalent
- simpler-model comparison
- multiple-testing awareness
- explicit pass/fail criteria

Deliverable:
- PASS / CONDITIONAL / FAIL with reasons

## Stage D — pre-deployment qualification
Goal:
- determine whether the candidate is robust enough to become a strategy object

Checks may include:
- threshold stability
- friction stress
- tail/lifecycle analysis
- probability calibration
- simplicity sanity
- symbol qualification
- concentration / heterogeneity

Deliverable:
- KEEP / MONITOR / EXCLUDE or equivalent
- explicit blockers if not ready

## Stage E — sleeve construction
Only after the single object is credible.

Rules:
- use pre-qualified symbols only
- test whether added symbols reduce drawdown without damaging Sharpe too much
- reject contaminating symbols
- document exact sleeve composition

Deliverable:
- frozen sleeve definition or rejection

## Stage F — pre-sizing
Only after the object or sleeve is frozen.

Rules:
- begin with simple weights
- compare a very small number of pre-declared schemes
- do not jump to Kelly / pyramiding / advanced sizing while concentration is unresolved

Deliverable:
- PASS / MONITOR / STOP on simple sizing only

## Stage G — frozen monitoring
Only after a strategy is frozen.

Rules:
- no re-optimization
- no silent structural edits
- monitoring accumulates evidence; it is not another tuning phase

---

# 7. Anti-overfitting rules

Do not do any of the following without a new issue and explicit documentation:

- threshold sweeps after seeing results
- symbol pruning based only on one favorable forward slice
- adding filters one by one just to save a weak signal
- repeatedly changing return definition, stop logic, or horizon until the result looks better
- promoting a strategy because it “almost works”
- moving to sizing before edge quality is credible

A filter is only acceptable if all of the following are true:
1. it addresses a clear failure mode
2. the mechanism is understandable
3. it improves the right metric(s)
4. it does not obviously damage forward behavior
5. it is documented before promotion

---

# 8. What must be persisted to TimescaleDB

Durable research outputs should not live only in stdout or ad hoc text files.

If a research output is important enough to inform future decisions, it should be persisted to a TimescaleDB table in `features`.

Examples:
- validation results
- symbol qualification rulings
- pre-deployment result tables
- sleeve simulation outputs
- concentration / dependency analysis
- forward monitoring snapshots

Text summaries may still exist, but Timescale tables are the durable record.

---

# 9. Coding rules

## Structure
- `scripts/shared/` → shared framework/utilities
- `scripts/ctu/` → active CTU research
- `scripts/etd_archive/` → ETD archive/reference only

## General
- scripts must be runnable from CLI
- keep constants explicit near the top
- strategy definitions must be visible and auditable
- avoid hidden logic and implicit thresholds
- prefer deterministic scripts over notebooks

## DB access
- use TimescaleDB only
- keep SQL explicit and readable
- strategy logic should be clear in code, not buried in opaque queries

## Outputs
- may print summaries
- may write lightweight artifacts to `results/`
- durable outputs should go to TimescaleDB `features` schema
- do not use parquet unless explicitly requested

---

# 10. Documentation rules

Every meaningful issue / script should leave behind:
- what was tested
- why it was tested
- what was frozen before testing
- what passed / failed
- what decision was taken next

For CTU specifically, maintain:
- exact current definition
- current universe
- current evidence
- known failure modes
- blockers
- whether the object is exploratory, candidate, frozen, or rejected

If an object becomes frozen, document:
- exact symbols
- exact thresholds
- exact filters
- exact weighting
- exact blocked changes during monitoring

---

# 11. Decision vocabulary

Every major research step should end with one of:
- **CONTINUE**
- **MONITOR**
- **FREEZE**
- **REJECT**
- **REVERT**
- **BLOCKED**

Do not leave strategy status ambiguous.

---

# 12. Priority order

Optimize for, in order:

1. honesty
2. reproducibility
3. out-of-sample behavior
4. robustness
5. risk-adjusted edge
6. interpretability of failure modes
7. simplicity of the final object

Do not optimize for:
- pretty backtests
- rescuing weak ideas
- maximizing historical PnL at any cost

---

# 13. Sizing discipline

Default stance:
- no advanced sizing until the edge is credible
- no Kelly until sleeve/object concentration is acceptable
- no pyramiding until the base entry object is frozen and validated

Default order:
1. equal weight
2. capped simple alternatives
3. frozen monitoring
4. only then consider more advanced sizing

---

# 14. Working style for Claude in this repo

When working here:
- keep CTU as the active research object
- preserve ETD as archived reference
- make all assumptions explicit
- challenge flattering results
- identify whether the edge is:
  - real
  - conditional
  - concentrated
  - fragile
  - contaminated
  - or likely accidental

Ask continuously:
> Would a serious quant team trust this as a reusable research step?

If not, tighten the process.

---

# 15. Final standard

The output of this repo should be strong enough that:
- another researcher can reproduce the result from code + Timescale data
- another model cannot silently mutate the strategy object
- a future frozen-strategy monitoring repo can port the final object without ambiguity
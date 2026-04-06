
---

# `README.md`

```markdown
# research_strats

This repository is for systematic strategy research only.

It is the clean home for active research strategies that should be kept separate from live trading infrastructure.

At the moment, the main active object is:

- **CTU**, an active research strategy

This repo may also contain archived strategy material for reference, but archived strategies are not the active development target unless explicitly reopened.

---

## Why this repo exists

The previous environment mixed together:

- strategy research
- ETL
- orchestration
- live trading code
- execution plumbing
- application state management

That makes disciplined research harder.

This repo exists to make research cleaner and more reproducible.

The goals are:

- isolate active research from execution code
- preserve clear strategy definitions
- prevent accidental strategy drift
- keep TimescaleDB as the research source of truth
- persist meaningful research outputs in the database
- follow a disciplined quant process from hypothesis to validation to freeze or rejection

---

## What belongs here

This repo should contain:

- research scripts
- shared research utilities
- TimescaleDB feature and result-table migrations
- strategy docs
- process docs
- archived strategy notes for reference
- lightweight result artifacts when useful

---

## What does not belong here

This repo should not contain:

- live trading engine code
- Redis logic
- IG execution code
- Airflow DAGs
- prod trading DB writes
- app-state persistence
- ad hoc local truth datasets
- parquet-first research workflows

If a piece of code is primarily about execution or orchestration, it belongs elsewhere.

---

## Data discipline

TimescaleDB is the source of truth.

Research inputs come from the `features` schema and upstream market data tables.

Durable research outputs should also be written back into TimescaleDB `features` tables.

Rules:

- no parquet as the primary research store
- no production trading DB writes
- no alternate hidden data store
- no silent manual edits to data used for research conclusions

---

## Current active strategy: CTU

CTU means:

**CONTRACTING_TRENDING_UP**

It is currently defined as:

- `vol_ratio_20_100 < 0.80`
- `efficiency_20 > 0.60`
- `breakout_direction = 'UP'`

Current return unit:

- `realized_return_sd30d - 0.07`

CTU is still an active research object.

It is not frozen.
It is not approved for live trading.
It is not approved for advanced sizing.
It is not approved for ETD integration.

See:

- `docs/strategies/CTU_STRATEGY.md`

---

## ETD in this repo

ETD is not the main active strategy here.

ETD may appear in this repo as:

- archived methodology
- reference material
- an example of how a strategy moved through freeze and sleeve qualification

Archived ETD material should not be edited casually.

---

## Research process philosophy

This repo follows a disciplined, institutional-style process.

The goal is not to make a backtest look good.

The goal is to determine whether an edge is:

- real
- robust
- portable
- risk-aware
- and honest out of sample

The preferred process is:

1. hypothesis
2. exploratory analysis
3. candidate freeze
4. formal validation
5. pre-deployment qualification
6. sleeve construction if warranted
7. simple pre-sizing if warranted
8. frozen monitoring if warranted

At each stage, the strategy can also be rejected.

---

## Repository layout

A target structure looks like this:

```text
research_strats/
├── CLAUDE.md
├── README.md
├── .env.example
├── requirements.txt or pyproject.toml
├── db/
│   └── features/
├── scripts/
│   ├── shared/
│   ├── ctu/
│   └── etd_archive/
├── docs/
│   ├── process/
│   ├── strategies/
│   └── issues/
├── results/
│   └── .gitkeep
└── tests/
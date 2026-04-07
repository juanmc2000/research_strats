-- Migration: 004_stat_mr_stageb_tables.sql
-- Stage B tables for STAT_MR focused research (4h/6h, reduced param space, side separation).
-- OOS cutoff: 2025-02-01 (revised from Stage A 2025-06-01).
-- Idempotent.

-- Trades with side_filter applied
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_stageb_trades (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    side_filter     TEXT NOT NULL,   -- 'long_only' | 'short_only' | 'combined'
    entry_ts        TIMESTAMPTZ NOT NULL,
    exit_ts         TIMESTAMPTZ,
    entry_price     NUMERIC NOT NULL,
    exit_price      NUMERIC,
    z_at_entry      NUMERIC NOT NULL,
    z_at_exit       NUMERIC,
    side            INT NOT NULL,
    bars_held       INT,
    return_pct      NUMERIC,
    period_label    TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, side_filter, entry_ts, side)
);

-- Analysis summary
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_stageb_analysis (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    side_filter     TEXT NOT NULL,
    period_label    TEXT NOT NULL,
    n_trades        INT,
    mean_return     NUMERIC,
    median_return   NUMERIC,
    std_return      NUMERIC,
    sharpe          NUMERIC,
    win_rate        NUMERIC,
    payoff          NUMERIC,
    cum_return      NUMERIC,
    max_drawdown    NUMERIC,
    avg_bars_held   NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, side_filter, period_label)
);

-- Kill/pass verdict
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_stageb_verdict (
    run_id          UUID NOT NULL,
    stage           TEXT NOT NULL,
    verdict         TEXT NOT NULL,   -- 'PASS' | 'KILL'
    reason          TEXT,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, stage)
);

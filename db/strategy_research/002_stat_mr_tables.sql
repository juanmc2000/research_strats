-- Migration: 002_stat_mr_tables.sql
-- Tables for Baseline Statistical Mean Reversion (STAT_MR) research.
-- All tables under strategy_research schema.
-- Idempotent: safe to run multiple times.

-- Features at signal-eligible bars (|z| > min threshold)
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_features (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    bar_ts          TIMESTAMPTZ NOT NULL,
    mid_close       NUMERIC NOT NULL,
    mean_n          NUMERIC NOT NULL,
    std_n           NUMERIC NOT NULL,
    z_score         NUMERIC NOT NULL,
    period_label    TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, bar_ts)
);

-- Entry signal events
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_signals (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    signal_ts       TIMESTAMPTZ NOT NULL,
    side            INT NOT NULL,         -- +1 long, -1 short
    z_at_entry      NUMERIC NOT NULL,
    mid_at_entry    NUMERIC NOT NULL,
    period_label    TEXT NOT NULL,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, signal_ts, side)
);

-- Completed trades
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_trades (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
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
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, entry_ts, side)
);

-- Aggregate metrics per (symbol, timeframe, lookback, threshold, period)
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_analysis_summary (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
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
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, period_label)
);

-- In-sample vs unseen comparison with degradation ratios
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_period_comparison (
    run_id              UUID NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    lookback            INT NOT NULL,
    threshold           NUMERIC NOT NULL,
    is_sharpe           NUMERIC,
    oos_sharpe          NUMERIC,
    is_mean_return      NUMERIC,
    oos_mean_return     NUMERIC,
    is_win_rate         NUMERIC,
    oos_win_rate        NUMERIC,
    is_payoff           NUMERIC,
    oos_payoff          NUMERIC,
    is_max_dd           NUMERIC,
    oos_max_dd          NUMERIC,
    sharpe_degradation  NUMERIC,
    return_degradation  NUMERIC,
    trade_count_ratio   NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold)
);

-- Ranked timeframe performance per symbol
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_timeframe_comparison (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    timeframe       TEXT NOT NULL,
    period_label    TEXT NOT NULL,
    sharpe          NUMERIC,
    n_trades        INT,
    win_rate        NUMERIC,
    tf_rank         INT,
    PRIMARY KEY (run_id, symbol, lookback, threshold, timeframe, period_label)
);

-- Parameter grid stability
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_param_robustness (
    run_id              UUID NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    period_label        TEXT NOT NULL,
    lookback            INT NOT NULL,
    threshold           NUMERIC NOT NULL,
    sharpe              NUMERIC,
    n_trades            INT,
    neighbor_mean_sharpe NUMERIC,
    stability_score     NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, period_label, lookback, threshold)
);

-- Subperiod performance (calendar year splits)
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_analysis_by_period (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    period_year     INT NOT NULL,
    n_trades        INT,
    mean_return     NUMERIC,
    sharpe          NUMERIC,
    win_rate        NUMERIC,
    cum_return      NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, period_year)
);

-- Return distribution by z-score bucket at entry
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_analysis_by_entry_bucket (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    period_label    TEXT NOT NULL,
    z_bucket        TEXT NOT NULL,   -- e.g. '[-3,-2.5)', '[-2.5,-2)', etc.
    side            INT NOT NULL,
    n_trades        INT,
    mean_return     NUMERIC,
    win_rate        NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, period_label, z_bucket, side)
);

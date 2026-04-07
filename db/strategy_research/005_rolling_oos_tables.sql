-- Migration: 005_rolling_oos_tables.sql
-- Rolling walk-forward OOS tables for STAT_MR and GEOM_MR.
-- Expanding window: train = all data before test_year, test = calendar year slice.
-- Idempotent.

-- STAT_MR rolling OOS — per fold results
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_rolling_folds (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold       NUMERIC NOT NULL,
    side_filter     TEXT NOT NULL,
    test_year       INT NOT NULL,
    -- train = all trades with entry year < test_year
    train_n_trades  INT,
    train_sharpe    NUMERIC,
    train_win_rate  NUMERIC,
    train_cum_ret   NUMERIC,
    -- test = trades with entry year == test_year
    test_n_trades   INT,
    test_sharpe     NUMERIC,
    test_win_rate   NUMERIC,
    test_cum_ret    NUMERIC,
    test_max_dd     NUMERIC,
    test_avg_bars   NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, side_filter, test_year)
);

-- STAT_MR rolling OOS — summary across all folds per param combo
CREATE TABLE IF NOT EXISTS strategy_research.stat_mr_rolling_summary (
    run_id              UUID NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    lookback            INT NOT NULL,
    threshold           NUMERIC NOT NULL,
    side_filter         TEXT NOT NULL,
    n_folds             INT,
    mean_test_sharpe    NUMERIC,
    median_test_sharpe  NUMERIC,
    std_test_sharpe     NUMERIC,
    min_test_sharpe     NUMERIC,
    max_test_sharpe     NUMERIC,
    pct_pos_folds       NUMERIC,  -- fraction of test years with sharpe > 0
    mean_test_wr        NUMERIC,
    mean_test_cum       NUMERIC,
    mean_test_dd        NUMERIC,
    mean_train_sharpe   NUMERIC,
    train_test_corr     NUMERIC,  -- Pearson r between train and test Sharpe across folds
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold, side_filter)
);

-- GEOM_MR rolling OOS — per fold results
CREATE TABLE IF NOT EXISTS strategy_research.geom_mr_rolling_folds (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    lookback        INT NOT NULL,
    threshold_mult  INT NOT NULL,
    test_year       INT NOT NULL,
    train_n_trades  INT,
    train_sharpe    NUMERIC,
    train_win_rate  NUMERIC,
    train_cum_ret   NUMERIC,
    test_n_trades   INT,
    test_sharpe     NUMERIC,
    test_win_rate   NUMERIC,
    test_cum_ret    NUMERIC,
    test_max_dd     NUMERIC,
    test_avg_bars   NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold_mult, test_year)
);

-- GEOM_MR rolling OOS — summary across all folds
CREATE TABLE IF NOT EXISTS strategy_research.geom_mr_rolling_summary (
    run_id              UUID NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    lookback            INT NOT NULL,
    threshold_mult      INT NOT NULL,
    n_folds             INT,
    mean_test_sharpe    NUMERIC,
    median_test_sharpe  NUMERIC,
    std_test_sharpe     NUMERIC,
    min_test_sharpe     NUMERIC,
    max_test_sharpe     NUMERIC,
    pct_pos_folds       NUMERIC,
    mean_test_wr        NUMERIC,
    mean_test_cum       NUMERIC,
    mean_test_dd        NUMERIC,
    mean_train_sharpe   NUMERIC,
    train_test_corr     NUMERIC,
    PRIMARY KEY (run_id, symbol, timeframe, lookback, threshold_mult)
);

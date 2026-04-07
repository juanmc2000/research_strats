-- Migration: 006_ctu_rolling_oos_tables.sql
-- Rolling walk-forward OOS tables for CTU locked signal.
-- Train = all events before test_year, test = calendar year slice.
-- Three threshold variants: baseline / narrow / wide.
-- Idempotent.

-- Per fold results — combined (all symbols) and per symbol
CREATE TABLE IF NOT EXISTS strategy_research.ctu_rolling_folds (
    run_id          UUID NOT NULL,
    threshold_set   TEXT NOT NULL,   -- 'baseline' | 'narrow' | 'wide'
    test_year       INT NOT NULL,
    scope           TEXT NOT NULL,   -- 'combined' | symbol name
    -- train = all qualifying events with year < test_year
    train_n         INT,
    train_mean_ret  NUMERIC,
    train_sharpe    NUMERIC,
    train_win_rate  NUMERIC,
    train_cum_ret   NUMERIC,
    -- test = qualifying events with year == test_year
    test_n          INT,
    test_mean_ret   NUMERIC,
    test_sharpe     NUMERIC,
    test_win_rate   NUMERIC,
    test_cum_ret    NUMERIC,
    test_max_dd     NUMERIC,
    -- bootstrap Sharpe CI on test (where n >= 10)
    boot_sharpe_p05 NUMERIC,
    boot_sharpe_p50 NUMERIC,
    boot_sharpe_p95 NUMERIC,
    PRIMARY KEY (run_id, threshold_set, test_year, scope)
);

-- Aggregate across all folds per threshold_set × scope
CREATE TABLE IF NOT EXISTS strategy_research.ctu_rolling_summary (
    run_id              UUID NOT NULL,
    threshold_set       TEXT NOT NULL,
    scope               TEXT NOT NULL,
    n_folds             INT,
    mean_test_sharpe    NUMERIC,
    median_test_sharpe  NUMERIC,
    std_test_sharpe     NUMERIC,
    min_test_sharpe     NUMERIC,
    max_test_sharpe     NUMERIC,
    pct_pos_folds       NUMERIC,
    mean_test_wr        NUMERIC,
    mean_test_cum       NUMERIC,
    mean_train_sharpe   NUMERIC,
    train_test_corr     NUMERIC,
    total_test_n        INT,
    PRIMARY KEY (run_id, threshold_set, scope)
);

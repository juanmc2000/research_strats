-- Migration: 007_etd_rolling_oos_tables.sql
-- Rolling walk-forward OOS tables for frozen ETD signal.
-- Two configs: raw_etd / atr_filtered_etd.
-- Symbols: USD/CHF, GBP/USD (as specified in Issue #22).
-- Idempotent.

CREATE TABLE IF NOT EXISTS strategy_research.etd_rolling_folds (
    run_id          UUID NOT NULL,
    symbol          TEXT NOT NULL,
    config          TEXT NOT NULL,   -- 'raw_etd' | 'atr_filtered_etd'
    test_year       INT NOT NULL,
    train_n         INT,
    train_mean_ret  NUMERIC,
    train_sharpe    NUMERIC,
    train_win_rate  NUMERIC,
    train_cum_ret   NUMERIC,
    test_n          INT,
    test_mean_ret   NUMERIC,
    test_sharpe     NUMERIC,
    test_win_rate   NUMERIC,
    test_cum_ret    NUMERIC,
    test_max_dd     NUMERIC,
    boot_sharpe_p05 NUMERIC,
    boot_sharpe_p50 NUMERIC,
    boot_sharpe_p95 NUMERIC,
    PRIMARY KEY (run_id, symbol, config, test_year)
);

CREATE TABLE IF NOT EXISTS strategy_research.etd_rolling_summary (
    run_id              UUID NOT NULL,
    symbol              TEXT NOT NULL,
    config              TEXT NOT NULL,
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
    PRIMARY KEY (run_id, symbol, config)
);

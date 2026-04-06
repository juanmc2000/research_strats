-- Migration 015: CTU symbol qualification tables
-- Issue #4: CTU Universe Qualification and Sleeve Construction
--
-- Creates three tables in the features schema for storing CTU universe
-- qualification outputs. All outputs are idempotent and use ON CONFLICT DO UPDATE.

-- ── features.ctu_symbol_qualification ─────────────────────────────────────────
-- One row per symbol per config_name. Stores per-split stats and the final
-- KEEP / MONITOR / EXCLUDE ruling with reason.

CREATE TABLE IF NOT EXISTS features.ctu_symbol_qualification (
    symbol                  VARCHAR(20)       NOT NULL,
    config_name             VARCHAR(40)       NOT NULL,
    status                  VARCHAR(10)       NOT NULL,    -- KEEP / MONITOR / EXCLUDE
    full_n                  INTEGER,
    train_n                 INTEGER,
    val_n                   INTEGER,
    fwd_n                   INTEGER,
    raw_full_sharpe         DOUBLE PRECISION,
    raw_fwd_sharpe          DOUBLE PRECISION,
    filtered_full_sharpe    DOUBLE PRECISION,
    filtered_fwd_sharpe     DOUBLE PRECISION,
    full_mean               DOUBLE PRECISION,
    fwd_mean                DOUBLE PRECISION,
    full_max_dd             DOUBLE PRECISION,
    fwd_max_dd              DOUBLE PRECISION,
    full_win_rate           DOUBLE PRECISION,
    fwd_win_rate            DOUBLE PRECISION,
    full_p_lt_neg1          DOUBLE PRECISION,
    fwd_p_lt_neg1           DOUBLE PRECISION,
    ruling_reason           TEXT,
    created_at              TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_symbol_qualification PRIMARY KEY (symbol, config_name)
);

-- ── features.ctu_symbol_cluster_analysis ──────────────────────────────────────
-- One row per (cluster_name, symbol, split_name). Stores cluster-level
-- aggregated metrics to identify whether CTU edge is family-specific.

CREATE TABLE IF NOT EXISTS features.ctu_symbol_cluster_analysis (
    cluster_name    VARCHAR(30)       NOT NULL,
    symbol          VARCHAR(20)       NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    p_lt_neg1       DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_symbol_cluster_analysis
        PRIMARY KEY (cluster_name, symbol, split_name)
);

-- ── features.ctu_universe_comparison ──────────────────────────────────────────
-- One row per (universe_name, split_name). Stores pooled performance for each
-- pre-declared universe to support forward-ranked universe selection.

CREATE TABLE IF NOT EXISTS features.ctu_universe_comparison (
    universe_name   VARCHAR(40)       NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    sharpe          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    p_lt_neg1       DOUBLE PRECISION,
    p_gt_pos1       DOUBLE PRECISION,
    cum_pnl         DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_universe_comparison
        PRIMARY KEY (universe_name, split_name)
);

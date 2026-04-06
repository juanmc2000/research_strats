-- Migration 016: CTU sleeve resolution tables
-- Issue #6: CTU Sleeve Resolution with Reconciliation Audit
--
-- Phase 0: reconciliation audit to diagnose whether the
-- universe-vs-sleeve contradiction is implementation or statistical.
-- Phase 1: sleeve candidate stats, symbol contributions, and role classification.

-- ── features.ctu_reconciliation_audit ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS features.ctu_reconciliation_audit (
    check_type      VARCHAR(40)       NOT NULL,
    symbol          VARCHAR(20),
    metric          VARCHAR(40)       NOT NULL,
    script_a        VARCHAR(60)       NOT NULL,
    script_b        VARCHAR(60)       NOT NULL,
    value_a         DOUBLE PRECISION,
    value_b         DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    status          VARCHAR(10)       NOT NULL,   -- PASS / FAIL / INFO
    note            TEXT,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_reconciliation_audit
        PRIMARY KEY (check_type, symbol, metric)
);

-- ── features.ctu_sleeve_candidate_stats ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS features.ctu_sleeve_candidate_stats (
    sleeve_name     VARCHAR(40)       NOT NULL,
    symbols_csv     TEXT              NOT NULL,
    split_name      VARCHAR(10)       NOT NULL,
    n_trades        INTEGER,
    mean_return     DOUBLE PRECISION,
    median_return   DOUBLE PRECISION,
    std_return      DOUBLE PRECISION,
    sharpe          DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    payoff          DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    p_lt_neg1       DOUBLE PRECISION,
    p_gt_pos1       DOUBLE PRECISION,
    cum_pnl         DOUBLE PRECISION,
    created_at      TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_sleeve_candidate_stats
        PRIMARY KEY (sleeve_name, split_name)
);

-- ── features.ctu_sleeve_symbol_contributions ──────────────────────────────────
CREATE TABLE IF NOT EXISTS features.ctu_sleeve_symbol_contributions (
    sleeve_name                 VARCHAR(40)       NOT NULL,
    split_name                  VARCHAR(10)       NOT NULL,
    symbol                      VARCHAR(20)       NOT NULL,
    n_trades                    INTEGER,
    pnl_contribution_pct        DOUBLE PRECISION,
    variance_contribution_pct   DOUBLE PRECISION,
    drawdown_contribution_pct   DOUBLE PRECISION,
    created_at                  TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_sleeve_symbol_contributions
        PRIMARY KEY (sleeve_name, split_name, symbol)
);

-- ── features.ctu_symbol_role_classification ───────────────────────────────────
CREATE TABLE IF NOT EXISTS features.ctu_symbol_role_classification (
    symbol                      VARCHAR(20)       NOT NULL,
    prior_status                VARCHAR(10),
    revised_status              VARCHAR(10),
    role_label                  VARCHAR(30),
    full_sharpe                 DOUBLE PRECISION,
    fwd_sharpe                  DOUBLE PRECISION,
    fwd_n                       INTEGER,
    left_tail_fwd               DOUBLE PRECISION,
    sleeve_compatibility_flag   BOOLEAN,
    rationale                   TEXT,
    created_at                  TIMESTAMPTZ       NOT NULL DEFAULT now(),
    CONSTRAINT pk_ctu_symbol_role_classification
        PRIMARY KEY (symbol)
);

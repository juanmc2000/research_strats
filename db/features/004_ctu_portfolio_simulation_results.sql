CREATE TABLE IF NOT EXISTS features.ctu_portfolio_simulation_results (
    id BIGSERIAL PRIMARY KEY,
    portfolio_name TEXT NOT NULL,
    split_name TEXT NOT NULL,
    n_trades INTEGER,
    mean_return DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    payoff DOUBLE PRECISION,
    p_lt_neg1 DOUBLE PRECISION,
    p_gt_pos1 DOUBLE PRECISION,
    concentration_metric DOUBLE PRECISION,
    verdict TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

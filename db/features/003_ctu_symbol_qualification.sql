CREATE TABLE IF NOT EXISTS features.ctu_symbol_qualification (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    pooled_mean DOUBLE PRECISION,
    pooled_sharpe DOUBLE PRECISION,
    forward_mean DOUBLE PRECISION,
    forward_sharpe DOUBLE PRECISION,
    cost_adjusted_mean DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    median_return DOUBLE PRECISION,
    ruling_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

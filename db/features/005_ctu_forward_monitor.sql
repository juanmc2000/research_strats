CREATE TABLE IF NOT EXISTS features.ctu_forward_monitor (
    id BIGSERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    candidate_state TEXT NOT NULL,
    symbol TEXT,
    split_name TEXT NOT NULL,
    n_events INTEGER,
    mean_return DOUBLE PRECISION,
    sharpe DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    win_rate DOUBLE PRECISION,
    verdict TEXT,
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

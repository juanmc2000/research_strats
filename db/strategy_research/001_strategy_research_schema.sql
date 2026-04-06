-- Migration: 001_strategy_research_schema.sql
-- Creates the strategy_research schema and shared experiment_runs table.
-- Idempotent: safe to run multiple times.

-- Schema must be created by a superuser before running this migration.
-- CREATE SCHEMA IF NOT EXISTS strategy_research;

CREATE TABLE IF NOT EXISTS strategy_research.experiment_runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name   TEXT NOT NULL,
    run_ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    config          JSONB,
    notes           TEXT
);

COMMENT ON TABLE strategy_research.experiment_runs IS
    'Registry of all research experiment runs across all isolated strategies.';

# ETD ARCHIVE — reference only. Do not modify without a new issue.
#!/usr/bin/env python3
"""
markov_etd_sleeve_portfolio.py
Issue #100: ETD Sleeve Portfolio — Cross-Symbol Filter Validation (FX only)

Advances ETD from single-variant (USD/CHF) testing to portfolio-level
qualification by:

  1. Constructing separate ETD sleeves across all 7 core G10 FX pairs
  2. Validating which filters are genuinely portable across symbols
  3. Isolating USD/CHF-specific from generic ETD logic
  4. Measuring sleeve correlations and diversification
  5. Testing an equal-weight minimal ETD portfolio
  6. Writing a decision memo with filter portability classification

Sleeve definitions
------------------
  A_raw            : no filter — all ETD trades
  B_generic        : ATR-transition filter only (portable candidate)
  C_usdchf_specific: ATR-transition + shock filter (USD/CHF-specific candidate)
  D_portfolio      : equal-weight combination of kept sleeves

Filter definitions (identical thresholds to Issue #99 — no re-optimisation)
----------------------------------------------------------------------------
  Shock filter      : shock_1h_sd < -3.0
  ATR-transition    : atr_bucket == TRANSITION_1_2  (z-score in [1, 2))
  Combined keep     : NOT (shock OR atr_transition)

Scope
-----
  FX symbols only.
  TimescaleDB research schema only.
  No production DB writes.
  No parquet outputs.

Usage
-----
  python scripts/markov_etd_sleeve_portfolio.py \\
      --output results/etd_sleeve_portfolio.csv \\
      --memo   results/etd_sleeve_portfolio_memo.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from typing import Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── FX symbol universe ─────────────────────────────────────────────────────────

FX_SYMBOLS: list[str] = [
    'EUR/USD', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
]
USD_CHF: str = 'USD/CHF'

# ── Constants (identical to Issue #99 — no re-optimisation) ───────────────────

MEDIUM_COST: float = 0.07
ATR_WINDOW: int = 30

BUCKET_LOW: str = 'LOW'
BUCKET_NORMAL: str = 'NORMAL'
BUCKET_TRANSITION: str = 'TRANSITION_1_2'
BUCKET_SPIKE: str = 'SPIKE_GE_2'

SHOCK_THRESHOLD: float = -3.0
VOL_EXPANDING_MIN: float = 1.20
EFF_TRENDING_MIN: float = 0.60

TRAIN_END_YEAR: int = 2022
VAL_END_YEAR: int = 2024

SLEEVE_NAMES: list[str] = ['A_raw', 'B_generic', 'C_usdchf_specific']

# Filter portability thresholds
GENERIC_SHARPE_MIN_SYMBOLS: int = 3   # must improve >= this many symbols
GENERIC_DEGRADE_MAX_SYMBOLS: int = 2  # must not degrade more than this many
GENERIC_FWD_SHARPE_DELTA_MIN: float = -0.05  # fwd Sharpe must not fall more than this


# ── Database ───────────────────────────────────────────────────────────────────

def connect() -> psycopg2.extensions.connection:
    load_dotenv(dotenv_path='.env')
    return psycopg2.connect(os.environ['TIMESCALE_DSN'])


# ── DDL ────────────────────────────────────────────────────────────────────────

DDL_SLEEVE_RETURNS = """
CREATE TABLE IF NOT EXISTS features.etd_sleeve_returns (
    symbol                     VARCHAR(20)   NOT NULL,
    event_hour_ts              TIMESTAMPTZ   NOT NULL,
    breakout_direction         VARCHAR(4)    NOT NULL,
    sleeve_name                VARCHAR(24)   NOT NULL,
    is_kept                    BOOLEAN       NOT NULL,
    realized_return_sd30d      DOUBLE PRECISION,
    excess_return_sd30d        DOUBLE PRECISION,
    cost_adjusted_return_sd30d DOUBLE PRECISION,
    created_at                 TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ   NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_sleeve_returns
        PRIMARY KEY (symbol, event_hour_ts, breakout_direction, sleeve_name)
);
"""

DDL_CROSS_SYMBOL_VALIDATION = """
CREATE TABLE IF NOT EXISTS features.etd_cross_symbol_filter_validation (
    symbol          VARCHAR(20)      NOT NULL,
    config_name     VARCHAR(20)      NOT NULL,
    split_name      VARCHAR(10)      NOT NULL,
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
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT pk_etd_cross_symbol_filter_validation
        PRIMARY KEY (symbol, config_name, split_name)
);
"""


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL_SLEEVE_RETURNS)
        cur.execute(DDL_CROSS_SYMBOL_VALIDATION)
        try:
            cur.execute("""
                SELECT create_hypertable(
                    'features.etd_sleeve_returns',
                    'event_hour_ts',
                    chunk_time_interval => INTERVAL '180 days',
                    if_not_exists => TRUE
                )
            """)
        except Exception:
            pass
    conn.commit()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_daily_metrics(symbol: str) -> list[dict]:
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT day_ts, daily_range AS atr_daily
            FROM features.daily_market_metrics
            WHERE symbol = %s
              AND daily_range IS NOT NULL
              AND daily_range > 0
            ORDER BY day_ts
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def load_etd_trades(symbol: str) -> list[dict]:
    conn = connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                be.symbol,
                be.event_hour_ts,
                be.breakout_direction,
                be.vol_ratio_20_100,
                be.efficiency_20,
                be.shock_1h_sd,
                be.realized_return_sd30d,
                be.max_favorable_excursion_sd30d,
                be.max_adverse_excursion_sd30d,
                be.bars_held,
                be.exit_reason,
                be.entry_range_sd_30d
            FROM features.breakout_events be
            WHERE be.symbol = %s
              AND be.exit_reason IS NOT NULL
              AND be.entry_range_sd_30d > 0
              AND be.vol_ratio_20_100 IS NOT NULL
              AND be.efficiency_20 IS NOT NULL
              AND be.realized_return_sd30d IS NOT NULL
              AND be.shock_1h_sd IS NOT NULL
            ORDER BY be.event_hour_ts
        """, (symbol,))
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── ATR context ────────────────────────────────────────────────────────────────

def compute_atr_context(daily_metrics: list[dict]) -> dict:
    """Return date -> atr_bucket mapping."""
    result: dict = {}
    for i, row in enumerate(daily_metrics):
        if i < ATR_WINDOW - 1:
            continue
        window = [float(daily_metrics[j]['atr_daily']) for j in range(i - ATR_WINDOW + 1, i + 1)]
        mean = sum(window) / ATR_WINDOW
        variance = sum((v - mean) ** 2 for v in window) / (ATR_WINDOW - 1)
        std = math.sqrt(variance)
        if std == 0.0:
            continue
        zscore = (float(row['atr_daily']) - mean) / std
        bucket = _atr_bucket(zscore)
        result[row['day_ts']] = {'atr_zscore': zscore, 'atr_bucket': bucket}
    return result


def _atr_bucket(zscore: float) -> str:
    if zscore < 0.0:
        return BUCKET_LOW
    if zscore < 1.0:
        return BUCKET_NORMAL
    if zscore < 2.0:
        return BUCKET_TRANSITION
    return BUCKET_SPIKE


# ── State classification ───────────────────────────────────────────────────────

def _classify_state(vr: float, ef: float, direction: str) -> str:
    vol = 'EXPANDING' if vr > VOL_EXPANDING_MIN else ('NEUTRAL' if vr >= 0.80 else 'CONTRACTING')
    eff = 'TRENDING' if ef > EFF_TRENDING_MIN else ('MIXED' if ef >= 0.30 else 'CHOPPY')
    return f'{vol}_{eff}_{direction}'


# ── Per-symbol ETD trade building ──────────────────────────────────────────────

def build_etd_trades(symbol: str) -> list[dict]:
    """
    Load and annotate ETD trades for a symbol.
    ETD state: EXPANDING_TRENDING_DOWN only.
    Returns list of annotated trade dicts with filter flags.
    """
    raw_daily = load_daily_metrics(symbol)
    atr_lookup = compute_atr_context(raw_daily)
    raw_trades = load_etd_trades(symbol)

    etd_trades = []
    for tr in raw_trades:
        vr = float(tr['vol_ratio_20_100'])
        ef = float(tr['efficiency_20'])
        direction = str(tr['breakout_direction'])
        state = _classify_state(vr, ef, direction)

        if state != 'EXPANDING_TRENDING_DOWN':
            continue

        ts = tr['event_hour_ts']
        trade_date = ts.date() if hasattr(ts, 'date') else ts
        atr_ctx = atr_lookup.get(trade_date)

        shock_val = float(tr['shock_1h_sd'])
        eff_val = float(tr['efficiency_20'])
        atr_zscore = atr_ctx['atr_zscore'] if atr_ctx else None
        atr_bucket = atr_ctx['atr_bucket'] if atr_ctx else None

        fail_shock = shock_val < SHOCK_THRESHOLD
        fail_atr = atr_bucket == BUCKET_TRANSITION
        keep_generic = not fail_atr
        keep_usdchf = not (fail_shock or fail_atr)

        excess = float(tr['realized_return_sd30d']) - MEDIUM_COST
        year = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])

        etd_trades.append(dict(
            symbol=symbol,
            event_hour_ts=ts,
            breakout_direction=direction,
            realized_return_sd30d=float(tr['realized_return_sd30d']),
            excess=excess,
            year=year,
            shock_val=shock_val,
            eff_val=eff_val,
            atr_zscore=atr_zscore,
            atr_bucket=atr_bucket,
            fail_shock=fail_shock,
            fail_atr=fail_atr,
            keep_generic=keep_generic,
            keep_usdchf=keep_usdchf,
        ))
    return etd_trades


# ── Sleeve membership ──────────────────────────────────────────────────────────

def is_kept_in_sleeve(trade: dict, sleeve: str) -> bool:
    if sleeve == 'A_raw':
        return True
    if sleeve == 'B_generic':
        return trade['keep_generic']
    if sleeve == 'C_usdchf_specific':
        return trade['keep_usdchf']
    raise ValueError(f'unknown sleeve: {sleeve}')


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> Optional[dict]:
    vals = [tr['excess'] for tr in trades]
    n = len(vals)
    if n < 2:
        return None
    mn = sum(vals) / n
    sv = sorted(vals)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2.0
    sd = statistics.stdev(vals)
    sharpe = mn / sd if sd > 0 else 0.0
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss != 0.0 else 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        cum += v
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    p_lt_neg1 = sum(1 for v in vals if v < -1.0) / n
    p_gt_pos1 = sum(1 for v in vals if v > 1.0) / n
    return dict(
        n=n, mean=mn, median=med, std=sd, sharpe=sharpe,
        win_rate=win_rate, payoff=payoff, cum_pnl=cum,
        max_dd=max_dd, p_lt_neg1=p_lt_neg1, p_gt_pos1=p_gt_pos1,
    )


def split_trades(trades: list[dict]) -> dict[str, list[dict]]:
    return {
        'full':  trades,
        'train': [tr for tr in trades if tr['year'] < TRAIN_END_YEAR],
        'val':   [tr for tr in trades if TRAIN_END_YEAR <= tr['year'] < VAL_END_YEAR],
        'fwd':   [tr for tr in trades if tr['year'] >= VAL_END_YEAR],
    }


def config_to_sleeve(config: str) -> Optional[str]:
    return {
        'baseline': 'A_raw',
        'atr_only': 'B_generic',
        'combined': 'C_usdchf_specific',
        'shock_only': None,  # computed inline
    }.get(config)


def apply_config(trades: list[dict], config: str) -> list[dict]:
    if config == 'baseline':
        return trades
    if config == 'atr_only':
        return [tr for tr in trades if not tr['fail_atr']]
    if config == 'shock_only':
        return [tr for tr in trades if not tr['fail_shock']]
    if config == 'combined':
        return [tr for tr in trades if tr['keep_usdchf']]
    raise ValueError(f'unknown config: {config}')


# ── TimescaleDB upserts ────────────────────────────────────────────────────────

def upsert_sleeve_returns(all_trades: dict[str, list[dict]]) -> int:
    """Upsert sleeve-level trade rows from per-symbol trade lists."""
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for symbol, trades in all_trades.items():
            for trade in trades:
                rr = trade['realized_return_sd30d']
                excess = trade['excess']
                for sleeve in SLEEVE_NAMES:
                    kept = is_kept_in_sleeve(trade, sleeve)
                    cur.execute("""
                        INSERT INTO features.etd_sleeve_returns (
                            symbol, event_hour_ts, breakout_direction,
                            sleeve_name, is_kept,
                            realized_return_sd30d, excess_return_sd30d,
                            cost_adjusted_return_sd30d, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (symbol, event_hour_ts, breakout_direction, sleeve_name)
                        DO UPDATE SET
                            is_kept                    = EXCLUDED.is_kept,
                            realized_return_sd30d      = EXCLUDED.realized_return_sd30d,
                            excess_return_sd30d        = EXCLUDED.excess_return_sd30d,
                            cost_adjusted_return_sd30d = EXCLUDED.cost_adjusted_return_sd30d,
                            updated_at                 = now()
                    """, (
                        symbol,
                        trade['event_hour_ts'],
                        trade['breakout_direction'],
                        sleeve,
                        kept,
                        rr if kept else None,
                        excess if kept else None,
                        excess if kept else None,
                    ))
                    count += 1
    conn.commit()
    conn.close()
    return count


def upsert_cross_symbol_validation(rows: list[dict]) -> int:
    conn = connect()
    count = 0
    with conn.cursor() as cur:
        for r in rows:
            cur.execute("""
                INSERT INTO features.etd_cross_symbol_filter_validation (
                    symbol, config_name, split_name,
                    n_trades, mean_return, median_return, std_return,
                    sharpe, win_rate, payoff, max_drawdown,
                    p_lt_neg1, p_gt_pos1, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (symbol, config_name, split_name)
                DO UPDATE SET
                    n_trades     = EXCLUDED.n_trades,
                    mean_return  = EXCLUDED.mean_return,
                    median_return= EXCLUDED.median_return,
                    std_return   = EXCLUDED.std_return,
                    sharpe       = EXCLUDED.sharpe,
                    win_rate     = EXCLUDED.win_rate,
                    payoff       = EXCLUDED.payoff,
                    max_drawdown = EXCLUDED.max_drawdown,
                    p_lt_neg1    = EXCLUDED.p_lt_neg1,
                    p_gt_pos1    = EXCLUDED.p_gt_pos1,
                    updated_at   = now()
            """, (
                r['symbol'], r['config_name'], r['split_name'],
                r.get('n_trades'), r.get('mean_return'), r.get('median_return'),
                r.get('std_return'), r.get('sharpe'), r.get('win_rate'),
                r.get('payoff'), r.get('max_drawdown'),
                r.get('p_lt_neg1'), r.get('p_gt_pos1'),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


# ── Correlation ────────────────────────────────────────────────────────────────

def _corr(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def compute_sleeve_correlation(
    all_trades: dict[str, list[dict]],
) -> dict[tuple[str, str], Optional[float]]:
    """
    Compute return correlation between sleeve A and sleeve B pooled across all symbols.
    Uses only trades that are in both sleeves (A is all trades, so B/C subset of A).
    Correlation is computed over the raw (excess) return series sorted by event_hour_ts.
    """
    # Build time-keyed return series per sleeve (pooled)
    returns_a: list[tuple] = []
    returns_b: list[tuple] = []
    returns_c: list[tuple] = []

    for trades in all_trades.values():
        for tr in trades:
            key = (tr['event_hour_ts'], tr['symbol'])
            returns_a.append((key, tr['excess']))
            if tr['keep_generic']:
                returns_b.append((key, tr['excess']))
            if tr['keep_usdchf']:
                returns_c.append((key, tr['excess']))

    # For A vs B: compute correlation using only trades in B (subset of A)
    a_keys = {k: v for k, v in returns_a}
    b_keys = {k: v for k, v in returns_b}
    c_keys = {k: v for k, v in returns_c}

    # A vs B correlation (over B's subset)
    shared_ab = sorted(b_keys.keys())
    xs_ab = [a_keys[k] for k in shared_ab if k in a_keys]
    ys_ab = [b_keys[k] for k in shared_ab if k in a_keys]
    corr_ab = _corr(xs_ab, ys_ab) if len(xs_ab) >= 2 else None

    # A vs C correlation (over C's subset)
    shared_ac = sorted(c_keys.keys())
    xs_ac = [a_keys[k] for k in shared_ac if k in a_keys]
    ys_ac = [c_keys[k] for k in shared_ac if k in a_keys]
    corr_ac = _corr(xs_ac, ys_ac) if len(xs_ac) >= 2 else None

    # B vs C correlation (over intersection)
    shared_bc = sorted(k for k in b_keys if k in c_keys)
    xs_bc = [b_keys[k] for k in shared_bc]
    ys_bc = [c_keys[k] for k in shared_bc]
    corr_bc = _corr(xs_bc, ys_bc) if len(xs_bc) >= 2 else None

    return {
        ('A_raw', 'B_generic'): corr_ab,
        ('A_raw', 'C_usdchf_specific'): corr_ac,
        ('B_generic', 'C_usdchf_specific'): corr_bc,
    }


# ── Equal-weight portfolio ─────────────────────────────────────────────────────

def build_equal_weight_portfolio(
    symbol_trades: dict[str, list[dict]],
    usdchf_sleeve: str,
    other_sleeve: str,
) -> list[dict]:
    """
    Construct an equal-weight ETD portfolio.
    - USD/CHF: uses usdchf_sleeve
    - Other symbols: uses other_sleeve
    Returns a combined list of trade dicts with portfolio_weight = 1/N symbols active.
    """
    combined = []
    n_symbols = len(symbol_trades)
    weight = 1.0 / n_symbols if n_symbols > 0 else 1.0
    for symbol, trades in symbol_trades.items():
        sleeve = usdchf_sleeve if symbol == USD_CHF else other_sleeve
        for tr in trades:
            if is_kept_in_sleeve(tr, sleeve):
                entry = dict(tr)
                entry['excess'] = tr['excess'] * weight
                combined.append(entry)
    combined.sort(key=lambda x: x['event_hour_ts'])
    return combined


# ── Output helpers ─────────────────────────────────────────────────────────────

def fmt(val: Optional[float], decimals: int = 3) -> str:
    if val is None:
        return 'N/A'
    return f'{val:.{decimals}f}'


def stats_csv_row(
    symbol: str, config: str, split: str, s: Optional[dict],
) -> dict:
    base = dict(symbol=symbol, config=config, split=split)
    if s is None:
        return {**base, 'n': '', 'mean': '', 'median': '', 'std': '',
                'sharpe': '', 'win_rate': '', 'payoff': '', 'max_dd': '',
                'p_lt_neg1': '', 'p_gt_pos1': ''}
    return {
        **base,
        'n': s['n'],
        'mean': fmt(s['mean']),
        'median': fmt(s['median']),
        'std': fmt(s['std']),
        'sharpe': fmt(s['sharpe']),
        'win_rate': fmt(s['win_rate']),
        'payoff': fmt(s['payoff']),
        'max_dd': fmt(s['max_dd']),
        'p_lt_neg1': fmt(s['p_lt_neg1']),
        'p_gt_pos1': fmt(s['p_gt_pos1']),
    }


def stats_db_row(symbol: str, config: str, split: str, s: Optional[dict]) -> dict:
    if s is None:
        return dict(symbol=symbol, config_name=config, split_name=split)
    return dict(
        symbol=symbol, config_name=config, split_name=split,
        n_trades=s['n'],
        mean_return=s['mean'],
        median_return=s['median'],
        std_return=s['std'],
        sharpe=s['sharpe'],
        win_rate=s['win_rate'],
        payoff=s['payoff'],
        max_drawdown=s['max_dd'],
        p_lt_neg1=s['p_lt_neg1'],
        p_gt_pos1=s['p_gt_pos1'],
    )


# ── Filter portability classification ─────────────────────────────────────────

def classify_filter_portability(
    per_symbol_stats: dict[str, dict[str, Optional[dict]]],
) -> dict[str, str]:
    """
    Classify ATR-transition and shock filters as generic or symbol-specific.

    ATR-transition (atr_only vs baseline):
      Generic if improves Sharpe on >= GENERIC_SHARPE_MIN_SYMBOLS
      and does not degrade more than GENERIC_DEGRADE_MAX_SYMBOLS.

    Shock filter (shock_only vs baseline):
      Same criterion.
    """
    atr_improve = 0
    atr_degrade = 0
    shock_improve = 0
    shock_degrade = 0

    for sym, stats in per_symbol_stats.items():
        base = stats.get('baseline')
        atr = stats.get('atr_only')
        shock = stats.get('shock_only')

        if base and atr:
            delta = atr['sharpe'] - base['sharpe']
            if delta > 0.01:
                atr_improve += 1
            elif delta < -0.01:
                atr_degrade += 1

        if base and shock:
            delta = shock['sharpe'] - base['sharpe']
            if delta > 0.01:
                shock_improve += 1
            elif delta < -0.01:
                shock_degrade += 1

    atr_label = (
        'GENERIC'
        if atr_improve >= GENERIC_SHARPE_MIN_SYMBOLS and atr_degrade <= GENERIC_DEGRADE_MAX_SYMBOLS
        else 'SYMBOL_SPECIFIC'
    )
    shock_label = (
        'GENERIC'
        if shock_improve >= GENERIC_SHARPE_MIN_SYMBOLS and shock_degrade <= GENERIC_DEGRADE_MAX_SYMBOLS
        else 'SYMBOL_SPECIFIC'
    )

    return {
        'atr_transition': atr_label,
        'shock': shock_label,
        'atr_improve_count': atr_improve,
        'atr_degrade_count': atr_degrade,
        'shock_improve_count': shock_improve,
        'shock_degrade_count': shock_degrade,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/etd_sleeve_portfolio.csv')
    parser.add_argument('--memo',    default='results/etd_sleeve_portfolio_memo.txt')
    parser.add_argument('--exclude', nargs='*', default=[], metavar='SYMBOL',
                        help='Symbols to exclude, e.g. --exclude USD/CAD')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    symbols = [s for s in FX_SYMBOLS if s not in args.exclude]

    # ── Create tables ──────────────────────────────────────────────────────────
    conn = connect()
    create_tables(conn)
    conn.close()

    # ── Load and annotate ETD trades per FX symbol ────────────────────────────
    print(f'Loading ETD trades for {len(symbols)} FX symbols (excluded: {args.exclude or "none"})...')
    all_trades: dict[str, list[dict]] = {}
    coverage: dict[str, dict] = {}

    for symbol in symbols:
        trades = build_etd_trades(symbol)
        all_trades[symbol] = trades
        n_shock = sum(1 for tr in trades if tr['fail_shock'])
        n_atr = sum(1 for tr in trades if tr['fail_atr'])
        n_keep_b = sum(1 for tr in trades if tr['keep_generic'])
        n_keep_c = sum(1 for tr in trades if tr['keep_usdchf'])
        coverage[symbol] = dict(
            n_total=len(trades),
            n_shock=n_shock,
            n_atr=n_atr,
            n_keep_b=n_keep_b,
            n_keep_c=n_keep_c,
        )
        pct_b = n_keep_b / len(trades) * 100 if trades else 0.0
        pct_c = n_keep_c / len(trades) * 100 if trades else 0.0
        print(
            f'  {symbol:<10} etd={len(trades):>4}  shock={n_shock:>3}  atr={n_atr:>3}'
            f'  keep_B={n_keep_b:>4} ({pct_b:.0f}%)  keep_C={n_keep_c:>4} ({pct_c:.0f}%)'
        )

    # ── Upsert sleeve returns ──────────────────────────────────────────────────
    print('\nUpserting etd_sleeve_returns...')
    sleeve_rows_written = upsert_sleeve_returns(all_trades)
    print(f'  {sleeve_rows_written} rows written')

    # ── Cross-symbol validation stats ─────────────────────────────────────────
    configs = ['baseline', 'shock_only', 'atr_only', 'combined']
    splits = ['full', 'train', 'val', 'fwd']

    per_symbol_stats: dict[str, dict[str, Optional[dict]]] = {}
    db_rows: list[dict] = []
    csv_rows: list[dict] = []
    fieldnames = ['symbol', 'config', 'split', 'n', 'mean', 'median', 'std',
                  'sharpe', 'win_rate', 'payoff', 'max_dd', 'p_lt_neg1', 'p_gt_pos1']

    for symbol in symbols:
        trades = all_trades[symbol]
        sym_stats: dict[str, Optional[dict]] = {}
        for cfg in configs:
            cfg_trades = apply_config(trades, cfg)
            sp = split_trades(cfg_trades)
            for sp_name, sp_trades in sp.items():
                s = compute_stats(sp_trades)
                if sp_name == 'full':
                    sym_stats[cfg] = s
                csv_rows.append(stats_csv_row(symbol, cfg, sp_name, s))
                db_rows.append(stats_db_row(symbol, cfg, sp_name, s))
        per_symbol_stats[symbol] = sym_stats

    # Pooled FX stats
    all_fx = [tr for trades in all_trades.values() for tr in trades]
    for cfg in configs:
        cfg_trades = apply_config(all_fx, cfg)
        sp = split_trades(cfg_trades)
        for sp_name, sp_trades in sp.items():
            s = compute_stats(sp_trades)
            csv_rows.append(stats_csv_row('_POOLED_FX', cfg, sp_name, s))
            db_rows.append(stats_db_row('_POOLED_FX', cfg, sp_name, s))

    print('\nUpserting etd_cross_symbol_filter_validation...')
    val_rows_written = upsert_cross_symbol_validation(db_rows)
    print(f'  {val_rows_written} rows written')

    with open(args.output, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_rows)

    # ── Sleeve correlation ─────────────────────────────────────────────────────
    corr_map = compute_sleeve_correlation(all_trades)

    # ── Portfolio tests ────────────────────────────────────────────────────────
    # Portfolio 1: USD/CHF uses C_usdchf_specific, others use B_generic
    port1 = build_equal_weight_portfolio(all_trades, 'C_usdchf_specific', 'B_generic')
    port1_sp = split_trades(port1)
    port1_stats = {sp_name: compute_stats(sp_trades) for sp_name, sp_trades in port1_sp.items()}

    # Portfolio 2: All symbols use A_raw (benchmark)
    port2 = build_equal_weight_portfolio(all_trades, 'A_raw', 'A_raw')
    port2_sp = split_trades(port2)
    port2_stats = {sp_name: compute_stats(sp_trades) for sp_name, sp_trades in port2_sp.items()}

    # Portfolio 3: All symbols use B_generic
    port3 = build_equal_weight_portfolio(all_trades, 'B_generic', 'B_generic')
    port3_sp = split_trades(port3)
    port3_stats = {sp_name: compute_stats(sp_trades) for sp_name, sp_trades in port3_sp.items()}

    # ── Filter portability classification ─────────────────────────────────────
    portability = classify_filter_portability(per_symbol_stats)

    # ── Determine recommendation ───────────────────────────────────────────────
    atr_generic = portability['atr_transition'] == 'GENERIC'
    shock_generic = portability['shock'] == 'GENERIC'

    pooled_base_full = compute_stats(apply_config(all_fx, 'baseline'))
    pooled_atr_full = compute_stats(apply_config(all_fx, 'atr_only'))
    pooled_comb_full = compute_stats(apply_config(all_fx, 'combined'))
    pooled_atr_fwd = compute_stats([tr for tr in apply_config(all_fx, 'atr_only')
                                    if tr['year'] >= VAL_END_YEAR])
    pooled_base_fwd = compute_stats([tr for tr in all_fx if tr['year'] >= VAL_END_YEAR])

    atr_sharpe_delta = (
        (pooled_atr_full['sharpe'] - pooled_base_full['sharpe'])
        if pooled_atr_full and pooled_base_full else None
    )
    atr_fwd_sharpe_delta = (
        (pooled_atr_fwd['sharpe'] - pooled_base_fwd['sharpe'])
        if pooled_atr_fwd and pooled_base_fwd else None
    )

    if atr_generic and (atr_fwd_sharpe_delta is not None and atr_fwd_sharpe_delta >= GENERIC_FWD_SHARPE_DELTA_MIN):
        if shock_generic:
            recommendation = 'A'  # promote as multi-symbol with generic filter; shock also generic
        else:
            recommendation = 'B'  # generic ATR sleeve + USD/CHF-specific sleeve
    elif not atr_generic:
        recommendation = 'C'  # keep USD/CHF ETD only; reject broad generalization
    else:
        recommendation = 'D'  # reject filtered ETD; keep raw ETD only

    rec_text = {
        'A': 'Promote ETD as multi-symbol strategy using generic ATR-transition filter (shock also generic)',
        'B': 'Promote ETD as two separate sleeves: generic ATR sleeve + USD/CHF-specialized shock+ATR sleeve',
        'C': 'Keep USD/CHF ETD only; reject broad ETD filter generalization',
        'D': 'Reject filtered ETD; keep raw ETD only across all symbols',
    }[recommendation]

    # ── Memo ───────────────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append('=' * 76)
    lines.append('ETD SLEEVE PORTFOLIO — CROSS-SYMBOL FILTER VALIDATION  (Issue #100)')
    lines.append('FX only. No threshold re-optimisation. TimescaleDB features schema only.')
    lines.append('=' * 76)
    lines.append('')
    lines.append('FILTER DEFINITIONS (identical thresholds to Issue #99)')
    lines.append(f'  Shock filter:      shock_1h_sd < {SHOCK_THRESHOLD}')
    lines.append(f'  ATR-transition:    atr_bucket == TRANSITION_1_2 (z in [1,2))')
    lines.append(f'  Combined keep:     NOT (shock OR atr_transition)')
    lines.append('')
    lines.append('SLEEVE DEFINITIONS')
    lines.append('  A_raw              : all ETD trades')
    lines.append('  B_generic          : ATR-transition filter only')
    lines.append('  C_usdchf_specific  : ATR-transition + shock filter')
    lines.append('')

    # Coverage table
    lines.append('FILTER COVERAGE PER FX SYMBOL')
    lines.append(
        f'  {"Symbol":<12} {"etd_n":>6} {"shock_f":>8} {"atr_f":>7}'
        f' {"keep_B":>7} {"B%":>5} {"keep_C":>7} {"C%":>5}'
    )
    lines.append(
        f'  {"-"*12} {"-"*6} {"-"*8} {"-"*7} {"-"*7} {"-"*5} {"-"*7} {"-"*5}'
    )
    for sym in symbols:
        c = coverage[sym]
        pct_b = c['n_keep_b'] / c['n_total'] * 100 if c['n_total'] else 0.0
        pct_c = c['n_keep_c'] / c['n_total'] * 100 if c['n_total'] else 0.0
        lines.append(
            f'  {sym:<12} {c["n_total"]:>6} {c["n_shock"]:>8} {c["n_atr"]:>7}'
            f' {c["n_keep_b"]:>7} {pct_b:>4.0f}% {c["n_keep_c"]:>7} {pct_c:>4.0f}%'
        )
    lines.append('')

    # Per-symbol cross-symbol validation
    lines.append('CROSS-SYMBOL ETD PERFORMANCE (full period, Sharpe)')
    lines.append(
        f'  {"Symbol":<12} {"n_base":>7} {"base_sh":>8} {"atr_sh":>8}'
        f' {"shock_sh":>9} {"comb_sh":>8} {"atr_delta":>10} {"shock_delta":>12}'
    )
    lines.append(
        f'  {"-"*12} {"-"*7} {"-"*8} {"-"*8} {"-"*9} {"-"*8} {"-"*10} {"-"*12}'
    )
    for sym in symbols:
        stats = per_symbol_stats[sym]
        base = stats.get('baseline')
        atr = stats.get('atr_only')
        shock = stats.get('shock_only')
        comb = stats.get('combined')
        base_n = base['n'] if base else 0
        base_sh = fmt(base['sharpe'] if base else None)
        atr_sh = fmt(atr['sharpe'] if atr else None)
        shock_sh = fmt(shock['sharpe'] if shock else None)
        comb_sh = fmt(comb['sharpe'] if comb else None)
        atr_d = fmt((atr['sharpe'] - base['sharpe']) if atr and base else None, 3)
        shock_d = fmt((shock['sharpe'] - base['sharpe']) if shock and base else None, 3)
        lines.append(
            f'  {sym:<12} {base_n:>7} {base_sh:>8} {atr_sh:>8}'
            f' {shock_sh:>9} {comb_sh:>8} {atr_d:>10} {shock_d:>12}'
        )
    lines.append('')

    # Pooled FX performance
    lines.append(f'POOLED FX PERFORMANCE (all {len(symbols)} FX symbols combined)')
    lines.append(
        f'  {"Config":<12} {"split":>6} {"n":>5} {"Sharpe":>8}'
        f' {"MaxDD":>8} {"P<-1":>7} {"WinR":>7} {"CumPnL":>9}'
    )
    lines.append(
        f'  {"-"*12} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*9}'
    )
    for cfg in configs:
        cfg_trades = apply_config(all_fx, cfg)
        for sp_name in splits:
            sp_trades = split_trades(cfg_trades)[sp_name]
            s = compute_stats(sp_trades)
            if s:
                lines.append(
                    f'  {cfg:<12} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["p_lt_neg1"]:>7.3f} {s["win_rate"]:>7.3f}'
                    f' {s["cum_pnl"]:>9.3f}'
                )
            else:
                lines.append(f'  {cfg:<12} {sp_name:>6}   N/A')
    lines.append('')

    # USD/CHF sleeve comparison
    lines.append('USD/CHF SLEEVE COMPARISON')
    usdchf_trades = all_trades.get(USD_CHF, [])
    usdchf_configs = ['baseline', 'atr_only', 'shock_only', 'combined']
    lines.append(
        f'  {"Config":<12} {"split":>6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"P<-1":>7}'
    )
    lines.append(f'  {"-"*12} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7}')
    for cfg in usdchf_configs:
        cfg_trades = apply_config(usdchf_trades, cfg)
        for sp_name in splits:
            sp_trades = split_trades(cfg_trades)[sp_name]
            s = compute_stats(sp_trades)
            if s:
                lines.append(
                    f'  {cfg:<12} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["p_lt_neg1"]:>7.3f}'
                )
            else:
                lines.append(f'  {cfg:<12} {sp_name:>6}   N/A')
    lines.append('')

    # Sleeve correlation
    lines.append('SLEEVE CORRELATION (pooled FX, return series over shared trade set)')
    for (s1, s2), corr in corr_map.items():
        lines.append(f'  {s1:<24} vs  {s2:<24}  corr = {fmt(corr, 3)}')
    lines.append('')

    # Portfolio comparison
    lines.append(f'EQUAL-WEIGHT PORTFOLIO COMPARISON (1/{len(symbols)} weight per symbol)')
    portfolio_configs = [
        ('Raw ETD portfolio (A_raw all)', port2_stats),
        ('Generic-filter portfolio (B_generic all)', port3_stats),
        ('Mixed portfolio (C_usdchf for USD/CHF, B_generic for others)', port1_stats),
    ]
    lines.append(
        f'  {"Portfolio":<50} {"split":>6} {"n":>5} {"Sharpe":>8} {"MaxDD":>8} {"P<-1":>7}'
    )
    lines.append(f'  {"-"*50} {"-"*6} {"-"*5} {"-"*8} {"-"*8} {"-"*7}')
    for port_label, port_s in portfolio_configs:
        for sp_name in splits:
            s = port_s.get(sp_name)
            if s:
                lines.append(
                    f'  {port_label:<50} {sp_name:>6} {s["n"]:>5}'
                    f' {s["sharpe"]:>8.3f} {s["max_dd"]:>8.3f}'
                    f' {s["p_lt_neg1"]:>7.3f}'
                )
            else:
                lines.append(f'  {port_label:<50} {sp_name:>6}   N/A')
    lines.append('')

    # Filter portability classification
    lines.append('FILTER PORTABILITY CLASSIFICATION')
    lines.append(f'  ATR-transition filter:')
    lines.append(f'    improves >= 0.01 Sharpe on: {portability["atr_improve_count"]}/{len(symbols)} symbols')
    lines.append(f'    degrades >= 0.01 Sharpe on: {portability["atr_degrade_count"]}/{len(symbols)} symbols')
    lines.append(f'    pooled full-period Sharpe delta:  {fmt(atr_sharpe_delta, 3)}')
    lines.append(f'    pooled forward Sharpe delta:      {fmt(atr_fwd_sharpe_delta, 3)}')
    lines.append(f'    CLASSIFICATION: {portability["atr_transition"]}')
    lines.append('')
    lines.append(f'  Shock filter:')
    lines.append(f'    improves >= 0.01 Sharpe on: {portability["shock_improve_count"]}/{len(symbols)} symbols')
    lines.append(f'    degrades >= 0.01 Sharpe on: {portability["shock_degrade_count"]}/{len(symbols)} symbols')
    lines.append(f'    CLASSIFICATION: {portability["shock"]}')
    lines.append('')

    # Decision memo
    lines.append('DECISION MEMO')
    lines.append(f'  Recommended option: {recommendation}')
    lines.append(f'  {rec_text}')
    lines.append('')
    lines.append('  Rationale:')
    if portability['atr_transition'] == 'GENERIC':
        lines.append(
            f'    ATR-transition filter improves Sharpe on {portability["atr_improve_count"]}'
            f'/{len(symbols)} FX symbols and is classified as GENERIC.'
        )
    else:
        lines.append(
            f'    ATR-transition filter improves only {portability["atr_improve_count"]}'
            f'/{len(symbols)} FX symbols — insufficient breadth for GENERIC classification.'
        )
    if portability['shock'] == 'GENERIC':
        lines.append(
            f'    Shock filter improves Sharpe on {portability["shock_improve_count"]}'
            f'/{len(symbols)} FX symbols — classified as GENERIC.'
        )
    else:
        lines.append(
            f'    Shock filter improves only {portability["shock_improve_count"]}'
            f'/{len(symbols)} FX symbols — classified as SYMBOL_SPECIFIC (likely USD/CHF-specific).'
        )
    lines.append('')
    lines.append(f'  Next steps:')
    if recommendation in ('A', 'B'):
        lines.append('    - Retain ATR-transition filter as generic ETD filter across FX')
        if recommendation == 'B':
            lines.append('    - Retain shock filter exclusively for USD/CHF sleeve')
            lines.append('    - Maintain two separate ETD objects: generic sleeve + USD/CHF-specialized sleeve')
        else:
            lines.append('    - Apply shock filter universally if evidence warrants')
        lines.append('    - Proceed to CTU sleeve integration in a subsequent issue')
    elif recommendation == 'C':
        lines.append('    - Maintain ETD as USD/CHF-only strategy')
        lines.append('    - Do not generalize filters to other FX pairs')
    else:
        lines.append('    - Revert to raw ETD across all FX symbols')
        lines.append('    - Reconsider filter design before further portfolio work')
    lines.append('')
    lines.append(f'Output: {args.output}')
    lines.append(f'Memo:   {args.memo}')

    memo_text = '\n'.join(lines) + '\n'
    with open(args.memo, 'w') as fh:
        fh.write(memo_text)

    print()
    print(memo_text)
    print(f'Memo written to {args.memo}')


if __name__ == '__main__':
    main()

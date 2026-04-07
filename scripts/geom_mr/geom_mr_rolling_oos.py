"""
GEOM_MR rolling walk-forward OOS — Issue #18
Expanding window: train = all data before test_year, test = calendar year.
Raw displacement signal (no z-score normalization). Threshold = mult × tick_size.
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Constants ────────────────────────────────────────────────────────────────
LOOKBACKS           = [10, 20, 30]
THRESHOLD_MULTS     = [5, 10, 20]
MIN_TRAIN_YEARS     = 3
MIN_TEST_TRADES     = 5

TIMEFRAMES = [
    ('5min',  'minute', '5min'),
    ('10min', 'minute', '10min'),
    ('15min', 'minute', '15min'),
    ('30min', 'minute', '30min'),
    ('1h',    'hourly', '1h'),
    ('4h',    'hourly', '4h'),
    ('6h',    'hourly', '6h'),
]

SYMBOLS = [
    'AUD/CAD','AUD/CHF','AUD/JPY','AUD/NZD','AUD/USD',
    'EUR/USD','GBP/USD','USD/JPY',
    'AUS200',
    'BTC/USD','BCH/USD',
    'NGAS',
    'XAU/USD',
]

# ── DB connection ─────────────────────────────────────────────────────────────
def get_conn():
    load_dotenv()
    dsn = os.getenv('TIMESCALE_DSN')
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv('TIMESCALE_HOST'), port=os.getenv('TIMESCALE_PORT'),
        dbname=os.getenv('TIMESCALE_DB'), user=os.getenv('TIMESCALE_USER'),
        password=os.getenv('TIMESCALE_PASSWORD'),
    )

# ── Migrations ────────────────────────────────────────────────────────────────
def run_migrations(conn):
    root = Path(__file__).resolve().parents[2]
    for sql_file in sorted((root / 'db' / 'strategy_research').glob('*.sql')):
        with open(sql_file) as f:
            sql = f.read()
        for stmt in sql.split(';'):
            stripped = '\n'.join(
                line for line in stmt.splitlines()
                if not line.strip().startswith('--')
            ).strip()
            if not stripped:
                continue
            try:
                conn.cursor().execute(stripped)
                conn.commit()
            except Exception:
                conn.rollback()

# ── Symbol metadata ───────────────────────────────────────────────────────────
def load_tick_sizes(conn):
    cur = conn.cursor()
    cur.execute("SELECT symbol, tick_size FROM market_data.symbols")
    return {r[0]: float(r[1]) for r in cur.fetchall()}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(conn, symbol, source, resample_rule):
    cur = conn.cursor()
    table = 'minute_prices' if source == 'minute' else 'hourly_prices'
    cur.execute(f"""
        SELECT m.date,
               (m.bid_close + m.ask_close) / 2.0 AS mid_close
        FROM market_data.{table} m
        JOIN market_data.symbols s ON s.id = m.symbol_id
        WHERE s.symbol = %s
        ORDER BY m.date
    """, (symbol,))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['ts', 'mid_close'])
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    df = df.set_index('ts').sort_index()
    if resample_rule != '1h':
        df = df['mid_close'].resample(resample_rule).last().dropna().to_frame()
    return df

# ── Feature computation ───────────────────────────────────────────────────────
def compute_features(df, lookback):
    prices = df['mid_close'].values.astype(np.float64)
    n = len(prices)
    disp = np.full(n, np.nan)
    for i in range(lookback, n):
        mu = prices[i - lookback:i].mean()
        disp[i] = prices[i] - mu
    df = df.copy()
    df['displacement'] = disp
    return df

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df, threshold_native):
    prices     = df['mid_close'].values.astype(np.float64)
    disps      = df['displacement'].values.astype(np.float64)
    timestamps = df.index
    n = len(df)

    trades = []
    in_pos = False
    side = 0
    entry_i = 0
    entry_price = 0.0

    for i in range(n):
        d = disps[i]
        if np.isnan(d):
            continue
        p = prices[i]
        if not in_pos:
            if d < -threshold_native:
                in_pos = True; side = 1
                entry_i = i; entry_price = p
            elif d > threshold_native:
                in_pos = True; side = -1
                entry_i = i; entry_price = p
        else:
            if (side == 1 and d >= 0.0) or (side == -1 and d <= 0.0):
                ret = (p - entry_price) / entry_price * side
                trades.append({
                    'entry_ts':    timestamps[entry_i].to_pydatetime(),
                    'exit_ts':     timestamps[i].to_pydatetime(),
                    'entry_price': entry_price,
                    'exit_price':  p,
                    'side':        side,
                    'bars_held':   i - entry_i,
                    'return_pct':  ret,
                })
                in_pos = False

    return trades

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    if not trades:
        return dict(n=0, sharpe=None, win_rate=None, cum_ret=None, max_dd=None, avg_bars=None)
    returns = np.array([t['return_pct'] for t in trades])
    n = len(returns)
    mean_r = float(np.mean(returns))
    std_r  = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe = mean_r / std_r if std_r > 0 else 0.0
    win_r  = float(np.mean(returns > 0))
    cum    = float(np.prod(1 + returns) - 1)
    eq     = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(eq)
    dd     = float(np.max((peak - eq) / peak)) if n > 1 else 0.0
    avg_bars = float(np.mean([t['bars_held'] for t in trades]))
    return dict(n=n, sharpe=sharpe, win_rate=win_r, cum_ret=cum, max_dd=dd, avg_bars=avg_bars)

def safe_corr(xs, ys):
    if len(xs) < 3:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

# ── Bulk insert helpers ────────────────────────────────────────────────────────
def upsert_run(conn, run_id, config):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO strategy_research.experiment_runs
            (run_id, strategy_name, run_ts, config, notes)
        VALUES (%s, 'GEOM_MR_ROLLING_OOS', NOW(), %s, 'Issue #18 rolling walk-forward OOS')
        ON CONFLICT (run_id) DO NOTHING
    """, (str(run_id), json.dumps(config)))
    conn.commit()

def insert_folds(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO strategy_research.geom_mr_rolling_folds
            (run_id, symbol, timeframe, lookback, threshold_mult, test_year,
             train_n_trades, train_sharpe, train_win_rate, train_cum_ret,
             test_n_trades,  test_sharpe,  test_win_rate,  test_cum_ret,
             test_max_dd, test_avg_bars)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    psycopg2.extras.execute_values(conn.cursor(), sql, rows, page_size=1000)
    conn.commit()

def insert_summary(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO strategy_research.geom_mr_rolling_summary
            (run_id, symbol, timeframe, lookback, threshold_mult,
             n_folds, mean_test_sharpe, median_test_sharpe, std_test_sharpe,
             min_test_sharpe, max_test_sharpe, pct_pos_folds,
             mean_test_wr, mean_test_cum, mean_test_dd,
             mean_train_sharpe, train_test_corr)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    psycopg2.extras.execute_values(conn.cursor(), sql, rows, page_size=500)
    conn.commit()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn    = get_conn()
    run_id  = uuid.uuid4()
    tick_sizes = load_tick_sizes(conn)

    config = {
        'lookbacks': LOOKBACKS, 'threshold_mults': THRESHOLD_MULTS,
        'min_train_years': MIN_TRAIN_YEARS,
        'timeframes': [t[0] for t in TIMEFRAMES],
    }

    print(f'run_id: {run_id}')
    print(f'Start: {datetime.now(timezone.utc).isoformat()}')
    print()

    run_migrations(conn)
    upsert_run(conn, run_id, config)

    total_combos = len(SYMBOLS) * len(TIMEFRAMES) * len(LOOKBACKS) * len(THRESHOLD_MULTS)
    done = 0

    for sym in SYMBOLS:
        tick = tick_sizes.get(sym)
        if tick is None:
            print(f'  WARNING: no tick_size for {sym}, skipping')
            continue

        for (tf_label, source, resample) in TIMEFRAMES:
            df_raw = load_data(conn, sym, source, resample)
            if df_raw is None or len(df_raw) < 200:
                continue

            first_year = df_raw.index.year.min()
            all_years  = sorted(df_raw.index.year.unique())
            test_years = [y for y in all_years if y >= first_year + MIN_TRAIN_YEARS]
            if not test_years:
                continue

            for lb in LOOKBACKS:
                df_feat = compute_features(df_raw, lb)

                for mult in THRESHOLD_MULTS:
                    threshold_native = mult * tick
                    all_trades = run_backtest(df_feat, threshold_native)

                    fold_rows     = []
                    summary_train = []
                    summary_test  = []

                    for test_year in test_years:
                        train_trades = [t for t in all_trades if t['entry_ts'].year < test_year]
                        test_trades  = [t for t in all_trades if t['entry_ts'].year == test_year]

                        tm = compute_metrics(train_trades)
                        ym = compute_metrics(test_trades)

                        fold_rows.append((
                            str(run_id), sym, tf_label, lb, mult, test_year,
                            tm['n'],      tm['sharpe'],   tm['win_rate'],  tm['cum_ret'],
                            ym['n'],      ym['sharpe'],   ym['win_rate'],  ym['cum_ret'],
                            ym['max_dd'], ym['avg_bars'],
                        ))

                        if tm['sharpe'] is not None:
                            summary_train.append(tm['sharpe'])
                        if ym['sharpe'] is not None and ym['n'] >= MIN_TEST_TRADES:
                            summary_test.append((test_year, ym['sharpe'], ym['win_rate'], ym['cum_ret'], ym['max_dd']))

                    insert_folds(conn, fold_rows)

                    if summary_test:
                        test_sharpes = [x[1] for x in summary_test]
                        test_wrs     = [x[2] for x in summary_test if x[2] is not None]
                        test_cums    = [x[3] for x in summary_test if x[3] is not None]
                        test_dds     = [x[4] for x in summary_test if x[4] is not None]
                        n_f   = len(test_sharpes)
                        arr   = np.array(test_sharpes)
                        pct_pos = float(np.mean(arr > 0))
                        train_aligned = summary_train[-n_f:] if len(summary_train) >= n_f else summary_train
                        corr = safe_corr(train_aligned, test_sharpes[:len(train_aligned)])
                    else:
                        n_f = 0; arr = np.array([]); pct_pos = None
                        test_wrs = []; test_cums = []; test_dds = []
                        corr = None

                    insert_summary(conn, [(
                        str(run_id), sym, tf_label, lb, mult,
                        n_f,
                        float(arr.mean())             if n_f else None,
                        float(np.median(arr))         if n_f else None,
                        float(arr.std(ddof=1))        if n_f > 1 else None,
                        float(arr.min())              if n_f else None,
                        float(arr.max())              if n_f else None,
                        pct_pos,
                        float(np.mean(test_wrs))      if test_wrs  else None,
                        float(np.mean(test_cums))     if test_cums else None,
                        float(np.mean(test_dds))      if test_dds  else None,
                        float(np.mean(summary_train)) if summary_train else None,
                        corr,
                    )])

                    done += 1
                    if done % 50 == 0:
                        pct = 100 * done / total_combos
                        print(f'  [{pct:5.1f}%] {done}/{total_combos}  last: {sym} {tf_label} lb={lb} mult={mult}')

    print(f'\nDone: {datetime.now(timezone.utc).isoformat()}')
    print(f'run_id: {run_id}')

    # ── Summary printout ───────────────────────────────────────────────────────
    cur = conn.cursor()
    print('\n=== GEOM_MR OOS summary by timeframe × threshold_mult ===')
    cur.execute("""
        SELECT timeframe, threshold_mult,
               COUNT(*) AS n_combos,
               ROUND(AVG(mean_test_sharpe)::numeric, 3)    AS mean_sharpe,
               ROUND(AVG(median_test_sharpe)::numeric, 3)  AS med_sharpe,
               ROUND(AVG(pct_pos_folds)::numeric, 3)       AS mean_pct_pos,
               ROUND(AVG(mean_test_wr)::numeric, 3)        AS mean_wr,
               ROUND(AVG(n_folds)::numeric, 1)             AS avg_folds
        FROM strategy_research.geom_mr_rolling_summary
        WHERE run_id = %s
        GROUP BY timeframe, threshold_mult
        ORDER BY timeframe, threshold_mult
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'TF':<6} {'mult':>5} {'combos':>6} {'mean_S':>8} {'med_S':>8} {'pct+':>6} {'wr':>6} {'folds':>6}")
    print('-' * 55)
    for r in rows:
        tf, mult, nc, ms, meds, pp, wr, folds = r
        print(f"{tf:<6} {mult:>5} {nc:>6} {float(ms):>+8.3f} {float(meds):>+8.3f} {float(pp):>6.1%} {float(wr):>6.1%} {float(folds):>6.1f}")

    print('\n=== GEOM_MR Top 20 combos by median OOS Sharpe ===')
    cur.execute("""
        SELECT symbol, timeframe, lookback, threshold_mult,
               n_folds, median_test_sharpe, mean_test_sharpe,
               pct_pos_folds, mean_test_wr, train_test_corr
        FROM strategy_research.geom_mr_rolling_summary
        WHERE run_id = %s
        ORDER BY median_test_sharpe DESC NULLS LAST
        LIMIT 20
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'sym':<12} {'tf':<6} {'lb':>4} {'mult':>5} {'folds':>5} {'med_S':>8} {'mean_S':>8} {'pct+':>6} {'wr':>6} {'corr':>7}")
    print('-' * 68)
    for r in rows:
        sym, tf, lb, mult, nf, meds, ms, pp, wr, corr = r
        corr_s = f'{float(corr):+.2f}' if corr is not None else '  —  '
        print(f"{sym:<12} {tf:<6} {lb:>4} {mult:>5} {nf:>5} {float(meds):>+8.3f} {float(ms):>+8.3f} {float(pp):>6.1%} {float(wr):>6.1%} {corr_s:>7}")

    conn.close()

if __name__ == '__main__':
    main()

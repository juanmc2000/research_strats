"""
STAT_MR rolling walk-forward OOS — Issue #17
Expanding window: train = all data before test_year, test = calendar year.
Covers all 7 timeframes, all param combos, long_only / short_only / combined.
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
THRESHOLDS          = [1.5, 2.0, 2.5]
SIDE_FILTERS        = ['long_only', 'short_only', 'combined']
MIN_TRAIN_YEARS     = 3
MIN_TEST_TRADES     = 5   # skip fold if fewer test trades (record as 0, don't crash)

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
    z = np.full(n, np.nan)
    for i in range(lookback, n):
        window = prices[i - lookback:i]
        mu = window.mean()
        sd = window.std(ddof=1)
        if sd > 0:
            z[i] = (prices[i] - mu) / sd
    df = df.copy()
    df['z_score'] = z
    return df

# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(df, threshold, side_filter):
    prices    = df['mid_close'].values.astype(np.float64)
    zscores   = df['z_score'].values.astype(np.float64)
    timestamps = df.index
    n = len(df)

    allow_long  = side_filter in ('long_only',  'combined')
    allow_short = side_filter in ('short_only', 'combined')

    trades = []
    in_pos = False
    side = 0
    entry_i = 0
    entry_price = 0.0
    entry_z = 0.0

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            continue
        p = prices[i]
        if not in_pos:
            if allow_long and z < -threshold:
                in_pos = True; side = 1
                entry_i = i; entry_price = p; entry_z = z
            elif allow_short and z > threshold:
                in_pos = True; side = -1
                entry_i = i; entry_price = p; entry_z = z
        else:
            exit_condition = (side == 1 and z >= 0.0) or (side == -1 and z <= 0.0)
            if exit_condition:
                ret = (p - entry_price) / entry_price * side
                trades.append({
                    'entry_ts':    timestamps[entry_i].to_pydatetime(),
                    'exit_ts':     timestamps[i].to_pydatetime(),
                    'entry_price': entry_price,
                    'exit_price':  p,
                    'entry_z':     entry_z,
                    'exit_z':      z,
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
    # drawdown on equity curve
    eq = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max((peak - eq) / peak)) if n > 1 else 0.0
    avg_bars = float(np.mean([t['bars_held'] for t in trades]))
    return dict(n=n, sharpe=sharpe, win_rate=win_r, cum_ret=cum, max_dd=dd, avg_bars=avg_bars)

# ── Correlation helper ─────────────────────────────────────────────────────────
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
        VALUES (%s, 'STAT_MR_ROLLING_OOS', NOW(), %s, 'Issue #17 rolling walk-forward OOS')
        ON CONFLICT (run_id) DO NOTHING
    """, (str(run_id), json.dumps(config)))
    conn.commit()

def insert_folds(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO strategy_research.stat_mr_rolling_folds
            (run_id, symbol, timeframe, lookback, threshold, side_filter, test_year,
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
        INSERT INTO strategy_research.stat_mr_rolling_summary
            (run_id, symbol, timeframe, lookback, threshold, side_filter,
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
    config  = {
        'lookbacks': LOOKBACKS, 'thresholds': THRESHOLDS,
        'side_filters': SIDE_FILTERS, 'min_train_years': MIN_TRAIN_YEARS,
        'timeframes': [t[0] for t in TIMEFRAMES],
    }

    print(f'run_id: {run_id}')
    print(f'Start: {datetime.now(timezone.utc).isoformat()}')
    print()

    run_migrations(conn)
    upsert_run(conn, run_id, config)

    total_combos = len(SYMBOLS) * len(TIMEFRAMES) * len(LOOKBACKS) * len(THRESHOLDS) * len(SIDE_FILTERS)
    done = 0

    for sym in SYMBOLS:
        for (tf_label, source, resample) in TIMEFRAMES:
            df_raw = load_data(conn, sym, source, resample)
            if df_raw is None or len(df_raw) < 200:
                continue

            # Determine available test years (per symbol)
            first_year = df_raw.index.year.min()
            all_years  = sorted(df_raw.index.year.unique())
            test_years = [y for y in all_years if y >= first_year + MIN_TRAIN_YEARS]

            if not test_years:
                continue

            for lb in LOOKBACKS:
                df_feat = compute_features(df_raw, lb)

                for thr in THRESHOLDS:
                    for sf in SIDE_FILTERS:
                        all_trades = run_backtest(df_feat, thr, sf)

                        fold_rows     = []
                        summary_train = []
                        summary_test  = []

                        for test_year in test_years:
                            train_trades = [t for t in all_trades if t['entry_ts'].year < test_year]
                            test_trades  = [t for t in all_trades if t['entry_ts'].year == test_year]

                            tm = compute_metrics(train_trades)
                            ym = compute_metrics(test_trades)

                            fold_rows.append((
                                str(run_id), sym, tf_label, lb, thr, sf, test_year,
                                tm['n'],       tm['sharpe'],   tm['win_rate'],  tm['cum_ret'],
                                ym['n'],       ym['sharpe'],   ym['win_rate'],  ym['cum_ret'],
                                ym['max_dd'],  ym['avg_bars'],
                            ))

                            if tm['sharpe'] is not None:
                                summary_train.append(tm['sharpe'])
                            if ym['sharpe'] is not None and ym['n'] >= MIN_TEST_TRADES:
                                summary_test.append((test_year, ym['sharpe'], ym['win_rate'], ym['cum_ret'], ym['max_dd']))

                        # Insert folds
                        insert_folds(conn, fold_rows)

                        # Build summary
                        if summary_test:
                            test_sharpes = [x[1] for x in summary_test]
                            test_wrs     = [x[2] for x in summary_test if x[2] is not None]
                            test_cums    = [x[3] for x in summary_test if x[3] is not None]
                            test_dds     = [x[4] for x in summary_test if x[4] is not None]
                            n_f          = len(test_sharpes)
                            arr          = np.array(test_sharpes)
                            pct_pos      = float(np.mean(arr > 0))
                            # train sharpes aligned to same test years
                            train_aligned = summary_train[-n_f:] if len(summary_train) >= n_f else summary_train
                            corr = safe_corr(train_aligned, test_sharpes[:len(train_aligned)])
                        else:
                            n_f = 0; arr = np.array([]); pct_pos = None
                            test_wrs = []; test_cums = []; test_dds = []
                            corr = None

                        insert_summary(conn, [(
                            str(run_id), sym, tf_label, lb, thr, sf,
                            n_f,
                            float(arr.mean())               if n_f else None,
                            float(np.median(arr))           if n_f else None,
                            float(arr.std(ddof=1))          if n_f > 1 else None,
                            float(arr.min())                if n_f else None,
                            float(arr.max())                if n_f else None,
                            pct_pos,
                            float(np.mean(test_wrs))        if test_wrs  else None,
                            float(np.mean(test_cums))       if test_cums else None,
                            float(np.mean(test_dds))        if test_dds  else None,
                            float(np.mean(summary_train))   if summary_train else None,
                            corr,
                        )])

                        done += 1
                        if done % 50 == 0:
                            pct = 100 * done / total_combos
                            print(f'  [{pct:5.1f}%] {done}/{total_combos}  last: {sym} {tf_label} lb={lb} thr={thr} {sf}')

    print(f'\nDone: {datetime.now(timezone.utc).isoformat()}')
    print(f'run_id: {run_id}')

    # ── Summary printout ───────────────────────────────────────────────────────
    cur = conn.cursor()
    print('\n=== OOS summary by timeframe × side_filter ===')
    cur.execute("""
        SELECT timeframe, side_filter,
               COUNT(*) AS n_combos,
               ROUND(AVG(mean_test_sharpe)::numeric, 3)    AS mean_sharpe,
               ROUND(AVG(median_test_sharpe)::numeric, 3)  AS med_sharpe,
               ROUND(AVG(pct_pos_folds)::numeric, 3)       AS mean_pct_pos,
               ROUND(AVG(mean_test_wr)::numeric, 3)        AS mean_wr,
               ROUND(AVG(n_folds)::numeric, 1)             AS avg_folds
        FROM strategy_research.stat_mr_rolling_summary
        WHERE run_id = %s
        GROUP BY timeframe, side_filter
        ORDER BY timeframe, side_filter
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'TF':<6} {'side':<12} {'combos':>6} {'mean_S':>8} {'med_S':>8} {'pct+':>6} {'wr':>6} {'folds':>6}")
    print('-' * 62)
    for r in rows:
        tf, sf, nc, ms, meds, pp, wr, folds = r
        print(f"{tf:<6} {sf:<12} {nc:>6} {float(ms):>+8.3f} {float(meds):>+8.3f} {float(pp):>6.1%} {float(wr):>6.1%} {float(folds):>6.1f}")

    print('\n=== Top 20 combos by median OOS Sharpe (long_only) ===')
    cur.execute("""
        SELECT symbol, timeframe, lookback, threshold,
               n_folds, median_test_sharpe, mean_test_sharpe,
               pct_pos_folds, mean_test_wr, train_test_corr
        FROM strategy_research.stat_mr_rolling_summary
        WHERE run_id = %s AND side_filter = 'long_only'
        ORDER BY median_test_sharpe DESC NULLS LAST
        LIMIT 20
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'sym':<12} {'tf':<6} {'lb':>4} {'thr':>5} {'folds':>5} {'med_S':>8} {'mean_S':>8} {'pct+':>6} {'wr':>6} {'corr':>7}")
    print('-' * 72)
    for r in rows:
        sym, tf, lb, thr, nf, meds, ms, pp, wr, corr = r
        corr_s = f'{float(corr):+.2f}' if corr is not None else '  —  '
        print(f"{sym:<12} {tf:<6} {lb:>4} {float(thr):>5.1f} {nf:>5} {float(meds):>+8.3f} {float(ms):>+8.3f} {float(pp):>6.1%} {float(wr):>6.1%} {corr_s:>7}")

    conn.close()

if __name__ == '__main__':
    main()

"""
Baseline Statistical Mean Reversion Research (STAT_MR)
=======================================================
Issue #8 — Stage A exploratory research.

Signal:  z_t = (close_t - mean_N) / std_N
Entry:   long if z < -threshold, short if z > +threshold
Exit:    z crosses 0

Parameter grid:
  lookback   ∈ {10, 20, 30}
  threshold  ∈ {1.5, 2.0, 2.5}
  timeframes: 5min, 10min, 15min, 30min, 1h, 4h, 6h

Research split:
  in_sample : date < 2025-06-01
  unseen    : date >= 2025-06-01

All outputs written to strategy_research schema only.
"""

import os
import json
import uuid
import statistics
from collections import defaultdict
from pathlib import Path

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OOS_CUTOFF    = pd.Timestamp('2025-06-01', tz='UTC')
LOOKBACKS     = [10, 20, 30]
THRESHOLDS    = [1.5, 2.0, 2.5]
FEAT_LOOKBACK = 20   # representative lookback for feature table storage

TIMEFRAMES = [
    ('5min',  '5min',  'minute'),
    ('10min', '10min', 'minute'),
    ('15min', '15min', 'minute'),
    ('30min', '30min', 'minute'),
    ('1h',    '1h',    'hourly'),
    ('4h',    '4h',    'hourly'),
    ('6h',    '6h',    'hourly'),
]

MINUTE_SYMBOLS = [
    'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD',
    'EUR/USD', 'GBP/USD', 'USD/JPY',
    'AUS200', 'BTC/USD', 'BCH/USD', 'NGAS', 'XAU/USD',
]

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
def connect():
    _root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=_root / '.env')
    dsn = os.environ.get('TIMESCALE_DSN')
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.environ['PGHOST'], dbname=os.environ['PGDATABASE'],
        user=os.environ['PGUSER'], password=os.environ['PGPASSWORD'],
        port=int(os.environ.get('PGPORT', 5432)),
    )


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
    print("Migrations verified.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_symbol_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT symbol, id FROM market_data.symbols")
    return {row[0]: row[1] for row in cur.fetchall()}


def load_minute_data(conn, symbol_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT date, (bid_close + ask_close) / 2.0
        FROM market_data.minute_prices
        WHERE symbol_id = %s ORDER BY date
    """, (symbol_id,))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['date', 'mid_close'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df['mid_close'] = df['mid_close'].astype(float)
    return df


def load_hourly_data(conn, symbol_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT date, (bid_close + ask_close) / 2.0
        FROM market_data.hourly_prices
        WHERE symbol_id = %s ORDER BY date
    """, (symbol_id,))
    rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['date', 'mid_close'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df['mid_close'] = df['mid_close'].astype(float)
    return df


def resample_to_tf(df, rule):
    return df['mid_close'].resample(rule).last().dropna().to_frame()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def compute_features(df, lookback):
    prices = df['mid_close']
    mean_n = prices.rolling(lookback).mean()
    std_n  = prices.rolling(lookback).std(ddof=1)
    z      = (prices - mean_n) / std_n.replace(0, np.nan)
    out = df.copy()
    out['mean_n']  = mean_n
    out['std_n']   = std_n
    out['z_score'] = z
    return out.dropna(subset=['z_score'])


def period_label(ts):
    return 'unseen' if ts >= OOS_CUTOFF else 'in_sample'


# ---------------------------------------------------------------------------
# Backtest — pure numpy loop for speed
# ---------------------------------------------------------------------------
def run_backtest(df_feat, threshold):
    prices    = df_feat['mid_close'].values
    zscores   = df_feat['z_score'].values
    timestamps = df_feat.index
    n = len(df_feat)

    trades = []
    in_pos = False
    side   = 0
    entry_i = 0
    entry_price = 0.0
    entry_z = 0.0

    for i in range(n):
        z = zscores[i]
        if np.isnan(z):
            continue
        p = prices[i]

        if not in_pos:
            if z < -threshold:
                in_pos = True; side = 1
                entry_i = i; entry_price = p; entry_z = z
            elif z > threshold:
                in_pos = True; side = -1
                entry_i = i; entry_price = p; entry_z = z
        else:
            if (side == 1 and z >= 0.0) or (side == -1 and z <= 0.0):
                ret = (p - entry_price) / entry_price * side
                ts_e = timestamps[entry_i]
                trades.append((
                    ts_e.to_pydatetime(),
                    timestamps[i].to_pydatetime(),
                    float(entry_price), float(p),
                    float(entry_z),     float(z),
                    side, i - entry_i,  float(ret),
                    period_label(ts_e),
                ))
                in_pos = False
    return trades


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _sharpe(rets):
    if len(rets) < 2: return None
    m = statistics.mean(rets)
    s = statistics.stdev(rets)
    return m / s if s > 0 else None

def _median(vals):
    if not vals: return None
    s = sorted(vals); n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2.0

def _max_dd(rets):
    cum = peak = mdd = 0.0
    for r in rets:
        cum += r
        if cum > peak: peak = cum
        if peak - cum > mdd: mdd = peak - cum
    return mdd

def _payoff(rets):
    wins  = [r for r in rets if r > 0]
    loses = [r for r in rets if r < 0]
    if not wins or not loses: return None
    return statistics.mean(wins) / abs(statistics.mean(loses))

def trade_stats(trades):
    if not trades: return None
    rets = [t[8] for t in trades]
    bars = [t[7] for t in trades]
    n    = len(rets)
    return {
        'n':      n,
        'mean':   statistics.mean(rets),
        'median': _median(rets),
        'std':    statistics.stdev(rets) if n > 1 else 0.0,
        'sharpe': _sharpe(rets),
        'wr':     sum(1 for r in rets if r > 0) / n,
        'payoff': _payoff(rets),
        'cum':    sum(rets),
        'mdd':    _max_dd(rets),
        'bars':   statistics.mean(bars),
    }


# ---------------------------------------------------------------------------
# Bulk DB writes
# ---------------------------------------------------------------------------
TRADE_COLS = ('entry_ts','exit_ts','entry_price','exit_price',
              'z_at_entry','z_at_exit','side','bars_held','return_pct','period_label')

def bulk_insert_trades(cur, run_id, symbol, tf_label, lb, thr, trades):
    if not trades:
        return
    rows = [(run_id, symbol, tf_label, lb, thr) + t for t in trades]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO strategy_research.stat_mr_trades
            (run_id, symbol, timeframe, lookback, threshold,
             entry_ts, exit_ts, entry_price, exit_price,
             z_at_entry, z_at_exit, side, bars_held, return_pct, period_label)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=5000)


def bulk_insert_signals(cur, run_id, symbol, tf_label, lb, thr, trades):
    if not trades:
        return
    rows = [(run_id, symbol, tf_label, lb, thr,
             t[0], t[6], t[4], t[2], t[9]) for t in trades]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO strategy_research.stat_mr_signals
            (run_id, symbol, timeframe, lookback, threshold,
             signal_ts, side, z_at_entry, mid_at_entry, period_label)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=5000)


def bulk_insert_features(cur, run_id, symbol, tf_label, lb, df_feat, entry_ts_set):
    """Store features only at actual entry bars (entries from any threshold)."""
    rows = []
    for ts, row in df_feat.iterrows():
        if ts.to_pydatetime() not in entry_ts_set:
            continue
        rows.append((
            run_id, symbol, tf_label, lb,
            ts.to_pydatetime(),
            float(row['mid_close']), float(row['mean_n']),
            float(row['std_n']),     float(row['z_score']),
            period_label(ts),
        ))
    if rows:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO strategy_research.stat_mr_features
                (run_id, symbol, timeframe, lookback, bar_ts, mid_close,
                 mean_n, std_n, z_score, period_label)
            VALUES %s ON CONFLICT DO NOTHING
        """, rows, page_size=5000)


def write_summary(cur, run_id, symbol, tf_label, lb, thr, period, st):
    if st is None: return
    cur.execute("""
        INSERT INTO strategy_research.stat_mr_analysis_summary
            (run_id, symbol, timeframe, lookback, threshold, period_label,
             n_trades, mean_return, median_return, std_return, sharpe,
             win_rate, payoff, cum_return, max_drawdown, avg_bars_held)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """, (run_id, symbol, tf_label, lb, thr, period,
          st['n'], st['mean'], st['median'], st['std'], st['sharpe'],
          st['wr'], st['payoff'], st['cum'], st['mdd'], st['bars']))


def write_period_cmp(cur, run_id, symbol, tf_label, lb, thr, is_st, oos_st):
    def v(d, k): return d[k] if d else None
    n_is  = (is_st['n']  if is_st  else 0) or 0
    n_oos = (oos_st['n'] if oos_st else 0) or 0
    sh_is  = v(is_st,  'sharpe'); sh_oos = v(oos_st, 'sharpe')
    mr_is  = v(is_st,  'mean');   mr_oos = v(oos_st, 'mean')
    sh_deg = sh_oos / sh_is if sh_is and sh_oos and sh_is != 0 else None
    mr_deg = mr_oos / mr_is if mr_is and mr_oos and mr_is != 0 else None
    tc_rat = n_oos / n_is   if n_is > 0 else None
    cur.execute("""
        INSERT INTO strategy_research.stat_mr_period_comparison
            (run_id, symbol, timeframe, lookback, threshold,
             is_sharpe, oos_sharpe, is_mean_return, oos_mean_return,
             is_win_rate, oos_win_rate, is_payoff, oos_payoff,
             is_max_dd, oos_max_dd,
             sharpe_degradation, return_degradation, trade_count_ratio)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """, (run_id, symbol, tf_label, lb, thr,
          sh_is, sh_oos, mr_is, mr_oos,
          v(is_st,'wr'), v(oos_st,'wr'),
          v(is_st,'payoff'), v(oos_st,'payoff'),
          v(is_st,'mdd'),  v(oos_st,'mdd'),
          sh_deg, mr_deg, tc_rat))


def write_by_year(cur, run_id, symbol, tf_label, lb, thr, trades):
    by_yr = defaultdict(list)
    for t in trades:
        by_yr[t[0].year].append(t)
    for yr, yt in sorted(by_yr.items()):
        st = trade_stats(yt)
        if not st: continue
        cur.execute("""
            INSERT INTO strategy_research.stat_mr_analysis_by_period
                (run_id, symbol, timeframe, lookback, threshold, period_year,
                 n_trades, mean_return, sharpe, win_rate, cum_return)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, lb, thr, yr,
              st['n'], st['mean'], st['sharpe'], st['wr'], st['cum']))


def write_entry_buckets(cur, run_id, symbol, tf_label, lb, trades):
    bkts = defaultdict(list)
    for t in trades:
        z = abs(t[4]); side = t[6]; pl = t[9]
        bkt = '1.5-2.0' if z < 2.0 else ('2.0-2.5' if z < 2.5 else ('2.5-3.0' if z < 3.0 else '3.0+'))
        bkts[(bkt, side, pl)].append(t[8])
    for (bkt, side, pl), rets in bkts.items():
        n  = len(rets)
        wr = sum(1 for r in rets if r > 0) / n
        mr = statistics.mean(rets)
        cur.execute("""
            INSERT INTO strategy_research.stat_mr_analysis_by_entry_bucket
                (run_id, symbol, timeframe, lookback, period_label, z_bucket, side,
                 n_trades, mean_return, win_rate)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, lb, pl, bkt, side, n, mr, wr))


def write_tf_comparison(cur, run_id, symbol, lb, thr, tf_data):
    # tf_data: list of (tf_label, is_sh, oos_sh, is_n, oos_n, is_wr, oos_wr)
    is_ranked  = sorted(tf_data, key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
    oos_ranked = sorted(tf_data, key=lambda x: x[2] if x[2] is not None else -999, reverse=True)
    for rank, row in enumerate(is_ranked, 1):
        cur.execute("""
            INSERT INTO strategy_research.stat_mr_timeframe_comparison
                (run_id, symbol, lookback, threshold, timeframe, period_label,
                 sharpe, n_trades, win_rate, tf_rank)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, lb, thr, row[0], 'in_sample',
              row[1], row[3], row[5], rank))
    for rank, row in enumerate(oos_ranked, 1):
        cur.execute("""
            INSERT INTO strategy_research.stat_mr_timeframe_comparison
                (run_id, symbol, lookback, threshold, timeframe, period_label,
                 sharpe, n_trades, win_rate, tf_rank)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, lb, thr, row[0], 'unseen',
              row[2], row[4], row[6], rank))


def write_param_robustness(cur, run_id, symbol, tf_label, period, grid):
    # grid: {(lb, thr): {sharpe, n}}
    for (lb, thr), res in grid.items():
        sh = res.get('sharpe')
        n  = res.get('n', 0)
        nbr = []
        for dlb in [-10, 10]:
            for dthr in [-0.5, 0.5]:
                nb = grid.get((lb + dlb, round(thr + dthr, 1)))
                if nb and nb.get('sharpe') is not None:
                    nbr.append(nb['sharpe'])
        nbr_mean = statistics.mean(nbr) if nbr else None
        stab = (1.0 - abs(sh - nbr_mean) / (abs(sh) + 1e-8)) if (sh is not None and nbr_mean is not None) else None
        cur.execute("""
            INSERT INTO strategy_research.stat_mr_param_robustness
                (run_id, symbol, timeframe, period_label, lookback, threshold,
                 sharpe, n_trades, neighbor_mean_sharpe, stability_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, period, lb, thr, sh, n, nbr_mean, stab))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    conn = connect()
    run_migrations(conn)
    cur = conn.cursor()

    run_id = str(uuid.uuid4())
    config = {
        'lookbacks':  LOOKBACKS,
        'thresholds': THRESHOLDS,
        'timeframes': [t[0] for t in TIMEFRAMES],
        'symbols':    MINUTE_SYMBOLS,
        'oos_cutoff': str(OOS_CUTOFF.date()),
    }
    cur.execute("""
        INSERT INTO strategy_research.experiment_runs (run_id, strategy_name, config, notes)
        VALUES (%s, %s, %s, %s)
    """, (run_id, 'STAT_MR', json.dumps(config), 'Issue #8 — Baseline Statistical MR'))
    conn.commit()
    print(f"Run ID: {run_id}")

    sym_ids = load_symbol_ids(conn)

    for symbol in MINUTE_SYMBOLS:
        sym_id = sym_ids.get(symbol)
        if sym_id is None:
            print(f"  SKIP {symbol}: not in symbols table")
            continue
        print(f"\n--- {symbol} ---")

        minute_df = load_minute_data(conn, sym_id)
        hourly_df = load_hourly_data(conn, sym_id)
        print(f"  Loaded: {len(minute_df) if minute_df is not None else 0} min "
              f"/ {len(hourly_df) if hourly_df is not None else 0} hourly")

        # tf_data for timeframe comparison: (tf_label, is_sh, oos_sh, is_n, oos_n, is_wr, oos_wr)
        tf_compare = defaultdict(lambda: defaultdict(list))

        for tf_label, rule, source in TIMEFRAMES:
            base_df = (resample_to_tf(minute_df, rule) if source == 'minute' else
                       (hourly_df.copy() if rule == '1h' else resample_to_tf(hourly_df, rule)))
            if base_df is None or len(base_df) < max(LOOKBACKS) + 10:
                continue

            print(f"  {tf_label:<6} bars={len(base_df)}", end='', flush=True)

            # Compute features for each lookback
            feat_by_lb = {lb: compute_features(base_df, lb) for lb in LOOKBACKS}

            # Collect all entry timestamps for feature storage (FEAT_LOOKBACK only)
            feat_entries = set()

            grid_is  = {}
            grid_oos = {}

            for lb in LOOKBACKS:
                df_feat = feat_by_lb[lb]
                for thr in THRESHOLDS:
                    trades = run_backtest(df_feat, thr)
                    if not trades:
                        continue

                    is_t  = [t for t in trades if t[9] == 'in_sample']
                    oos_t = [t for t in trades if t[9] == 'unseen']

                    # Collect entry timestamps for feature storage
                    if lb == FEAT_LOOKBACK:
                        feat_entries.update(t[0] for t in trades)

                    bulk_insert_trades( cur, run_id, symbol, tf_label, lb, thr, trades)
                    bulk_insert_signals(cur, run_id, symbol, tf_label, lb, thr, trades)

                    is_st  = trade_stats(is_t)
                    oos_st = trade_stats(oos_t)
                    all_st = trade_stats(trades)

                    write_summary(cur, run_id, symbol, tf_label, lb, thr, 'in_sample', is_st)
                    write_summary(cur, run_id, symbol, tf_label, lb, thr, 'unseen',    oos_st)
                    write_summary(cur, run_id, symbol, tf_label, lb, thr, 'combined',  all_st)
                    write_period_cmp(cur, run_id, symbol, tf_label, lb, thr, is_st, oos_st)
                    write_by_year(cur, run_id, symbol, tf_label, lb, thr, trades)
                    write_entry_buckets(cur, run_id, symbol, tf_label, lb, trades)

                    grid_is[(lb, thr)]  = {'sharpe': is_st['sharpe']  if is_st  else None,
                                           'n':      is_st['n']       if is_st  else 0}
                    grid_oos[(lb, thr)] = {'sharpe': oos_st['sharpe'] if oos_st else None,
                                           'n':      oos_st['n']      if oos_st else 0}

                    # For timeframe comparison
                    tf_compare[(lb, thr)][tf_label] = (
                        is_st['sharpe']  if is_st  else None,
                        oos_st['sharpe'] if oos_st else None,
                        is_st['n']       if is_st  else 0,
                        oos_st['n']      if oos_st else 0,
                        is_st['wr']      if is_st  else None,
                        oos_st['wr']     if oos_st else None,
                    )

            # Features: only at actual entry bars for FEAT_LOOKBACK
            bulk_insert_features(cur, run_id, symbol, tf_label,
                                 FEAT_LOOKBACK, feat_by_lb[FEAT_LOOKBACK], feat_entries)

            write_param_robustness(cur, run_id, symbol, tf_label, 'in_sample', grid_is)
            write_param_robustness(cur, run_id, symbol, tf_label, 'unseen',    grid_oos)

            n_is_total = sum(v['n'] for v in grid_is.values())
            print(f"  IS_trades={n_is_total}")

        # Timeframe comparison
        for (lb, thr), tf_dict in tf_compare.items():
            tf_data = [
                (tfl, v[0], v[1], v[2], v[3], v[4], v[5])
                for tfl, v in tf_dict.items()
            ]
            write_tf_comparison(cur, run_id, symbol, lb, thr, tf_data)

        conn.commit()
        print(f"  Committed.")

    conn.commit()
    print(f"\nDone. Run ID: {run_id}")
    _print_summary(conn, run_id)
    conn.close()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(conn, run_id):
    cur = conn.cursor()
    print("\n" + "=" * 70)
    print("STAT_MR RESULTS SUMMARY")
    print("=" * 70)

    cur.execute("""
        SELECT symbol, timeframe, lookback, threshold,
               sharpe, n_trades, win_rate, mean_return
        FROM strategy_research.stat_mr_analysis_summary
        WHERE run_id = %s AND period_label = 'unseen' AND n_trades >= 5
        ORDER BY sharpe DESC NULLS LAST
        LIMIT 20
    """, (run_id,))
    rows = cur.fetchall()
    print(f"\nTop 20 combos — unseen Sharpe (n≥5):")
    print(f"  {'symbol':<12} {'tf':<7} {'lb':>4} {'thr':>5}  {'sh_oos':>7}  {'n':>5}  {'wr':>6}  {'mean':>9}")
    print("  " + "-" * 68)
    for r in rows:
        sh = f"{r[4]:+.3f}" if r[4] is not None else "   —  "
        print(f"  {r[0]:<12} {r[1]:<7} {r[2]:>4} {r[3]:>5.1f}  {sh:>7}  {r[5]:>5}  {r[6]:.3f}  {r[7]:+.6f}")

    cur.execute("""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN is_sharpe > 0  THEN 1 ELSE 0 END) AS pos_is,
            SUM(CASE WHEN oos_sharpe > 0 THEN 1 ELSE 0 END) AS pos_oos,
            AVG(sharpe_degradation) FILTER (WHERE sharpe_degradation IS NOT NULL) AS avg_deg,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe_degradation)
                FILTER (WHERE sharpe_degradation IS NOT NULL) AS med_deg
        FROM strategy_research.stat_mr_period_comparison
        WHERE run_id = %s AND is_sharpe IS NOT NULL AND oos_sharpe IS NOT NULL
    """, (run_id,))
    r = cur.fetchone()
    if r and r[0]:
        print(f"\nPeriod degradation (all combos with both periods):")
        print(f"  Total combos:               {r[0]}")
        print(f"  In-sample positive Sharpe:  {r[1]}")
        print(f"  Unseen positive Sharpe:     {r[2]}")
        if r[3] is not None: print(f"  Avg Sharpe degradation:     {r[3]:.3f}")
        if r[4] is not None: print(f"  Median Sharpe degradation:  {r[4]:.3f}")

    cur.execute("""
        SELECT DISTINCT ON (symbol)
            symbol, timeframe, lookback, threshold, sharpe, n_trades, win_rate
        FROM strategy_research.stat_mr_analysis_summary
        WHERE run_id = %s AND period_label = 'unseen' AND n_trades >= 5
        ORDER BY symbol, sharpe DESC NULLS LAST
    """, (run_id,))
    rows = cur.fetchall()
    print(f"\nBest unseen combo per symbol:")
    print(f"  {'symbol':<12} {'tf':<7} {'lb':>4} {'thr':>5}  {'sh_oos':>7}  {'n':>5}  {'wr':>6}")
    print("  " + "-" * 55)
    for r in rows:
        sh = f"{r[4]:+.3f}" if r[4] is not None else "   —  "
        print(f"  {r[0]:<12} {r[1]:<7} {r[2]:>4} {r[3]:>5.1f}  {sh:>7}  {r[5]:>5}  {r[6]:.3f}")


if __name__ == '__main__':
    main()

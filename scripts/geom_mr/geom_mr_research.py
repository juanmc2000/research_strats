"""
Baseline Geometric Mean Reversion Research (GEOM_MR)
=====================================================
Issue #9 — Stage A exploratory research.

Signal:  d_t = close_t - mean_N  (raw displacement, NO z-score normalization)
Entry:   long if d < -threshold, short if d > +threshold
Exit:    d crosses 0

Threshold is instrument-native: multiplier × tick_size
Parameter grid:
  lookback            ∈ {10, 20, 30}
  threshold_multiplier ∈ {5, 10, 20}  (× tick_size)
  timeframes: 5min, 10min, 15min, 30min, 1h, 4h, 6h

Research split:
  in_sample : date < 2025-06-01
  unseen    : date >= 2025-06-01

All outputs written to strategy_research schema only.
No z-score conversion at any point.
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
OOS_CUTOFF           = pd.Timestamp('2025-06-01', tz='UTC')
LOOKBACKS            = [10, 20, 30]
THRESHOLD_MULTS      = [5, 10, 20]
FEAT_LOOKBACK        = 20   # representative lookback for feature storage

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
def load_symbol_meta(conn):
    """Returns {symbol: {'id': int, 'tick_size': float}}"""
    cur = conn.cursor()
    cur.execute("SELECT symbol, id, tick_size FROM market_data.symbols")
    return {r[0]: {'id': r[1], 'tick_size': float(r[2]) if r[2] else None}
            for r in cur.fetchall()}


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
# Features — raw displacement, no normalization
# ---------------------------------------------------------------------------
def compute_features(df, lookback):
    prices = df['mid_close']
    mean_n = prices.rolling(lookback).mean()
    disp   = prices - mean_n          # raw displacement — never normalize
    out = df.copy()
    out['mean_n']       = mean_n
    out['displacement'] = disp
    return out.dropna(subset=['displacement'])


def period_label(ts):
    return 'unseen' if ts >= OOS_CUTOFF else 'in_sample'


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_backtest(df_feat, threshold_native):
    """
    Single-position state machine.
    Entry: |displacement| > threshold_native
    Exit:  displacement crosses 0
    """
    prices     = df_feat['mid_close'].values
    disps      = df_feat['displacement'].values
    timestamps = df_feat.index
    n = len(df_feat)

    trades = []
    in_pos = False
    side   = 0
    entry_i = 0
    entry_price = 0.0
    entry_disp  = 0.0

    for i in range(n):
        d = disps[i]
        if np.isnan(d):
            continue
        p = prices[i]

        if not in_pos:
            if d < -threshold_native:
                in_pos = True; side = 1
                entry_i = i; entry_price = p; entry_disp = d
            elif d > threshold_native:
                in_pos = True; side = -1
                entry_i = i; entry_price = p; entry_disp = d
        else:
            if (side == 1 and d >= 0.0) or (side == -1 and d <= 0.0):
                ret = (p - entry_price) / entry_price * side
                ts_e = timestamps[entry_i]
                trades.append((
                    ts_e.to_pydatetime(),
                    timestamps[i].to_pydatetime(),
                    float(entry_price), float(p),
                    float(entry_disp),  float(d),
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
def bulk_insert_trades(cur, run_id, symbol, tf_label, lb, mult, thr_native, trades):
    if not trades: return
    rows = [(run_id, symbol, tf_label, lb, mult, thr_native) + t for t in trades]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO strategy_research.geom_mr_trades
            (run_id, symbol, timeframe, lookback, threshold_mult, threshold_native,
             entry_ts, exit_ts, entry_price, exit_price,
             displacement_entry, displacement_exit,
             side, bars_held, return_pct, period_label)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=5000)


def bulk_insert_signals(cur, run_id, symbol, tf_label, lb, mult, thr_native, trades):
    if not trades: return
    rows = [(run_id, symbol, tf_label, lb, mult, thr_native,
             t[0], t[6], t[4], t[2], t[9]) for t in trades]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO strategy_research.geom_mr_signals
            (run_id, symbol, timeframe, lookback, threshold_mult, threshold_native,
             signal_ts, side, displacement_entry, mid_at_entry, period_label)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=5000)


def bulk_insert_features(cur, run_id, symbol, tf_label, lb, tick_size, df_feat, entry_ts_set):
    rows = []
    for ts, row in df_feat.iterrows():
        if ts.to_pydatetime() not in entry_ts_set:
            continue
        rows.append((
            run_id, symbol, tf_label, lb,
            ts.to_pydatetime(),
            float(row['mid_close']), float(row['mean_n']),
            float(row['displacement']), tick_size,
            period_label(ts),
        ))
    if rows:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO strategy_research.geom_mr_features
                (run_id, symbol, timeframe, lookback, bar_ts, mid_close,
                 mean_n, displacement, tick_size, period_label)
            VALUES %s ON CONFLICT DO NOTHING
        """, rows, page_size=5000)


def write_summary(cur, run_id, symbol, tf_label, lb, mult, period, st):
    if st is None: return
    cur.execute("""
        INSERT INTO strategy_research.geom_mr_analysis_summary
            (run_id, symbol, timeframe, lookback, threshold_mult, period_label,
             n_trades, mean_return, median_return, std_return, sharpe,
             win_rate, payoff, cum_return, max_drawdown, avg_bars_held)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """, (run_id, symbol, tf_label, lb, mult, period,
          st['n'], st['mean'], st['median'], st['std'], st['sharpe'],
          st['wr'], st['payoff'], st['cum'], st['mdd'], st['bars']))


def write_period_cmp(cur, run_id, symbol, tf_label, lb, mult, is_st, oos_st):
    def v(d, k): return d[k] if d else None
    n_is  = (is_st['n']  if is_st  else 0) or 0
    n_oos = (oos_st['n'] if oos_st else 0) or 0
    sh_is  = v(is_st,  'sharpe'); sh_oos = v(oos_st, 'sharpe')
    mr_is  = v(is_st,  'mean');   mr_oos = v(oos_st, 'mean')
    sh_deg = sh_oos / sh_is if sh_is and sh_oos and sh_is != 0 else None
    mr_deg = mr_oos / mr_is if mr_is and mr_oos and mr_is != 0 else None
    tc_rat = n_oos / n_is   if n_is > 0 else None
    cur.execute("""
        INSERT INTO strategy_research.geom_mr_period_comparison
            (run_id, symbol, timeframe, lookback, threshold_mult,
             is_sharpe, oos_sharpe, is_mean_return, oos_mean_return,
             is_win_rate, oos_win_rate, is_payoff, oos_payoff,
             is_max_dd, oos_max_dd,
             sharpe_degradation, return_degradation, trade_count_ratio)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING
    """, (run_id, symbol, tf_label, lb, mult,
          sh_is, sh_oos, mr_is, mr_oos,
          v(is_st,'wr'), v(oos_st,'wr'),
          v(is_st,'payoff'), v(oos_st,'payoff'),
          v(is_st,'mdd'),  v(oos_st,'mdd'),
          sh_deg, mr_deg, tc_rat))


def write_by_year(cur, run_id, symbol, tf_label, lb, mult, trades):
    by_yr = defaultdict(list)
    for t in trades:
        by_yr[t[0].year].append(t)
    for yr, yt in sorted(by_yr.items()):
        st = trade_stats(yt)
        if not st: continue
        cur.execute("""
            INSERT INTO strategy_research.geom_mr_analysis_by_period
                (run_id, symbol, timeframe, lookback, threshold_mult, period_year,
                 n_trades, mean_return, sharpe, win_rate, cum_return)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, lb, mult, yr,
              st['n'], st['mean'], st['sharpe'], st['wr'], st['cum']))


def write_disp_buckets(cur, run_id, symbol, tf_label, lb, tick_size, trades):
    """Group trades by displacement bucket at entry (in tick multiples)."""
    bkts = defaultdict(list)
    for t in trades:
        d_abs = abs(t[4]) / tick_size if tick_size and tick_size > 0 else 0
        side = t[6]; pl = t[9]
        bkt = ('5-10x'  if d_abs < 10 else
               '10-20x' if d_abs < 20 else
               '20-40x' if d_abs < 40 else '40x+')
        bkts[(bkt, side, pl)].append(t[8])
    for (bkt, side, pl), rets in bkts.items():
        n  = len(rets)
        wr = sum(1 for r in rets if r > 0) / n
        mr = statistics.mean(rets)
        cur.execute("""
            INSERT INTO strategy_research.geom_mr_analysis_by_displacement_bucket
                (run_id, symbol, timeframe, lookback, period_label, disp_bucket, side,
                 n_trades, mean_return, win_rate)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, lb, pl, bkt, side, n, mr, wr))


def write_tf_comparison(cur, run_id, symbol, lb, mult, tf_data):
    is_ranked  = sorted(tf_data, key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
    oos_ranked = sorted(tf_data, key=lambda x: x[2] if x[2] is not None else -999, reverse=True)
    for rank, row in enumerate(is_ranked, 1):
        cur.execute("""
            INSERT INTO strategy_research.geom_mr_timeframe_comparison
                (run_id, symbol, lookback, threshold_mult, timeframe, period_label,
                 sharpe, n_trades, win_rate, tf_rank)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, lb, mult, row[0], 'in_sample',
              row[1], row[3], row[5], rank))
    for rank, row in enumerate(oos_ranked, 1):
        cur.execute("""
            INSERT INTO strategy_research.geom_mr_timeframe_comparison
                (run_id, symbol, lookback, threshold_mult, timeframe, period_label,
                 sharpe, n_trades, win_rate, tf_rank)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, lb, mult, row[0], 'unseen',
              row[2], row[4], row[6], rank))


def write_param_robustness(cur, run_id, symbol, tf_label, period, grid):
    for (lb, mult), res in grid.items():
        sh = res.get('sharpe'); n = res.get('n', 0)
        nbr = []
        for dlb in [-10, 10]:
            for dm in [-5, 5]:
                nb = grid.get((lb + dlb, mult + dm))
                if nb and nb.get('sharpe') is not None:
                    nbr.append(nb['sharpe'])
        nbr_mean = statistics.mean(nbr) if nbr else None
        stab = (1.0 - abs(sh - nbr_mean) / (abs(sh) + 1e-8)) if (sh is not None and nbr_mean is not None) else None
        cur.execute("""
            INSERT INTO strategy_research.geom_mr_param_robustness
                (run_id, symbol, timeframe, period_label, lookback, threshold_mult,
                 sharpe, n_trades, neighbor_mean_sharpe, stability_score)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        """, (run_id, symbol, tf_label, period, lb, mult, sh, n, nbr_mean, stab))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    conn = connect()
    run_migrations(conn)
    cur = conn.cursor()

    run_id = str(uuid.uuid4())
    config = {
        'lookbacks':           LOOKBACKS,
        'threshold_multipliers': THRESHOLD_MULTS,
        'timeframes':          [t[0] for t in TIMEFRAMES],
        'symbols':             MINUTE_SYMBOLS,
        'oos_cutoff':          str(OOS_CUTOFF.date()),
        'normalization':       'none — raw displacement only',
    }
    cur.execute("""
        INSERT INTO strategy_research.experiment_runs (run_id, strategy_name, config, notes)
        VALUES (%s, %s, %s, %s)
    """, (run_id, 'GEOM_MR', json.dumps(config),
          'Issue #9 — Baseline Geometric MR (no normalization)'))
    conn.commit()
    print(f"Run ID: {run_id}")

    sym_meta = load_symbol_meta(conn)

    for symbol in MINUTE_SYMBOLS:
        meta = sym_meta.get(symbol)
        if meta is None:
            print(f"  SKIP {symbol}: not in symbols table")
            continue
        sym_id    = meta['id']
        tick_size = meta['tick_size'] or 0.0001  # fallback for missing tick

        print(f"\n--- {symbol} (tick={tick_size}) ---")

        minute_df = load_minute_data(conn, sym_id)
        hourly_df = load_hourly_data(conn, sym_id)
        print(f"  Loaded: {len(minute_df) if minute_df is not None else 0} min "
              f"/ {len(hourly_df) if hourly_df is not None else 0} hourly")

        # Compute thresholds in native price units
        thresholds = {mult: mult * tick_size for mult in THRESHOLD_MULTS}

        tf_compare = defaultdict(lambda: defaultdict(list))

        for tf_label, rule, source in TIMEFRAMES:
            base_df = (resample_to_tf(minute_df, rule) if source == 'minute' else
                       (hourly_df.copy() if rule == '1h' else resample_to_tf(hourly_df, rule)))
            if base_df is None or len(base_df) < max(LOOKBACKS) + 10:
                continue

            print(f"  {tf_label:<6} bars={len(base_df)}", end='', flush=True)

            feat_by_lb = {lb: compute_features(base_df, lb) for lb in LOOKBACKS}

            feat_entries = set()
            grid_is  = {}
            grid_oos = {}

            for lb in LOOKBACKS:
                df_feat = feat_by_lb[lb]
                for mult in THRESHOLD_MULTS:
                    thr = thresholds[mult]
                    trades = run_backtest(df_feat, thr)
                    if not trades:
                        continue

                    is_t  = [t for t in trades if t[9] == 'in_sample']
                    oos_t = [t for t in trades if t[9] == 'unseen']

                    if lb == FEAT_LOOKBACK:
                        feat_entries.update(t[0] for t in trades)

                    bulk_insert_trades( cur, run_id, symbol, tf_label, lb, mult, thr, trades)
                    bulk_insert_signals(cur, run_id, symbol, tf_label, lb, mult, thr, trades)

                    is_st  = trade_stats(is_t)
                    oos_st = trade_stats(oos_t)
                    all_st = trade_stats(trades)

                    write_summary(cur, run_id, symbol, tf_label, lb, mult, 'in_sample', is_st)
                    write_summary(cur, run_id, symbol, tf_label, lb, mult, 'unseen',    oos_st)
                    write_summary(cur, run_id, symbol, tf_label, lb, mult, 'combined',  all_st)
                    write_period_cmp(cur, run_id, symbol, tf_label, lb, mult, is_st, oos_st)
                    write_by_year(cur, run_id, symbol, tf_label, lb, mult, trades)
                    write_disp_buckets(cur, run_id, symbol, tf_label, lb, tick_size, trades)

                    grid_is[(lb, mult)]  = {'sharpe': is_st['sharpe']  if is_st  else None,
                                             'n':      is_st['n']       if is_st  else 0}
                    grid_oos[(lb, mult)] = {'sharpe': oos_st['sharpe'] if oos_st else None,
                                             'n':      oos_st['n']      if oos_st else 0}

                    tf_compare[(lb, mult)][tf_label] = (
                        is_st['sharpe']  if is_st  else None,
                        oos_st['sharpe'] if oos_st else None,
                        is_st['n']       if is_st  else 0,
                        oos_st['n']      if oos_st else 0,
                        is_st['wr']      if is_st  else None,
                        oos_st['wr']     if oos_st else None,
                    )

            bulk_insert_features(cur, run_id, symbol, tf_label,
                                 FEAT_LOOKBACK, tick_size, feat_by_lb[FEAT_LOOKBACK], feat_entries)

            write_param_robustness(cur, run_id, symbol, tf_label, 'in_sample', grid_is)
            write_param_robustness(cur, run_id, symbol, tf_label, 'unseen',    grid_oos)

            n_is_total = sum(v['n'] for v in grid_is.values())
            print(f"  IS_trades={n_is_total}")

        for (lb, mult), tf_dict in tf_compare.items():
            tf_data = [(tfl, v[0], v[1], v[2], v[3], v[4], v[5])
                       for tfl, v in tf_dict.items()]
            write_tf_comparison(cur, run_id, symbol, lb, mult, tf_data)

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
    print("GEOM_MR RESULTS SUMMARY")
    print("=" * 70)

    cur.execute("""
        SELECT symbol, timeframe, lookback, threshold_mult,
               sharpe, n_trades, win_rate, mean_return
        FROM strategy_research.geom_mr_analysis_summary
        WHERE run_id = %s AND period_label = 'unseen' AND n_trades >= 5
        ORDER BY sharpe DESC NULLS LAST
        LIMIT 20
    """, (run_id,))
    rows = cur.fetchall()
    print(f"\nTop 20 combos — unseen Sharpe (n≥5):")
    print(f"  {'symbol':<12} {'tf':<7} {'lb':>4} {'mult':>5}  {'sh_oos':>7}  {'n':>5}  {'wr':>6}  {'mean':>9}")
    print("  " + "-" * 68)
    for r in rows:
        sh = f"{r[4]:+.3f}" if r[4] is not None else "   —  "
        print(f"  {r[0]:<12} {r[1]:<7} {r[2]:>4} {r[3]:>5}x  {sh:>7}  {r[5]:>5}  {r[6]:.3f}  {r[7]:+.6f}")

    cur.execute("""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN is_sharpe > 0  THEN 1 ELSE 0 END) AS pos_is,
            SUM(CASE WHEN oos_sharpe > 0 THEN 1 ELSE 0 END) AS pos_oos,
            AVG(sharpe_degradation) FILTER (WHERE sharpe_degradation IS NOT NULL) AS avg_deg,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe_degradation)
                FILTER (WHERE sharpe_degradation IS NOT NULL) AS med_deg
        FROM strategy_research.geom_mr_period_comparison
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

    # Compare with STAT_MR at the same symbol level
    cur.execute("""
        SELECT DISTINCT ON (symbol)
            symbol, timeframe, lookback, threshold_mult, sharpe, n_trades, win_rate
        FROM strategy_research.geom_mr_analysis_summary
        WHERE run_id = %s AND period_label = 'unseen' AND n_trades >= 5
        ORDER BY symbol, sharpe DESC NULLS LAST
    """, (run_id,))
    rows = cur.fetchall()
    print(f"\nBest unseen combo per symbol:")
    print(f"  {'symbol':<12} {'tf':<7} {'lb':>4} {'mult':>5}  {'sh_oos':>7}  {'n':>5}  {'wr':>6}")
    print("  " + "-" * 57)
    for r in rows:
        sh = f"{r[4]:+.3f}" if r[4] is not None else "   —  "
        print(f"  {r[0]:<12} {r[1]:<7} {r[2]:>4} {r[3]:>5}x  {sh:>7}  {r[5]:>5}  {r[6]:.3f}")

    # Head-to-head vs STAT_MR (best unseen per symbol)
    cur.execute("""
        SELECT DISTINCT ON (s.symbol)
            s.symbol,
            s.sharpe AS stat_mr_sh,
            g.sharpe AS geom_mr_sh,
            s.sharpe - g.sharpe AS stat_advantage
        FROM (
            SELECT DISTINCT ON (symbol) symbol, sharpe
            FROM strategy_research.stat_mr_analysis_summary
            WHERE period_label = 'unseen' AND n_trades >= 5
            ORDER BY symbol, sharpe DESC NULLS LAST
        ) s
        JOIN (
            SELECT DISTINCT ON (symbol) symbol, sharpe
            FROM strategy_research.geom_mr_analysis_summary
            WHERE run_id = %s AND period_label = 'unseen' AND n_trades >= 5
            ORDER BY symbol, sharpe DESC NULLS LAST
        ) g ON s.symbol = g.symbol
        ORDER BY s.symbol
    """, (run_id,))
    rows = cur.fetchall()
    if rows:
        print(f"\nSTAT_MR vs GEOM_MR — best unseen Sharpe per symbol:")
        print(f"  {'symbol':<12}  {'stat_sh':>8}  {'geom_sh':>8}  {'stat_adv':>9}")
        print("  " + "-" * 45)
        for r in rows:
            stat_sh = f"{r[1]:+.3f}" if r[1] is not None else "   —  "
            geom_sh = f"{r[2]:+.3f}" if r[2] is not None else "   —  "
            adv     = f"{r[3]:+.3f}" if r[3] is not None else "   —  "
            print(f"  {r[0]:<12}  {stat_sh:>8}  {geom_sh:>8}  {adv:>9}")


if __name__ == '__main__':
    main()

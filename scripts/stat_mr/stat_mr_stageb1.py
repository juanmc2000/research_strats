"""
STAT_MR Stage B1 — Focused 4h/6h Re-run with Side Separation
=============================================================
Issue #12.

Scope:
  timeframes  : 4h, 6h
  lookbacks   : 20, 30
  thresholds  : 2.0, 2.5
  side_filter : long_only | short_only | combined

OOS cutoff revised to 2025-02-01 (more OOS data vs Stage A 2025-06-01).

Kill criteria (immediate stop if ANY triggered):
  1. Median OOS Sharpe across combos < +0.05
  2. Long-only does NOT outperform combined
  3. Short-only shows persistent negative or unstable performance
  4. Best region not stable across symbols
  5. OOS performance driven by < 50 trades per combo

Outputs written to strategy_research schema only.
"""

import os, json, uuid, statistics
from collections import defaultdict
from pathlib import Path

import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ─── Constants ───────────────────────────────────────────────────────────────
OOS_CUTOFF   = pd.Timestamp('2025-02-01', tz='UTC')
LOOKBACKS    = [20, 30]
THRESHOLDS   = [2.0, 2.5]
TIMEFRAMES   = [('4h', '4h', 'hourly'), ('6h', '6h', 'hourly')]
SIDE_FILTERS = ['long_only', 'short_only', 'combined']

MINUTE_SYMBOLS = [
    'AUD/CAD','AUD/CHF','AUD/JPY','AUD/NZD','AUD/USD',
    'EUR/USD','GBP/USD','USD/JPY',
    'AUS200','BTC/USD','BCH/USD','NGAS','XAU/USD',
]

KILL_MIN_OOS_SHARPE  = 0.05
KILL_MIN_OOS_TRADES  = 50

# ─── DB ──────────────────────────────────────────────────────────────────────
def connect():
    root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=root / '.env')
    dsn = os.environ.get('TIMESCALE_DSN')
    return psycopg2.connect(dsn) if dsn else psycopg2.connect(
        host=os.environ['PGHOST'], dbname=os.environ['PGDATABASE'],
        user=os.environ['PGUSER'], password=os.environ['PGPASSWORD'],
        port=int(os.environ.get('PGPORT', 5432)))

def run_migrations(conn):
    root = Path(__file__).resolve().parents[2]
    for f in sorted((root / 'db' / 'strategy_research').glob('*.sql')):
        for stmt in f.read_text().split(';'):
            s = '\n'.join(l for l in stmt.splitlines() if not l.strip().startswith('--')).strip()
            if not s: continue
            try: conn.cursor().execute(s); conn.commit()
            except: conn.rollback()
    print("Migrations verified.")

# ─── Data ────────────────────────────────────────────────────────────────────
def load_symbol_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT symbol, id FROM market_data.symbols")
    return {r[0]: r[1] for r in cur.fetchall()}

def load_hourly(conn, sid):
    cur = conn.cursor()
    cur.execute("""SELECT date,(bid_close+ask_close)/2.0 FROM market_data.hourly_prices
                   WHERE symbol_id=%s ORDER BY date""", (sid,))
    rows = cur.fetchall()
    if not rows: return None
    df = pd.DataFrame(rows, columns=['date','mid_close'])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date').sort_index()
    df['mid_close'] = df['mid_close'].astype(float)
    return df

def resample(df, rule):
    if rule == '1h': return df.copy()
    return df['mid_close'].resample(rule).last().dropna().to_frame()

# ─── Features ────────────────────────────────────────────────────────────────
def compute_features(df, lb):
    p = df['mid_close']
    mean = p.rolling(lb).mean()
    std  = p.rolling(lb).std(ddof=1)
    z    = (p - mean) / std.replace(0, np.nan)
    out  = df.copy()
    out['mean_n'] = mean; out['std_n'] = std; out['z_score'] = z
    return out.dropna(subset=['z_score'])

def period_label(ts):
    return 'unseen' if ts >= OOS_CUTOFF else 'in_sample'

# ─── Backtest ────────────────────────────────────────────────────────────────
def run_backtest(df_feat, threshold, side_filter):
    """side_filter: 'long_only' | 'short_only' | 'combined'"""
    prices = df_feat['mid_close'].values
    zs     = df_feat['z_score'].values
    ts     = df_feat.index
    n      = len(df_feat)

    allow_long  = side_filter in ('long_only',  'combined')
    allow_short = side_filter in ('short_only', 'combined')

    trades = []
    in_pos = False; side = 0
    ei = 0; ep = 0.0; ez = 0.0

    for i in range(n):
        z = zs[i]
        if np.isnan(z): continue
        p = prices[i]
        if not in_pos:
            if allow_long  and z < -threshold: in_pos=True; side=1;  ei=i; ep=p; ez=z
            elif allow_short and z >  threshold: in_pos=True; side=-1; ei=i; ep=p; ez=z
        else:
            if (side==1 and z>=0.0) or (side==-1 and z<=0.0):
                ret = (p-ep)/ep*side
                te  = ts[ei]
                trades.append((te.to_pydatetime(), ts[i].to_pydatetime(),
                                float(ep), float(p), float(ez), float(z),
                                side, i-ei, float(ret), period_label(te)))
                in_pos = False
    return trades

# ─── Stats ───────────────────────────────────────────────────────────────────
def _sharpe(rets):
    if len(rets)<2: return None
    m=statistics.mean(rets); s=statistics.stdev(rets)
    return m/s if s>0 else None

def _median(v):
    if not v: return None
    s=sorted(v); n=len(s)
    return (s[n//2]+s[(n-1)//2])/2.0

def _mdd(rets):
    cum=peak=mdd=0.0
    for r in rets:
        cum+=r
        if cum>peak: peak=cum
        if peak-cum>mdd: mdd=peak-cum
    return mdd

def _payoff(rets):
    w=[r for r in rets if r>0]; l=[r for r in rets if r<0]
    if not w or not l: return None
    return statistics.mean(w)/abs(statistics.mean(l))

def stats(trades):
    if not trades: return None
    rets=[t[8] for t in trades]; bars=[t[7] for t in trades]; n=len(rets)
    return {'n':n,'mean':statistics.mean(rets),'median':_median(rets),
            'std':statistics.stdev(rets) if n>1 else 0.0,
            'sharpe':_sharpe(rets),'wr':sum(1 for r in rets if r>0)/n,
            'payoff':_payoff(rets),'cum':sum(rets),'mdd':_mdd(rets),
            'bars':statistics.mean(bars)}

# ─── DB writes ───────────────────────────────────────────────────────────────
def insert_trades(cur, run_id, sym, tf, lb, thr, sf, trades):
    if not trades: return
    rows = [(run_id,sym,tf,lb,thr,sf)+t for t in trades]
    psycopg2.extras.execute_values(cur, """
        INSERT INTO strategy_research.stat_mr_stageb_trades
        (run_id,symbol,timeframe,lookback,threshold,side_filter,
         entry_ts,exit_ts,entry_price,exit_price,z_at_entry,z_at_exit,
         side,bars_held,return_pct,period_label)
        VALUES %s ON CONFLICT DO NOTHING""", rows, page_size=2000)

def insert_analysis(cur, run_id, sym, tf, lb, thr, sf, period, st):
    if st is None: return
    cur.execute("""
        INSERT INTO strategy_research.stat_mr_stageb_analysis
        (run_id,symbol,timeframe,lookback,threshold,side_filter,period_label,
         n_trades,mean_return,median_return,std_return,sharpe,win_rate,
         payoff,cum_return,max_drawdown,avg_bars_held)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT DO NOTHING""",
        (run_id,sym,tf,lb,thr,sf,period,
         st['n'],st['mean'],st['median'],st['std'],st['sharpe'],
         st['wr'],st['payoff'],st['cum'],st['mdd'],st['bars']))

# ─── Kill criteria ───────────────────────────────────────────────────────────
def evaluate_kill_criteria(conn, run_id):
    cur = conn.cursor()
    kills = []

    # 1. Median OOS Sharpe < 0.05 (long_only)
    cur.execute("""
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe) AS med_sh
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND side_filter='long_only'
          AND n_trades IS NOT NULL
    """, (run_id,))
    r = cur.fetchone()
    med_sh = float(r[0]) if r and r[0] is not None else None
    if med_sh is None or med_sh < KILL_MIN_OOS_SHARPE:
        kills.append(f"KILL-1: median OOS long_only Sharpe = {med_sh} < {KILL_MIN_OOS_SHARPE}")

    # 2. Long-only must outperform combined (median OOS Sharpe)
    cur.execute("""
        SELECT side_filter,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe) AS med
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND side_filter IN ('long_only','combined')
          AND n_trades IS NOT NULL
        GROUP BY side_filter
    """, (run_id,))
    sh_map = {r[0]: float(r[1]) for r in cur.fetchall() if r[1] is not None}
    lo = sh_map.get('long_only'); co = sh_map.get('combined')
    if lo is not None and co is not None and lo <= co:
        kills.append(f"KILL-2: long_only OOS Sharpe ({lo:.3f}) does not outperform combined ({co:.3f})")

    # 3. Short-only persistent negative
    cur.execute("""
        SELECT SUM(CASE WHEN sharpe<0 THEN 1 ELSE 0 END), COUNT(*)
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND side_filter='short_only'
          AND n_trades IS NOT NULL
    """, (run_id,))
    r = cur.fetchone()
    if r and r[1] and r[1] > 0 and float(r[0])/r[1] > 0.60:
        kills.append(f"KILL-3: short_only OOS negative in {r[0]}/{r[1]} ({100*r[0]//r[1]}%) combos")

    # 4. Best region stability — check if top-3 symbols dominate vs spread
    cur.execute("""
        SELECT symbol, AVG(sharpe) AS avg_sh
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND side_filter='long_only'
          AND n_trades IS NOT NULL
        GROUP BY symbol ORDER BY avg_sh DESC
    """, (run_id,))
    sym_rows = cur.fetchall()
    if sym_rows:
        pos_syms = sum(1 for r in sym_rows if r[1] is not None and r[1] > 0)
        if pos_syms < len(sym_rows) / 2:
            kills.append(f"KILL-4: only {pos_syms}/{len(sym_rows)} symbols have positive OOS Sharpe")

    # 5. OOS trades per combo < 50
    cur.execute("""
        SELECT AVG(n_trades), MIN(n_trades)
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND side_filter='long_only'
          AND n_trades IS NOT NULL
    """, (run_id,))
    r = cur.fetchone()
    if r and r[1] is not None and r[1] < KILL_MIN_OOS_TRADES:
        kills.append(f"KILL-5: minimum OOS trades per combo = {r[1]} < {KILL_MIN_OOS_TRADES}")

    return kills

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    conn = connect()
    run_migrations(conn)
    cur = conn.cursor()

    run_id = str(uuid.uuid4())
    config = {'lookbacks':LOOKBACKS,'thresholds':THRESHOLDS,
              'timeframes':[t[0] for t in TIMEFRAMES],
              'side_filters':SIDE_FILTERS,'oos_cutoff':str(OOS_CUTOFF.date())}
    cur.execute("""INSERT INTO strategy_research.experiment_runs
        (run_id,strategy_name,config,notes) VALUES(%s,%s,%s,%s)""",
        (run_id,'STAT_MR_B1',json.dumps(config),'Issue #12 — Stage B1'))
    conn.commit()
    print(f"Run ID: {run_id}")

    sym_ids = load_symbol_ids(conn)

    for symbol in MINUTE_SYMBOLS:
        sid = sym_ids.get(symbol)
        if sid is None: continue
        print(f"\n--- {symbol} ---")
        hourly_df = load_hourly(conn, sid)
        if hourly_df is None: continue

        for tf_label, rule, _ in TIMEFRAMES:
            base_df = resample(hourly_df, rule)
            if base_df is None or len(base_df) < max(LOOKBACKS)+10: continue

            for lb in LOOKBACKS:
                df_feat = compute_features(base_df, lb)
                for thr in THRESHOLDS:
                    for sf in SIDE_FILTERS:
                        trades = run_backtest(df_feat, thr, sf)
                        if not trades: continue
                        is_t  = [t for t in trades if t[9]=='in_sample']
                        oos_t = [t for t in trades if t[9]=='unseen']
                        insert_trades(cur, run_id, symbol, tf_label, lb, thr, sf, trades)
                        insert_analysis(cur, run_id, symbol, tf_label, lb, thr, sf, 'in_sample', stats(is_t))
                        insert_analysis(cur, run_id, symbol, tf_label, lb, thr, sf, 'unseen',    stats(oos_t))
                        insert_analysis(cur, run_id, symbol, tf_label, lb, thr, sf, 'combined',  stats(trades))
                n_is  = sum(1 for t in run_backtest(df_feat,THRESHOLDS[0],'long_only') if t[9]=='in_sample')
            print(f"  {tf_label:<4} lb={lb}  done", flush=True)

        conn.commit()
        print(f"  committed.")

    conn.commit()
    print(f"\nRun complete. Run ID: {run_id}")

    # ── Kill criteria ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("KILL CRITERIA EVALUATION")
    print("="*70)
    kills = evaluate_kill_criteria(conn, run_id)
    if kills:
        verdict = 'KILL'
        print(f"\n  *** KILL — {len(kills)} criterion/criteria triggered ***")
        for k in kills: print(f"  {k}")
    else:
        verdict = 'PASS'
        print("\n  PASS — all kill criteria cleared")

    cur.execute("""INSERT INTO strategy_research.stat_mr_stageb_verdict
        (run_id,stage,verdict,reason) VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
        (run_id,'B1',verdict,'; '.join(kills) if kills else 'All criteria met'))
    conn.commit()

    _print_summary(conn, run_id)
    conn.close()
    return verdict

# ─── Summary ────────────────────────────────────────────────────────────────
def _print_summary(conn, run_id):
    cur = conn.cursor()
    print("\n" + "="*70)
    print("STAT_MR STAGE B1 — RESULTS SUMMARY")
    print("="*70)

    # Overall IS vs OOS by side_filter
    print(f"\n{'side_filter':<13}  {'period':<10}  {'med_sh':>7}  {'avg_sh':>7}  {'pct_pos':>8}  {'med_wr':>7}  {'med_n':>7}")
    print("  " + "-"*60)
    cur.execute("""
        SELECT side_filter, period_label,
               ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe)::numeric,3),
               ROUND(AVG(sharpe)::numeric,3),
               ROUND(100.0*SUM(CASE WHEN sharpe>0 THEN 1 ELSE 0 END)/COUNT(*)::numeric,1),
               ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY win_rate)::numeric,3),
               ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY n_trades)::numeric,0)
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND n_trades IS NOT NULL
        GROUP BY side_filter, period_label
        ORDER BY side_filter, period_label
    """, (run_id,))
    for r in cur.fetchall():
        print(f"  {r[0]:<13}  {r[1]:<10}  {r[2]:>7}  {r[3]:>7}  {r[4]:>7}%  {r[5]:>7}  {r[6]:>7.0f}")

    # Per symbol, long_only OOS
    print(f"\nLong-only unseen Sharpe per symbol:")
    print(f"  {'symbol':<12}  {'tf':<5}  {'lb':>4}  {'thr':>5}  {'sh_is':>7}  {'sh_oos':>7}  {'n_oos':>6}  {'wr_oos':>7}")
    print("  " + "-"*65)
    cur.execute("""
        SELECT DISTINCT ON (a_oos.symbol)
               a_oos.symbol, a_oos.timeframe, a_oos.lookback, a_oos.threshold,
               a_is.sharpe, a_oos.sharpe, a_oos.n_trades, a_oos.win_rate
        FROM strategy_research.stat_mr_stageb_analysis a_oos
        JOIN strategy_research.stat_mr_stageb_analysis a_is
          ON a_is.run_id=a_oos.run_id AND a_is.symbol=a_oos.symbol
         AND a_is.timeframe=a_oos.timeframe AND a_is.lookback=a_oos.lookback
         AND a_is.threshold=a_oos.threshold AND a_is.side_filter=a_oos.side_filter
         AND a_is.period_label='in_sample'
        WHERE a_oos.run_id=%s AND a_oos.period_label='unseen'
          AND a_oos.side_filter='long_only' AND a_oos.n_trades IS NOT NULL
        ORDER BY a_oos.symbol, a_oos.sharpe DESC NULLS LAST
    """, (run_id,))
    for r in cur.fetchall():
        sh_is  = f"{r[4]:+.3f}" if r[4] is not None else "   —  "
        sh_oos = f"{r[5]:+.3f}" if r[5] is not None else "   —  "
        print(f"  {r[0]:<12}  {r[1]:<5}  {r[2]:>4}  {float(r[3]):>5.1f}  {sh_is:>7}  {sh_oos:>7}  {r[6]:>6}  {r[7]:.3f}")

    # Long vs short vs combined (OOS, median across all combos)
    print(f"\nLong vs Short vs Combined (OOS median Sharpe by TF):")
    print(f"  {'tf':<5}  {'long_only':>10}  {'short_only':>11}  {'combined':>9}")
    print("  " + "-"*40)
    cur.execute("""
        SELECT timeframe, side_filter,
               ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY sharpe)::numeric,3)
        FROM strategy_research.stat_mr_stageb_analysis
        WHERE run_id=%s AND period_label='unseen' AND n_trades IS NOT NULL
        GROUP BY timeframe, side_filter ORDER BY timeframe, side_filter
    """, (run_id,))
    tf_sf = defaultdict(dict)
    for r in cur.fetchall(): tf_sf[r[0]][r[1]] = r[2]
    for tf in sorted(tf_sf):
        lo = tf_sf[tf].get('long_only')
        so = tf_sf[tf].get('short_only')
        co = tf_sf[tf].get('combined')
        lo_s = f"{float(lo):+.3f}" if lo is not None else "   —  "
        so_s = f"{float(so):+.3f}" if so is not None else "   —  "
        co_s = f"{float(co):+.3f}" if co is not None else "   —  "
        print(f"  {tf:<5}  {lo_s:>10}  {so_s:>11}  {co_s:>9}")

if __name__ == '__main__':
    verdict = main()
    import sys
    sys.exit(0 if verdict == 'PASS' else 1)

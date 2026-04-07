"""
CTU rolling walk-forward OOS — Issue #20
Locked signal: vol_ratio_20_100 < vol_lo, efficiency_20 > eff_hi, breakout_direction='UP'
Return: realized_return_sd30d - COST
Expanding window: train = events before test_year, test = events in test_year.
"""

import os
import sys
import uuid
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctu.constants import (
    CTU_SYMBOLS, VOL_CONTRACTING_MAX, EFF_TRENDING_MIN,
    MEDIUM_COST, N_BOOTSTRAP, BLOCK_SIZE,
    THRESHOLD_SETS,
)

# ── Constants ────────────────────────────────────────────────────────────────
COST        = MEDIUM_COST
TEST_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025]

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
def load_ctu_events(conn, vol_lo, eff_hi):
    """Load all CTU-qualifying events across the universe."""
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol,
               EXTRACT(YEAR FROM event_hour_ts)::int AS yr,
               realized_return_sd30d - %(cost)s       AS ret
        FROM features.breakout_events
        WHERE symbol = ANY(%(syms)s)
          AND exit_reason IS NOT NULL
          AND entry_range_sd_30d > 0
          AND vol_ratio_20_100 < %(vol_lo)s
          AND efficiency_20    > %(eff_hi)s
          AND breakout_direction = 'UP'
          AND realized_return_sd30d IS NOT NULL
        ORDER BY symbol, event_hour_ts
    """, {'cost': COST, 'syms': CTU_SYMBOLS, 'vol_lo': vol_lo, 'eff_hi': eff_hi})
    rows = cur.fetchall()
    # Returns dict: {symbol: {year: [ret, ...]}}
    events = {}
    for sym, yr, ret in rows:
        yr = int(yr)
        if sym not in events:
            events[sym] = {}
        if yr not in events[sym]:
            events[sym][yr] = []
        events[sym][yr].append(float(ret))
    return events

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(returns):
    """
    Returns are in realized_return_sd30d units (vol-normalized), not raw %.
    Use arithmetic aggregation — do NOT compound geometrically.
    cum = sum of returns (total accumulated vol-normalized points)
    dd  = max drawdown on running cumulative sum (arithmetic equity curve)
    """
    if not returns:
        return dict(n=0, mean=None, sharpe=None, wr=None, cum=None, dd=None)
    r = np.array(returns)
    n = len(r)
    mean   = float(r.mean())
    std    = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    wr     = float((r > 0).mean())
    cum    = float(r.sum())                   # arithmetic total
    eq     = np.cumsum(r)                     # arithmetic equity curve
    peak   = np.maximum.accumulate(eq)
    drawdowns = peak - eq
    dd     = float(drawdowns.max()) if n > 1 else 0.0
    return dict(n=n, mean=mean, sharpe=sharpe, wr=wr, cum=cum, dd=dd)

# ── Block bootstrap ───────────────────────────────────────────────────────────
def bootstrap_sharpe(returns, n_boot=N_BOOTSTRAP, block=BLOCK_SIZE, seed=42):
    r = np.array(returns)
    n = len(r)
    if n < 10:
        return None, None, None
    rng = np.random.default_rng(seed)
    boot_sharpes = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=(n // block) + 1)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        std = sample.std(ddof=1)
        boot_sharpes.append(sample.mean() / std if std > 0 else 0.0)
    bs = np.array(boot_sharpes)
    return float(np.percentile(bs, 5)), float(np.percentile(bs, 50)), float(np.percentile(bs, 95))

def safe_corr(xs, ys):
    if len(xs) < 3:
        return None
    x, y = np.array(xs, float), np.array(ys, float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

# ── Insert helpers ────────────────────────────────────────────────────────────
def upsert_run(conn, run_id, config):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO strategy_research.experiment_runs
            (run_id, strategy_name, run_ts, config, notes)
        VALUES (%s, 'CTU_ROLLING_OOS', NOW(), %s, 'Issue #20 CTU rolling OOS')
        ON CONFLICT (run_id) DO NOTHING
    """, (str(run_id), json.dumps(config)))
    conn.commit()

def insert_folds(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO strategy_research.ctu_rolling_folds
            (run_id, threshold_set, test_year, scope,
             train_n, train_mean_ret, train_sharpe, train_win_rate, train_cum_ret,
             test_n,  test_mean_ret,  test_sharpe,  test_win_rate,  test_cum_ret,
             test_max_dd,
             boot_sharpe_p05, boot_sharpe_p50, boot_sharpe_p95)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    psycopg2.extras.execute_values(conn.cursor(), sql, rows, page_size=500)
    conn.commit()

def insert_summary(conn, rows):
    if not rows:
        return
    sql = """
        INSERT INTO strategy_research.ctu_rolling_summary
            (run_id, threshold_set, scope,
             n_folds, mean_test_sharpe, median_test_sharpe, std_test_sharpe,
             min_test_sharpe, max_test_sharpe, pct_pos_folds,
             mean_test_wr, mean_test_cum, mean_train_sharpe, train_test_corr,
             total_test_n)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    psycopg2.extras.execute_values(conn.cursor(), sql, rows, page_size=200)
    conn.commit()

# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate_threshold_set(conn, run_id, ts_name, ts_params, events):
    """Run all folds for one threshold set and insert results."""
    fold_rows = []

    # Scopes: 'combined' + each symbol
    scopes = ['combined'] + CTU_SYMBOLS

    for test_year in TEST_YEARS:
        for scope in scopes:
            # Gather returns
            if scope == 'combined':
                train_rets = [r for sym in CTU_SYMBOLS
                              for yr, rs in events.get(sym, {}).items()
                              if yr < test_year for r in rs]
                test_rets  = [r for sym in CTU_SYMBOLS
                              for rs in ([events.get(sym, {}).get(test_year, [])])
                              for r in rs]
            else:
                sym_data = events.get(scope, {})
                train_rets = [r for yr, rs in sym_data.items() if yr < test_year for r in rs]
                test_rets  = sym_data.get(test_year, [])

            tm = compute_metrics(train_rets)
            ym = compute_metrics(test_rets)
            bp05, bp50, bp95 = bootstrap_sharpe(test_rets)

            fold_rows.append((
                str(run_id), ts_name, test_year, scope,
                tm['n'],      tm['mean'],   tm['sharpe'],  tm['wr'],   tm['cum'],
                ym['n'],      ym['mean'],   ym['sharpe'],  ym['wr'],   ym['cum'],
                ym['dd'],
                bp05, bp50, bp95,
            ))

    insert_folds(conn, fold_rows)

    # Build summaries
    summary_rows = []
    for scope in scopes:
        fold_results = [r for r in fold_rows if r[3] == scope]

        # tuple indices: 0=run_id,1=ts,2=yr,3=scope,4=train_n,5=train_mean,
        # 6=train_sharpe,7=train_wr,8=train_cum,9=test_n,10=test_mean,
        # 11=test_sharpe,12=test_wr,13=test_cum,14=test_dd,15=bp05,16=bp50,17=bp95
        train_sharpes = [r[6]  for r in fold_results if r[6]  is not None]
        test_sharpes  = [r[11] for r in fold_results if r[11] is not None and r[9] > 0]
        test_wrs      = [r[12] for r in fold_results if r[12] is not None and r[9] > 0]
        test_cums     = [r[13] for r in fold_results if r[13] is not None and r[9] > 0]
        total_n       = sum(r[9] for r in fold_results if r[9])

        if not test_sharpes:
            continue

        arr = np.array(test_sharpes)
        n_f = len(arr)
        corr = safe_corr(train_sharpes[-n_f:], test_sharpes[:len(train_sharpes)])

        summary_rows.append((
            str(run_id), ts_name, scope,
            n_f,
            float(arr.mean()),
            float(np.median(arr)),
            float(arr.std(ddof=1)) if n_f > 1 else None,
            float(arr.min()),
            float(arr.max()),
            float((arr > 0).mean()),
            float(np.mean(test_wrs)) if test_wrs else None,
            float(np.mean(test_cums)) if test_cums else None,
            float(np.mean(train_sharpes)) if train_sharpes else None,
            corr,
            total_n,
        ))

    insert_summary(conn, summary_rows)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn   = get_conn()
    run_id = uuid.uuid4()

    config = {
        'test_years': TEST_YEARS,
        'threshold_sets': {k: v for k, v in THRESHOLD_SETS.items()},
        'cost': COST,
        'symbols': CTU_SYMBOLS,
    }

    print(f'run_id: {run_id}')
    print(f'Start: {datetime.now(timezone.utc).isoformat()}')

    run_migrations(conn)
    upsert_run(conn, run_id, config)

    for ts_name, ts_params in THRESHOLD_SETS.items():
        print(f'\nLoading events: {ts_name}  vol_lo={ts_params["vol_lo"]}  eff_hi={ts_params["eff_hi"]}')
        events = load_ctu_events(conn, ts_params['vol_lo'], ts_params['eff_hi'])
        total = sum(len(rs) for sym_d in events.values() for rs in sym_d.values())
        print(f'  Total qualifying events: {total}')
        evaluate_threshold_set(conn, run_id, ts_name, ts_params, events)
        print(f'  Done.')

    print(f'\nDone: {datetime.now(timezone.utc).isoformat()}')

    # ── Print results ─────────────────────────────────────────────────────────
    cur = conn.cursor()

    print('\n' + '='*70)
    print('CTU ROLLING OOS — COMBINED PORTFOLIO VIEW')
    print('='*70)

    # Summary table: threshold × scope=combined
    cur.execute("""
        SELECT threshold_set, n_folds,
               ROUND(mean_test_sharpe::numeric,3)   AS mean_S,
               ROUND(median_test_sharpe::numeric,3) AS med_S,
               ROUND(std_test_sharpe::numeric,3)    AS std_S,
               ROUND(pct_pos_folds::numeric,3)      AS pct_pos,
               ROUND(mean_test_wr::numeric,3)       AS mean_wr,
               ROUND(mean_test_cum::numeric,4)      AS mean_cum,
               ROUND(train_test_corr::numeric,3)    AS corr,
               total_test_n
        FROM strategy_research.ctu_rolling_summary
        WHERE run_id = %s AND scope = 'combined'
        ORDER BY threshold_set
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"\n{'thr_set':<10} {'folds':>5} {'mean_S':>8} {'med_S':>8} {'std_S':>7} {'pct+':>6} {'wr':>6} {'cum':>8} {'IS/OOS':>7} {'total_n':>7}")
    print('-' * 76)
    for r in rows:
        ts, nf, ms, meds, stds, pp, wr, cum, corr, tn = r
        corr_s = f'{float(corr):>+.2f}' if corr is not None else '  — '
        print(f"{ts:<10} {nf:>5} {float(ms):>+8.3f} {float(meds):>+8.3f} {float(stds):>7.3f} {float(pp):>6.1%} {float(wr):>6.1%} {float(cum):>+8.4f} {corr_s:>7} {tn:>7}")

    # Per fold detail (baseline combined)
    print(f'\n--- Baseline: per test year (combined) ---')
    cur.execute("""
        SELECT test_year,
               train_n, ROUND(train_sharpe::numeric,3)  AS train_S,
               test_n,  ROUND(test_sharpe::numeric,3)   AS test_S,
               ROUND(test_win_rate::numeric,3)           AS wr,
               ROUND(test_cum_ret::numeric,4)            AS cum,
               ROUND(boot_sharpe_p05::numeric,3)         AS bp05,
               ROUND(boot_sharpe_p95::numeric,3)         AS bp95
        FROM strategy_research.ctu_rolling_folds
        WHERE run_id = %s AND threshold_set = 'baseline' AND scope = 'combined'
        ORDER BY test_year
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'year':>5} {'train_n':>7} {'train_S':>8} {'test_n':>6} {'test_S':>8} {'wr':>6} {'cum':>8} {'[p05':>6} {'p95]':>6}")
    print('-' * 68)
    for r in rows:
        yr, tn, ts, yn, ys, wr, cum, bp05, bp95 = r
        ts_s  = f'{float(ts):>+8.3f}' if ts is not None else '      —  '
        ys_s  = f'{float(ys):>+8.3f}' if ys is not None else '      —  '
        wr_s  = f'{float(wr):>6.1%}' if wr is not None else '   — '
        cum_s = f'{float(cum):>+8.4f}' if cum is not None else '      —  '
        bp05_s = f'{float(bp05):>+6.3f}' if bp05 is not None else '  — '
        bp95_s = f'{float(bp95):>+6.3f}' if bp95 is not None else '  — '
        print(f"{yr:>5} {tn if tn else 0:>7} {ts_s} {yn if yn else 0:>6} {ys_s} {wr_s} {cum_s} {bp05_s} {bp95_s}")

    # Per symbol summary (baseline)
    print(f'\n--- Baseline: per symbol summary ---')
    cur.execute("""
        SELECT scope,
               n_folds,
               ROUND(mean_test_sharpe::numeric,3)   AS mean_S,
               ROUND(median_test_sharpe::numeric,3) AS med_S,
               ROUND(pct_pos_folds::numeric,3)      AS pct_pos,
               ROUND(mean_test_wr::numeric,3)       AS mean_wr,
               total_test_n
        FROM strategy_research.ctu_rolling_summary
        WHERE run_id = %s AND threshold_set = 'baseline'
          AND scope != 'combined'
        ORDER BY median_test_sharpe DESC NULLS LAST
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"{'symbol':<12} {'folds':>5} {'mean_S':>8} {'med_S':>8} {'pct+':>6} {'wr':>6} {'n':>5}")
    print('-' * 54)
    for r in rows:
        sc, nf, ms, meds, pp, wr, tn = r
        ms_s   = f'{float(ms):>+8.3f}'   if ms   is not None else '      —  '
        meds_s = f'{float(meds):>+8.3f}' if meds is not None else '      —  '
        pp_s   = f'{float(pp):>6.1%}'    if pp   is not None else '   —  '
        wr_s   = f'{float(wr):>6.1%}'    if wr   is not None else '   —  '
        print(f"{sc:<12} {nf:>5} {ms_s} {meds_s} {pp_s} {wr_s} {tn:>5}")

    # Per symbol × year detail (baseline)
    print(f'\n--- Baseline: per symbol per year (test Sharpe) ---')
    cur.execute("""
        SELECT scope, test_year,
               test_n,
               ROUND(test_sharpe::numeric,3)   AS test_S,
               ROUND(test_win_rate::numeric,3) AS wr
        FROM strategy_research.ctu_rolling_folds
        WHERE run_id = %s AND threshold_set = 'baseline'
          AND scope != 'combined'
        ORDER BY scope, test_year
    """, (str(run_id),))
    rows = cur.fetchall()

    # Pivot
    from collections import defaultdict
    data = defaultdict(dict)
    wrs  = defaultdict(dict)
    ns   = defaultdict(dict)
    syms = set(); years = set()
    for sc, yr, n, s, w in rows:
        data[sc][yr] = float(s) if s is not None else None
        wrs[sc][yr]  = float(w) if w is not None else None
        ns[sc][yr]   = n or 0
        syms.add(sc); years.add(yr)
    syms  = sorted(syms)
    years = sorted(years)

    header = f"{'symbol':<12}" + ''.join(f"{y:>9}" for y in years)
    print(header)
    print('-' * (12 + 9*len(years)))
    for sym in syms:
        row = f"{sym:<12}"
        for yr in years:
            v = data[sym].get(yr)
            n = ns[sym].get(yr, 0)
            if v is not None and n > 0:
                row += f" {v:>+7.3f}({n:<2})"
            else:
                row += f"{'—':>12}"
        print(row)

    conn.close()

if __name__ == '__main__':
    main()

# ETD ARCHIVE — reference only. Do not modify without a new issue.
"""
ETD frozen rolling walk-forward OOS — Issue #22
Symbols: USD/CHF and GBP/USD.
Configs: raw_etd (all ETD), atr_filtered_etd (exclude ATR TRANSITION_1_2 bucket).
Expanding window: train = events before test_year, test = events in test_year.
"""

import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ── Frozen ETD constants (from ETD archive — do not change) ──────────────────
VOL_EXPANDING_MIN  = 1.20
EFF_TRENDING_MIN   = 0.60
MEDIUM_COST        = 0.07
ATR_WINDOW         = 30
BUCKET_TRANSITION  = 'TRANSITION_1_2'
N_BOOTSTRAP        = 1000
BLOCK_SIZE         = 20

# ── Scope ─────────────────────────────────────────────────────────────────────
SYMBOLS    = ['USD/CHF', 'GBP/USD']
CONFIGS       = ['raw_etd', 'atr_filtered_etd']
TEST_YEARS    = [2020, 2021, 2022, 2023, 2024, 2025]
MIN_TEST_N    = 5   # minimum test trades to include fold in summary aggregation

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

# ── ATR bucket logic (frozen from ETD archive) ────────────────────────────────
def _atr_bucket(zscore: float) -> str:
    if zscore < 0.0:      return 'LOW'
    if zscore < 1.0:      return 'NORMAL'
    if zscore < 2.0:      return BUCKET_TRANSITION
    return 'SPIKE_GE_2'

def build_atr_lookup(conn, symbol):
    cur = conn.cursor()
    cur.execute("""
        SELECT day_ts, daily_range
        FROM features.daily_market_metrics
        WHERE symbol = %s AND daily_range IS NOT NULL AND daily_range > 0
        ORDER BY day_ts
    """, (symbol,))
    rows = cur.fetchall()
    lookup = {}
    for i, (day_ts, rng) in enumerate(rows):
        if i < ATR_WINDOW - 1:
            continue
        window = [float(rows[j][1]) for j in range(i - ATR_WINDOW + 1, i + 1)]
        mean = sum(window) / ATR_WINDOW
        var  = sum((v - mean) ** 2 for v in window) / (ATR_WINDOW - 1)
        std  = math.sqrt(var)
        if std == 0.0:
            continue
        zscore = (float(rng) - mean) / std
        lookup[day_ts] = _atr_bucket(zscore)
    return lookup

# ── Data loading ──────────────────────────────────────────────────────────────
def load_etd_events(conn, symbol, atr_lookup):
    """
    Load all ETD-qualifying events with ATR bucket annotation.
    Returns list of dicts: {year, ret, fail_atr}
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT event_hour_ts,
               vol_ratio_20_100,
               efficiency_20,
               breakout_direction,
               realized_return_sd30d
        FROM features.breakout_events
        WHERE symbol = %s
          AND exit_reason IS NOT NULL
          AND entry_range_sd_30d > 0
          AND vol_ratio_20_100 IS NOT NULL
          AND efficiency_20    IS NOT NULL
          AND realized_return_sd30d IS NOT NULL
          AND shock_1h_sd IS NOT NULL
        ORDER BY event_hour_ts
    """, (symbol,))
    rows = cur.fetchall()

    events = []
    for (ts, vr, ef, direction, raw_ret) in rows:
        vr = float(vr); ef = float(ef)
        # Frozen ETD state check
        vol_state = 'EXPANDING' if vr > VOL_EXPANDING_MIN else ('NEUTRAL' if vr >= 0.80 else 'CONTRACTING')
        eff_state = 'TRENDING'  if ef > EFF_TRENDING_MIN  else ('MIXED'   if ef >= 0.30  else 'CHOPPY')
        state     = f'{vol_state}_{eff_state}_{direction}'
        if state != 'EXPANDING_TRENDING_DOWN':
            continue

        trade_date = ts.date() if hasattr(ts, 'date') else ts
        atr_bucket = atr_lookup.get(trade_date)
        fail_atr   = (atr_bucket == BUCKET_TRANSITION)
        year       = ts.year if hasattr(ts, 'year') else int(str(ts)[:4])
        ret        = float(raw_ret) - MEDIUM_COST

        events.append({'year': year, 'ret': ret, 'fail_atr': fail_atr})

    return events

def apply_config(events, config):
    if config == 'raw_etd':
        return events
    if config == 'atr_filtered_etd':
        return [e for e in events if not e['fail_atr']]
    raise ValueError(f'unknown config: {config}')

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(returns):
    if not returns:
        return dict(n=0, mean=None, sharpe=None, wr=None, cum=None, dd=None)
    r = np.array(returns)
    n = len(r)
    mean   = float(r.mean())
    std    = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    wr     = float((r > 0).mean())
    cum    = float(r.sum())
    eq     = np.cumsum(r)
    peak   = np.maximum.accumulate(eq)
    dd     = float((peak - eq).max()) if n > 1 else 0.0
    return dict(n=n, mean=mean, sharpe=sharpe, wr=wr, cum=cum, dd=dd)

def bootstrap_sharpe(returns, n_boot=N_BOOTSTRAP, block=BLOCK_SIZE, seed=42):
    r = np.array(returns)
    n = len(r)
    if n < 10:
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=(n // block) + 1)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        std = sample.std(ddof=1)
        boot.append(sample.mean() / std if std > 0 else 0.0)
    bs = np.array(boot)
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
        VALUES (%s, 'ETD_ROLLING_OOS', NOW(), %s, 'Issue #22 ETD rolling OOS')
        ON CONFLICT (run_id) DO NOTHING
    """, (str(run_id), json.dumps(config)))
    conn.commit()

def insert_folds(conn, rows):
    if not rows: return
    psycopg2.extras.execute_values(conn.cursor(), """
        INSERT INTO strategy_research.etd_rolling_folds
            (run_id, symbol, config, test_year,
             train_n, train_mean_ret, train_sharpe, train_win_rate, train_cum_ret,
             test_n,  test_mean_ret,  test_sharpe,  test_win_rate,  test_cum_ret,
             test_max_dd, boot_sharpe_p05, boot_sharpe_p50, boot_sharpe_p95)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=200)
    conn.commit()

def insert_summary(conn, rows):
    if not rows: return
    psycopg2.extras.execute_values(conn.cursor(), """
        INSERT INTO strategy_research.etd_rolling_summary
            (run_id, symbol, config, n_folds,
             mean_test_sharpe, median_test_sharpe, std_test_sharpe,
             min_test_sharpe, max_test_sharpe, pct_pos_folds,
             mean_test_wr, mean_test_cum, mean_train_sharpe, train_test_corr,
             total_test_n)
        VALUES %s ON CONFLICT DO NOTHING
    """, rows, page_size=100)
    conn.commit()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    conn   = get_conn()
    run_id = uuid.uuid4()

    run_config = {'symbols': SYMBOLS, 'configs': CONFIGS, 'test_years': TEST_YEARS,
                  'vol_expanding_min': VOL_EXPANDING_MIN, 'eff_trending_min': EFF_TRENDING_MIN,
                  'cost': MEDIUM_COST, 'atr_window': ATR_WINDOW}

    print(f'run_id: {run_id}')
    print(f'Start: {datetime.now(timezone.utc).isoformat()}')

    run_migrations(conn)
    upsert_run(conn, run_id, run_config)

    all_fold_rows    = []
    all_summary_rows = []

    for sym in SYMBOLS:
        print(f'\n{sym}')
        atr_lookup = build_atr_lookup(conn, sym)
        raw_events = load_etd_events(conn, sym, atr_lookup)

        for cfg in CONFIGS:
            events = apply_config(raw_events, cfg)
            total  = len(events)
            print(f'  {cfg}: {total} qualifying events')

            fold_rows     = []
            train_sharpes = []
            test_results  = []

            for test_year in TEST_YEARS:
                train_rets = [e['ret'] for e in events if e['year'] < test_year]
                test_rets  = [e['ret'] for e in events if e['year'] == test_year]

                tm = compute_metrics(train_rets)
                ym = compute_metrics(test_rets)
                bp05, bp50, bp95 = bootstrap_sharpe(test_rets)

                fold_rows.append((
                    str(run_id), sym, cfg, test_year,
                    tm['n'],   tm['mean'],   tm['sharpe'],  tm['wr'],   tm['cum'],
                    ym['n'],   ym['mean'],   ym['sharpe'],  ym['wr'],   ym['cum'],
                    ym['dd'],  bp05, bp50, bp95,
                ))

                if tm['sharpe'] is not None:
                    train_sharpes.append(tm['sharpe'])
                if ym['sharpe'] is not None and ym['n'] >= MIN_TEST_N:
                    test_results.append((ym['sharpe'], ym['wr'], ym['cum']))

            all_fold_rows.extend(fold_rows)

            if test_results:
                s_arr  = np.array([x[0] for x in test_results])
                wrs    = [x[1] for x in test_results]
                cums   = [x[2] for x in test_results]
                n_f    = len(s_arr)
                corr   = safe_corr(train_sharpes[-n_f:], s_arr.tolist()[:len(train_sharpes)])
                total_n = sum(r[9] for r in fold_rows if r[9])

                all_summary_rows.append((
                    str(run_id), sym, cfg, n_f,
                    float(s_arr.mean()),
                    float(np.median(s_arr)),
                    float(s_arr.std(ddof=1)) if n_f > 1 else None,
                    float(s_arr.min()),
                    float(s_arr.max()),
                    float((s_arr > 0).mean()),
                    float(np.mean(wrs)),
                    float(np.mean(cums)),
                    float(np.mean(train_sharpes)) if train_sharpes else None,
                    corr,
                    total_n,
                ))

    insert_folds(conn, all_fold_rows)
    insert_summary(conn, all_summary_rows)

    print(f'\nDone: {datetime.now(timezone.utc).isoformat()}')

    # ── Results printout ──────────────────────────────────────────────────────
    cur = conn.cursor()

    print('\n' + '='*70)
    print('ETD ROLLING OOS — SUMMARY')
    print('='*70)

    cur.execute("""
        SELECT symbol, config, n_folds,
               ROUND(mean_test_sharpe::numeric,3)   AS mean_S,
               ROUND(median_test_sharpe::numeric,3) AS med_S,
               ROUND(std_test_sharpe::numeric,3)    AS std_S,
               ROUND(pct_pos_folds::numeric,3)      AS pct_pos,
               ROUND(mean_test_wr::numeric,3)       AS mean_wr,
               ROUND(mean_test_cum::numeric,3)      AS mean_cum,
               ROUND(train_test_corr::numeric,3)    AS corr,
               total_test_n
        FROM strategy_research.etd_rolling_summary
        WHERE run_id = %s
        ORDER BY symbol, config
    """, (str(run_id),))
    rows = cur.fetchall()
    print(f"\n{'symbol':<12} {'config':<20} {'folds':>5} {'mean_S':>8} {'med_S':>8} {'std_S':>7} {'pct+':>6} {'wr':>6} {'cum':>7} {'IS/OOS':>7} {'n':>5}")
    print('-' * 94)
    for r in rows:
        sym, cfg, nf, ms, meds, stds, pp, wr, cum, corr, tn = r
        corr_s = f'{float(corr):>+.2f}' if corr is not None else '   — '
        print(f"{sym:<12} {cfg:<20} {nf:>5} {float(ms):>+8.3f} {float(meds):>+8.3f} {float(stds):>7.3f} {float(pp):>6.1%} {float(wr):>6.1%} {float(cum):>+7.3f} {corr_s:>7} {tn:>5}")

    for sym in SYMBOLS:
        print(f'\n--- {sym}: per test year ---')
        cur.execute("""
            SELECT config, test_year,
                   train_n, ROUND(train_sharpe::numeric,3)  AS train_S,
                   test_n,  ROUND(test_sharpe::numeric,3)   AS test_S,
                   ROUND(test_win_rate::numeric,3)           AS wr,
                   ROUND(test_cum_ret::numeric,3)            AS cum,
                   ROUND(boot_sharpe_p05::numeric,3)         AS bp05,
                   ROUND(boot_sharpe_p95::numeric,3)         AS bp95
            FROM strategy_research.etd_rolling_folds
            WHERE run_id = %s AND symbol = %s
            ORDER BY config, test_year
        """, (str(run_id), sym))
        rows = cur.fetchall()

        cur_cfg = None
        for r in rows:
            cfg, yr, tn, ts, yn, ys, wr, cum, bp05, bp95 = r
            if cfg != cur_cfg:
                cur_cfg = cfg
                print(f'\n  [{cfg}]')
                print(f"  {'year':>5} {'train_n':>7} {'train_S':>8} {'test_n':>6} {'test_S':>8} {'wr':>6} {'cum':>7} {'[p05':>6} {'p95]':>6}")
                print('  ' + '-' * 66)
            ts_s   = f'{float(ts):>+8.3f}' if ts is not None else '      —  '
            ys_s   = f'{float(ys):>+8.3f}' if ys is not None else '      —  '
            wr_s   = f'{float(wr):>6.1%}' if wr is not None else '   — '
            cum_s  = f'{float(cum):>+7.3f}' if cum is not None else '     — '
            bp05_s = f'{float(bp05):>+6.3f}' if bp05 is not None else '  — '
            bp95_s = f'{float(bp95):>+6.3f}' if bp95 is not None else '  — '
            print(f"  {yr:>5} {tn if tn else 0:>7} {ts_s} {yn if yn else 0:>6} {ys_s} {wr_s} {cum_s} {bp05_s} {bp95_s}")

    # raw vs ATR filter comparison
    print('\n--- raw_etd vs atr_filtered_etd: OOS Sharpe per fold ---')
    for sym in SYMBOLS:
        print(f'\n  {sym}')
        cur.execute("""
            SELECT test_year,
                   MAX(CASE WHEN config='raw_etd'         THEN ROUND(test_sharpe::numeric,3) END) AS raw_S,
                   MAX(CASE WHEN config='atr_filtered_etd' THEN ROUND(test_sharpe::numeric,3) END) AS filt_S,
                   MAX(CASE WHEN config='raw_etd'         THEN test_n END)                         AS raw_n,
                   MAX(CASE WHEN config='atr_filtered_etd' THEN test_n END)                        AS filt_n
            FROM strategy_research.etd_rolling_folds
            WHERE run_id = %s AND symbol = %s
            GROUP BY test_year ORDER BY test_year
        """, (str(run_id), sym))
        rows = cur.fetchall()
        print(f"  {'year':>5} {'raw_S':>8} {'raw_n':>6} {'filt_S':>8} {'filt_n':>7} {'delta':>7}")
        print('  ' + '-' * 46)
        for yr, rs, fs, rn, fn in rows:
            rs_s = f'{float(rs):>+8.3f}' if rs is not None else '      —  '
            fs_s = f'{float(fs):>+8.3f}' if fs is not None else '      —  '
            delta = f'{(float(fs)-float(rs)):>+7.3f}' if rs is not None and fs is not None else '      —  '
            print(f"  {yr:>5} {rs_s} {rn if rn else 0:>6} {fs_s} {fn if fn else 0:>7} {delta}")

    conn.close()

if __name__ == '__main__':
    main()

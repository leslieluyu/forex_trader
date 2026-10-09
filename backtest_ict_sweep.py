"""
Systematic approximation of the "顺势+截取+结构" (trend + liquidity-sweep +
structure) gold intraday method from the Bilibili video
(BV1iEMS6XEgv, UP "币圈避難针", claims 500U -> 11000U).

The method as shown is discretionary ICT-style trading:
  1. 顺势 (trend): only trade with the higher-timeframe trend.
  2. 截取 (liquidity sweep): on 15m/1h, a candle's wick pierces a recent
     swing high/low (sweeping resting stops) then closes back inside ->
     "smart money" grabbed liquidity, expect a reversal in trend direction.
  3. 结构 (structure / FVG): zoom to 1m, wait for a structure break (break
     of a recent minor swing point) confirming the reversal, enter there.
     Stop goes just beyond the sweep's extreme (video used ~5 points).

This script formalizes a testable version of that logic:
  - 15m swing low/high = rolling min/max of prior K bars (excluded current).
  - Bullish sweep: 15m low < prior swing low AND 15m close > prior swing low
    (wick rejection). Mirror for bearish sweep (swing high).
  - D1 close vs D1 SMA50 = trend filter (same convention used throughout
    this session's other backtests, for direct comparability).
  - Within a timeout window after the sweep bar, look at 1m bars for a
    structure break: 1m close > sweep bar's high (bull) / < sweep bar's low
    (bear). Entry at the NEXT 1m bar's open (no lookahead).
  - Stop: sweep bar's extreme (low for longs/high for shorts) minus/plus a
    buffer. Exit: 2x ATR(1m) trailing stop + Donchian(1m,20) close-exit,
    the same validated exit structure used for every other strategy this
    session, so results are apples-to-apples comparable.
  - Cost: $0.30 flat round-trip (same convention as research/). Real
    overnight swap applied if a trade crosses >=1 night (swap_long=-522.2
    points, swap_short=0.0 points, confirmed live on this Exness account).

Known simplification vs the video: HistData.com M1 data has volume=0
throughout (no real tick/volume data), so the "一定会有量" (must have
volume) confirmation in the video's sweep definition could NOT be tested
-- this version is wick-rejection price action only. FVG (fair value gap)
confirmation was also dropped in favor of the simpler structure-break
trigger, to keep the rule unambiguous and avoid hand-tuning an FVG
tolerance parameter after seeing results.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"
FLAT_COST = 0.30
SWAP_LONG_POINTS = -522.2
SWAP_SHORT_POINTS = 0.0
POINT_SIZE = 0.001  # XAUUSDm point size (from probe_swap.py)
CONTRACT_SIZE = 100.0  # 1 lot = 100 oz on this account; swap_long is $/lot/night already in price terms below

SWING_LOOKBACK_15M = 10          # bars to define the "recent swing" that gets swept
SWEEP_TIMEOUT_MIN = 60           # minutes allowed for the 1m structure break to occur
ATR_LEN = 20
ATR_MULT = 2.0
DONCH_EXIT = 20
STOP_BUFFER = 0.30               # small cushion beyond the sweep extreme (matches video's "~5 points" at low price levels, roughly proportionate at gold's $1000-4000 range tested here as an absolute $ buffer, not a %, to keep it simple/non-curve-fit)


def count_nights(entry, exitt):
    nights = 0
    d = entry.normalize() + pd.Timedelta(days=1)
    while d <= exitt:
        mult = 3 if d.dayofweek == 2 else 1
        nights += mult
        d += pd.Timedelta(days=1)
    return nights


def build_data():
    m1 = pd.read_parquet(DATA_PATH).sort_values("time").reset_index(drop=True)

    d1 = m1.set_index("time").resample("1D").agg(close=("close", "last")).dropna()
    d1["sma50"] = d1["close"].rolling(50).mean()
    d1["up_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)
    d1["dn_ok"] = d1["close"].shift(1) < d1["sma50"].shift(1)
    up_by_date = d1["up_ok"].copy(); up_by_date.index = up_by_date.index.date
    dn_by_date = d1["dn_ok"].copy(); dn_by_date.index = dn_by_date.index.date

    bars15 = m1.set_index("time").resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")).dropna().reset_index()
    bars15["date"] = bars15["time"].dt.date
    bars15["up_ok"] = bars15["date"].map(up_by_date).fillna(False)
    bars15["dn_ok"] = bars15["date"].map(dn_by_date).fillna(False)

    sl = bars15["low"].rolling(SWING_LOOKBACK_15M).min().shift(1)
    sh = bars15["high"].rolling(SWING_LOOKBACK_15M).max().shift(1)
    bars15["prior_swing_low"] = sl
    bars15["prior_swing_high"] = sh
    bars15["bull_sweep"] = (bars15["low"] < sl) & (bars15["close"] > sl) & bars15["up_ok"]
    bars15["bear_sweep"] = (bars15["high"] > sh) & (bars15["close"] < sh) & bars15["dn_ok"]

    return m1, bars15


def run():
    m1, bars15 = build_data()
    m1_time = m1["time"].to_numpy()
    m1_o = m1["open"].to_numpy(); m1_h = m1["high"].to_numpy()
    m1_l = m1["low"].to_numpy(); m1_c = m1["close"].to_numpy()
    n1 = len(m1)

    prev_close = pd.Series(m1_c).shift(1).to_numpy()
    tr = np.maximum.reduce([
        m1_h - m1_l,
        np.abs(m1_h - np.nan_to_num(prev_close, nan=m1_h[0])),
        np.abs(m1_l - np.nan_to_num(prev_close, nan=m1_l[0])),
    ])
    atr1 = pd.Series(tr).rolling(ATR_LEN, min_periods=ATR_LEN).mean().to_numpy()
    patr1 = pd.Series(atr1).shift(1).to_numpy()
    donch_low_exit = pd.Series(m1_l).rolling(DONCH_EXIT).min().shift(1).to_numpy()
    donch_high_exit = pd.Series(m1_h).rolling(DONCH_EXIT).max().shift(1).to_numpy()

    m1_idx_of_time = pd.Series(np.arange(n1), index=m1_time)

    sweeps = bars15[bars15["bull_sweep"] | bars15["bear_sweep"]].copy()
    print(f"15m bars total: {len(bars15)}  bull sweeps: {bars15['bull_sweep'].sum()}  bear sweeps: {bars15['bear_sweep'].sum()}")

    trades = []
    last_exit_i = -1
    warmup1 = max(ATR_LEN, DONCH_EXIT) + 5

    for _, row in sweeps.iterrows():
        is_bull = bool(row["bull_sweep"])
        sweep_close_time = row["time"] + pd.Timedelta(minutes=15)  # 15m bar close time
        pos = m1_idx_of_time.index.searchsorted(sweep_close_time)
        if pos >= n1 or pos <= last_exit_i or pos < warmup1:
            continue
        timeout_pos = min(n1 - 1, pos + SWEEP_TIMEOUT_MIN)
        sweep_high, sweep_low = float(row["high"]), float(row["low"])

        entry_i = None
        direction = None
        for j in range(pos, timeout_pos):
            if is_bull and m1_c[j] > sweep_high:
                entry_i = j + 1
                direction = 1
                break
            if (not is_bull) and m1_c[j] < sweep_low:
                entry_i = j + 1
                direction = -1
                break
        if entry_i is None or entry_i <= last_exit_i or entry_i >= n1 - 1:
            continue
        if np.isnan(patr1[entry_i]) or patr1[entry_i] <= 0:
            continue

        entry_price = float(m1_o[entry_i])
        entry_time = pd.Timestamp(m1_time[entry_i])
        if direction == 1:
            stop = sweep_low - STOP_BUFFER
            risk = entry_price - stop
        else:
            stop = sweep_high + STOP_BUFFER
            risk = stop - entry_price
        if risk <= 1e-6:
            continue

        cur_stop = stop
        exit_i, exit_price = None, None
        for j in range(entry_i, n1):
            if j > entry_i and not np.isnan(atr1[j - 1]) and atr1[j - 1] > 0:
                if direction == 1:
                    cur_stop = max(cur_stop, m1_c[j - 1] - ATR_MULT * atr1[j - 1])
                else:
                    cur_stop = min(cur_stop, m1_c[j - 1] + ATR_MULT * atr1[j - 1])
            if direction == 1:
                if m1_l[j] <= cur_stop:
                    exit_i, exit_price = j, float(cur_stop); break
                if j > entry_i and not np.isnan(donch_low_exit[j]) and m1_c[j] < donch_low_exit[j]:
                    exit_i, exit_price = j, float(m1_c[j]); break
            else:
                if m1_h[j] >= cur_stop:
                    exit_i, exit_price = j, float(cur_stop); break
                if j > entry_i and not np.isnan(donch_high_exit[j]) and m1_c[j] > donch_high_exit[j]:
                    exit_i, exit_price = j, float(m1_c[j]); break
        if exit_i is None:
            exit_i, exit_price = n1 - 1, float(m1_c[n1 - 1])

        exit_time = pd.Timestamp(m1_time[exit_i])
        raw_pnl = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        pnl = raw_pnl - FLAT_COST
        trades.append((pd.Timestamp(m1_time[exit_i]).date(), entry_time, exit_time,
                        direction, entry_price, risk, pnl))
        last_exit_i = exit_i

    t = pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "dir", "entry_price", "risk", "pnl"])
    t["nights"] = t.apply(lambda r: count_nights(r["entry_time"], r["exit_time"]), axis=1)
    swap_price_per_night = {1: SWAP_LONG_POINTS * POINT_SIZE, -1: SWAP_SHORT_POINTS * POINT_SIZE}
    t["pnl_no_swap"] = t["pnl"]
    t["pnl"] = t.apply(lambda r: r["pnl_no_swap"] + r["nights"] * swap_price_per_night[r["dir"]], axis=1)

    es_no_swap = equity_stats(t.assign(pnl=t["pnl_no_swap"]))
    es_swap = equity_stats(t)
    wins = t[t["pnl"] > 0]["pnl"]; losses = t[t["pnl"] <= 0]["pnl"]
    payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    print(f"\n=== ICT sweep+structure (long+short, D1 SMA50 trend filter) ===")
    print(f"n={len(t)}  long={len(t[t['dir']==1])}  short={len(t[t['dir']==-1])}  win={(t['pnl']>0).mean():.1%}  payoff={payoff:.2f}  pf={pf:.2f}")
    print(f"  no swap:   ret%={es_no_swap['total_return_pct']:8.1f}  dd%={es_no_swap['max_dd_pct']:7.1f}  Sharpe={es_no_swap['sharpe']:.2f}  CAGR%={(es_no_swap['cagr']*100 if es_no_swap['cagr']==es_no_swap['cagr'] else float('nan')):.2f}")
    print(f"  w/ swap:   ret%={es_swap['total_return_pct']:8.1f}  dd%={es_swap['max_dd_pct']:7.1f}  Sharpe={es_swap['sharpe']:.2f}  CAGR%={(es_swap['cagr']*100 if es_swap['cagr']==es_swap['cagr'] else float('nan')):.2f}")
    print(f"  % trades crossing >=1 overnight: {(t['nights']>0).mean():.1%}")
    print(f"  median hold (min): {((t['exit_time']-t['entry_time']).dt.total_seconds()/60).median():.1f}")

    print("\n=== Walk-forward (fixed params, 1y rolling OOS) ===")
    t_sorted = t.sort_values("entry_time").reset_index(drop=True)
    start = pd.Timestamp(t_sorted["entry_time"].min())
    end = pd.Timestamp(t_sorted["entry_time"].max())
    cur = start + pd.DateOffset(years=3)
    pos_f, neg_f = 0, 0
    while cur < end:
        window_end = cur + pd.DateOffset(years=1)
        mask = (t_sorted["entry_time"] >= cur) & (t_sorted["entry_time"] < window_end)
        g = t_sorted[mask]
        if len(g) > 0:
            es_f = equity_stats(g)
            pos_f += es_f["total_return_pct"] > 0
            neg_f += es_f["total_return_pct"] <= 0
            print(f"{cur.date()} -> {window_end.date()}: n={len(g):3d}  ret%={es_f['total_return_pct']:7.1f}  max_dd%={es_f['max_dd_pct']:6.1f}")
        cur = window_end
    print(f"positive folds: {pos_f}  negative folds: {neg_f}")

    return t


if __name__ == "__main__":
    run()

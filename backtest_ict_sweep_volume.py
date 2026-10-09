"""
Follow-up to backtest_ict_sweep.py, triggered by comparing our result against
a different author's EA backtest of a similarly-named ICT strategy (bilibili
BV1bR8C6JEzW, "编程小船长") which showed much better numbers (67.8% win rate,
PF 1.26) than our 25.5%/PF 0.54. One concrete, verifiable architectural
difference: that EA runs inside MT5's own strategy tester, which has access
to real tick_volume -- letting it actually implement the video's "一定会有
量" (must have volume) sweep-confirmation condition. Our original test used
HistData.com OHLC data where volume=0 throughout, so that confirmation was
necessarily dropped and disclosed as a simplification.

This script re-runs the same sweep+structure logic, but on this Exness
account's own MT5-exported XAUUSDm M1 data (data/XAUUSDm_M1.parquet), which
DOES carry real tick_volume -- letting us test with vs without a volume
filter directly, to see whether volume confirmation is in fact the source of
the gap.

HARD LIMITATION (disclosed upfront, not after seeing results): this broker's
MT5 terminal only retains M1 history back to ~2026-07-01 (probed via
probe_m1_depth.py) -- about 3 months, vs the 16+ years used in every other
backtest in this project. Results here are a small, recent-window diagnostic
on whether volume filtering changes anything directionally -- NOT a
statistically powered replacement for the original walk-forward conclusion
Trend filter (D1 SMA50) still uses the long D1 history (data/XAUUSDm_D1.parquet,
2014-2026) since D1 bars are cheap and that file has full depth.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

M1_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSDm_M1.parquet"
D1_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSDm_D1.parquet"
FLAT_COST = 0.30
SWAP_LONG_POINTS = -522.2
SWAP_SHORT_POINTS = 0.0
POINT_SIZE = 0.001

SWING_LOOKBACK_15M = 10
SWEEP_TIMEOUT_MIN = 60
ATR_LEN = 20
ATR_MULT = 2.0
DONCH_EXIT = 20
STOP_BUFFER = 0.30
VOL_LOOKBACK_15M = 20   # rolling average window for the volume filter


def count_nights(entry, exitt):
    nights = 0
    d = entry.normalize() + pd.Timedelta(days=1)
    while d <= exitt:
        mult = 3 if d.dayofweek == 2 else 1
        nights += mult
        d += pd.Timedelta(days=1)
    return nights


def build_data():
    m1 = pd.read_parquet(M1_PATH).sort_values("time").reset_index(drop=True)
    d1 = pd.read_parquet(D1_PATH).sort_values("time").set_index("time")
    d1["sma50"] = d1["close"].rolling(50).mean()
    d1["up_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)
    d1["dn_ok"] = d1["close"].shift(1) < d1["sma50"].shift(1)
    up_by_date = d1["up_ok"].copy(); up_by_date.index = up_by_date.index.date
    dn_by_date = d1["dn_ok"].copy(); dn_by_date.index = dn_by_date.index.date

    bars15 = m1.set_index("time").resample("15min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("tick_volume", "sum")).dropna().reset_index()
    bars15["date"] = bars15["time"].dt.date
    bars15["up_ok"] = bars15["date"].map(up_by_date).fillna(False)
    bars15["dn_ok"] = bars15["date"].map(dn_by_date).fillna(False)

    sl = bars15["low"].rolling(SWING_LOOKBACK_15M).min().shift(1)
    sh = bars15["high"].rolling(SWING_LOOKBACK_15M).max().shift(1)
    vol_avg = bars15["volume"].rolling(VOL_LOOKBACK_15M).mean().shift(1)
    bars15["has_volume"] = bars15["volume"] > vol_avg

    bars15["bull_sweep"] = (bars15["low"] < sl) & (bars15["close"] > sl) & bars15["up_ok"]
    bars15["bear_sweep"] = (bars15["high"] > sh) & (bars15["close"] < sh) & bars15["dn_ok"]

    return m1, bars15


def run(require_volume):
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
    donch_low_exit = pd.Series(m1_l).rolling(DONCH_EXIT).min().shift(1).to_numpy()
    donch_high_exit = pd.Series(m1_h).rolling(DONCH_EXIT).max().shift(1).to_numpy()
    m1_idx_of_time = pd.Series(np.arange(n1), index=m1_time)

    mask = bars15["bull_sweep"] | bars15["bear_sweep"]
    if require_volume:
        mask = mask & bars15["has_volume"]
    sweeps = bars15[mask].copy()

    trades = []
    last_exit_i = -1
    warmup1 = max(ATR_LEN, DONCH_EXIT) + 5

    for _, row in sweeps.iterrows():
        is_bull = bool(row["bull_sweep"])
        sweep_close_time = row["time"] + pd.Timedelta(minutes=15)
        pos = m1_idx_of_time.index.searchsorted(sweep_close_time)
        if pos >= n1 or pos <= last_exit_i or pos < warmup1:
            continue
        timeout_pos = min(n1 - 1, pos + SWEEP_TIMEOUT_MIN)
        sweep_high, sweep_low = float(row["high"]), float(row["low"])

        entry_i = None
        direction = None
        for j in range(pos, timeout_pos):
            if is_bull and m1_c[j] > sweep_high:
                entry_i, direction = j + 1, 1; break
            if (not is_bull) and m1_c[j] < sweep_low:
                entry_i, direction = j + 1, -1; break
        if entry_i is None or entry_i <= last_exit_i or entry_i >= n1 - 1:
            continue
        if np.isnan(atr1[entry_i]) or atr1[entry_i] <= 0:
            continue

        entry_price = float(m1_o[entry_i])
        entry_time = pd.Timestamp(m1_time[entry_i])
        if direction == 1:
            stop = sweep_low - STOP_BUFFER; risk = entry_price - stop
        else:
            stop = sweep_high + STOP_BUFFER; risk = stop - entry_price
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
        trades.append((entry_time, exit_time, direction, entry_price, risk, pnl))
        last_exit_i = exit_i

    t = pd.DataFrame(trades, columns=["entry_time", "exit_time", "dir", "entry_price", "risk", "pnl"])
    if t.empty:
        print("  no trades")
        return t
    t["nights"] = t.apply(lambda r: count_nights(r["entry_time"], r["exit_time"]), axis=1)
    swap_price_per_night = {1: SWAP_LONG_POINTS * POINT_SIZE, -1: SWAP_SHORT_POINTS * POINT_SIZE}
    t["pnl"] = t.apply(lambda r: r["pnl"] + r["nights"] * swap_price_per_night[r["dir"]], axis=1)

    es = equity_stats(t)
    wins = t[t["pnl"] > 0]["pnl"]; losses = t[t["pnl"] <= 0]["pnl"]
    payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    print(f"  n={len(t)}  win={(t['pnl']>0).mean():.1%}  payoff={payoff:.2f}  pf={pf:.2f}  "
          f"ret%={es['total_return_pct']:.1f}  Sharpe={es['sharpe']:.2f}")
    return t


if __name__ == "__main__":
    m1 = pd.read_parquet(M1_PATH)
    print(f"MT5 real-tick_volume M1 data window: {m1['time'].min()} -> {m1['time'].max()} "
          f"({(m1['time'].max()-m1['time'].min()).days} days) -- small-sample diagnostic only\n")

    print("WITHOUT volume filter (same rule as backtest_ict_sweep.py, this short window):")
    run(require_volume=False)
    print("\nWITH volume filter (sweep bar's 15m volume > prior 20-bar rolling average):")
    run(require_volume=True)

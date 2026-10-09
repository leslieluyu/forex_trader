"""
Sweep the validated Donchian(entry)+SMA50(D1 filter)+ATR-trailing-stop
structure across several entry timeframes to see if H1 is actually the best
choice or just the one that happened to get tested first.

Keeps the exact same rule structure as backtest_donchian_sma_atr.py /
research/backtest_h1_sma50_wf.py, parameterized by bar frequency.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

FLAT_COST = 0.30  # matches research/backtest_cross_source.py: $0.30 flat round-trip, not proportional spread
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").set_index("time")

d1 = m1.resample("1D").agg(close=("close", "last")).dropna()
d1["sma50"] = d1["close"].rolling(50).mean()
d1["trend_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)
trend_ok_by_date = d1["trend_ok"].copy()
trend_ok_by_date.index = trend_ok_by_date.index.date


def build_bars(freq):
    b = m1.resample(freq).agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last")).dropna()
    b = b.reset_index()
    b["date"] = b["time"].dt.date
    b["trend_ok"] = b["date"].map(trend_ok_by_date).fillna(False)
    return b


def backtest(bars, donch_entry=20, donch_exit=10, atr_len=20, atr_mult=2.0, cost=True):
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    trend_ok = bars["trend_ok"].to_numpy()
    flat_cost = FLAT_COST if cost else 0.0
    times = bars["time"].to_numpy()
    dates = bars["date"].to_numpy()
    n = len(bars)

    prev_close = pd.Series(c).shift(1).to_numpy()
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - np.nan_to_num(prev_close, nan=h[0])),
        np.abs(l - np.nan_to_num(prev_close, nan=l[0])),
    ])
    # match research/backtest_h1_sma50_wf.py exactly: simple rolling-mean ATR
    # (not EWM), unshifted for trailing-stop use, separately shifted once
    # ("patr") for the initial-stop reference at the signal bar.
    atrN = pd.Series(tr).rolling(atr_len, min_periods=atr_len).mean().to_numpy()
    patr = pd.Series(atrN).shift(1).to_numpy()
    donch_high = pd.Series(h).rolling(donch_entry).max().shift(1).to_numpy()
    low_exit = pd.Series(l).rolling(donch_exit).min().shift(1).to_numpy()

    warmup = max(donch_entry, atr_len, donch_exit) + 5
    trades = []
    i = warmup
    while i < n - 1:
        if np.isnan(donch_high[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > donch_high[i]):
            i += 1
            continue
        if not trend_ok[i]:
            i += 1
            continue
        entry_i = i + 1
        entry_price = float(o[entry_i])
        entry_time = times[entry_i]
        stop = entry_price - atr_mult * float(patr[i])
        risk = entry_price - stop
        if risk <= 0.01:
            i += 1
            continue
        exit_i, exit_price = None, None
        cur_stop = stop
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrN[j - 1]) and atrN[j - 1] > 0:
                cur_stop = max(cur_stop, c[j - 1] - atr_mult * atrN[j - 1])
            if l[j] <= cur_stop:
                exit_i, exit_price = j, float(cur_stop)
                break
            if j > entry_i and not np.isnan(low_exit[j]) and c[j] < low_exit[j]:
                exit_i, exit_price = j, float(c[j])
                break
        if exit_i is None:
            exit_i, exit_price = n - 1, float(c[n - 1])
        pnl = (exit_price - entry_price) - flat_cost
        trades.append((dates[exit_i], entry_time, times[exit_i], entry_price, risk, pnl))
        i = exit_i + 1
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


def naive_sharpe(trades):
    if trades.empty:
        return np.nan
    daily = trades.groupby("date")["pnl"].sum()
    return daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan


timeframes = [("M15", "15min"), ("M30", "30min"), ("H1", "1h"), ("H4", "4h"), ("D1", "1D")]

print(f"{'TF':<5} {'n':>6} {'trades/yr':>10} {'win%':>7} {'naive_Sh':>9} "
      f"{'compound_Sh':>12} {'total_ret%':>12} {'CAGR%':>8} {'max_dd%':>9}")
for label, freq in timeframes:
    bars = build_bars(freq)
    t = backtest(bars)
    if t.empty:
        print(f"{label:<5}  no trades")
        continue
    years = (bars["time"].max() - bars["time"].min()).days / 365.25
    es = equity_stats(t)
    win = (t["pnl"] > 0).mean()
    print(f"{label:<5} {len(t):>6} {len(t)/years:>10.1f} {win:>6.1%} {naive_sharpe(t):>9.2f} "
          f"{es['sharpe']:>12.2f} {es['total_return_pct']:>12.1f} "
          f"{(es['cagr']*100 if es['cagr']==es['cagr'] else float('nan')):>8.1f} {es['max_dd_pct']:>9.1f}")

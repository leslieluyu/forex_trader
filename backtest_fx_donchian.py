"""
Same validated structure (H1 Donchian(20) breakout + D1 SMA50 filter + 2x ATR
trail + Donchian(20) close-exit, long-only) applied to other forex majors,
to see if the edge found (then mostly erased by real costs) on gold shows up
anywhere else -- this time with REAL swap cost baked in from the start,
not discovered after the fact.

Cost model: $ spread cost approximated via a flat round-trip in price terms
per symbol (typical raw ECN-ish spread cost scaled to pip value), PLUS real
overnight swap_long (confirmed via live MT5 symbol_info on this Exness
account) applied per night held (Wed = 3x).

Swap rates (points) and point size/contract from probe_fx_swaps.py:
  EURUSD: swap_long=-5.60, point=0.00001, contract=100000 -> $/lot/night = swap*point*contract
  GBPUSD: swap_long=-2.20, point=0.00001, contract=100000
  AUDUSD: swap_long=0.00,  point=0.00001, contract=100000  (long is FREE here)
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

FLAT_COST_PRICE = {  # approx round-trip spread cost in price terms (not pips)
    "EURUSD": 0.00015,   # ~1.5 pip round trip
    "GBPUSD": 0.00020,   # ~2.0 pip round trip
    "AUDUSD": 0.00020,   # ~2.0 pip round trip
}
SWAP_LONG_POINTS = {"EURUSD": -5.60, "GBPUSD": -2.20, "AUDUSD": 0.00}
POINT_SIZE = {"EURUSD": 0.00001, "GBPUSD": 0.00001, "AUDUSD": 0.00001}
CONTRACT_SIZE = 100000.0

ATR_LEN = 20
ATR_MULT = 2.0
DONCH_ENTRY = 20
DONCH_EXIT = 20


def count_nights(entry, exitt):
    nights = 0
    d = entry.normalize() + pd.Timedelta(days=1)
    while d <= exitt:
        mult = 3 if d.dayofweek == 2 else 1
        nights += mult
        d += pd.Timedelta(days=1)
    return nights


def run(symbol):
    data_path = f"/Volumes/My Passport/working_data/forex_trader/data/{symbol}_histdata_M1.parquet"
    m1 = pd.read_parquet(data_path).sort_values("time").set_index("time")

    d1 = m1.resample("1D").agg(close=("close", "last")).dropna()
    d1["sma50"] = d1["close"].rolling(50).mean()
    d1["trend_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)
    trend_ok_by_date = d1["trend_ok"].copy()
    trend_ok_by_date.index = trend_ok_by_date.index.date

    bars = m1.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last")).dropna()
    bars = bars.reset_index()
    bars["date"] = bars["time"].dt.date
    bars["trend_ok"] = bars["date"].map(trend_ok_by_date).fillna(False)

    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    trend_ok = bars["trend_ok"].to_numpy()
    times = bars["time"].to_numpy()
    dates = bars["date"].to_numpy()
    n = len(bars)

    prev_close = pd.Series(c).shift(1).to_numpy()
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - np.nan_to_num(prev_close, nan=h[0])),
        np.abs(l - np.nan_to_num(prev_close, nan=l[0])),
    ])
    atrN = pd.Series(tr).rolling(ATR_LEN, min_periods=ATR_LEN).mean().to_numpy()
    patr = pd.Series(atrN).shift(1).to_numpy()
    donch_high = pd.Series(h).rolling(DONCH_ENTRY).max().shift(1).to_numpy()
    low_exit = pd.Series(l).rolling(DONCH_EXIT).min().shift(1).to_numpy()

    warmup = max(DONCH_ENTRY, ATR_LEN, DONCH_EXIT) + 5
    flat_cost = FLAT_COST_PRICE[symbol]
    trades = []
    i = warmup
    while i < n - 1:
        if np.isnan(donch_high[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > donch_high[i]) or not trend_ok[i]:
            i += 1
            continue
        entry_i = i + 1
        entry_price = float(o[entry_i])
        entry_time = times[entry_i]
        stop = entry_price - ATR_MULT * float(patr[i])
        risk = entry_price - stop
        if risk <= 1e-7:
            i += 1
            continue
        cur_stop = stop
        exit_i, exit_price = None, None
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrN[j - 1]) and atrN[j - 1] > 0:
                cur_stop = max(cur_stop, c[j - 1] - ATR_MULT * atrN[j - 1])
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

    t = pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["nights"] = t.apply(lambda r: count_nights(r["entry_time"], r["exit_time"]), axis=1)

    swap_per_unit_per_night = SWAP_LONG_POINTS[symbol] * POINT_SIZE[symbol]  # price terms per 1 unit (not per lot)
    t["pnl_no_swap"] = t["pnl"]
    t["pnl"] = t["pnl"] + t["nights"] * swap_per_unit_per_night  # already negative or zero

    es_no_swap = equity_stats(t.assign(pnl=t["pnl_no_swap"]))
    es_swap = equity_stats(t)
    wins = t[t["pnl"] > 0]["pnl"]
    losses = t[t["pnl"] <= 0]["pnl"]
    payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    print(f"=== {symbol} H1 Donchian20+SMA50+2xATR trail+Donch20 exit (long-only) ===")
    print(f"n={len(t)}  win={(t['pnl']>0).mean():.1%}  payoff={payoff:.2f}  pf={pf:.2f}")
    print(f"  no swap:   ret%={es_no_swap['total_return_pct']:8.1f}  dd%={es_no_swap['max_dd_pct']:7.1f}  Sharpe={es_no_swap['sharpe']:.2f}  CAGR%={es_no_swap['cagr']*100:.2f}")
    print(f"  w/ swap:   ret%={es_swap['total_return_pct']:8.1f}  dd%={es_swap['max_dd_pct']:7.1f}  Sharpe={es_swap['sharpe']:.2f}  CAGR%={(es_swap['cagr']*100 if es_swap['cagr']==es_swap['cagr'] else float('nan')):.2f}")
    print(f"  % trades crossing >=1 overnight: {(t['nights']>0).mean():.1%}")
    print()
    return t


if __name__ == "__main__":
    for sym in ["EURUSD", "GBPUSD", "AUDUSD"]:
        run(sym)

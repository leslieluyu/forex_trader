"""
Short-side mirror of the validated H1 Donchian(20)+SMA50(D1)+ATR(2x)-trail
structure. Motivation: on this Exness XAUUSDm account, swap_short = 0.0
(confirmed live via MT5 symbol_info) while swap_long = -522.2 points
(~-$52.22/lot/night) -- shorts carry zero overnight funding cost, which
removes the long side's core tension between "let the trend run" and
"avoid overnight swap."

Entry: H1 close < 20-bar low (Donchian breakdown), AND prior D1 close <
D1 SMA50 (downtrend filter) -- exact mirror of the long rule.
Exit: 2x ATR trailing stop (trails down as price falls) OR H1 close >
20-bar high (Donchian exit), whichever first. Donch exit window matches
the long side's validated choice (20, not the original 10).
Cost: $0.30 flat round-trip (matches research methodology). No swap cost
modeled (real swap_short = 0 on this account) -- this is the whole point
of testing the short side.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

FLAT_COST = 0.30
ATR_LEN = 20
ATR_MULT = 2.0
DONCH_ENTRY = 20
DONCH_EXIT = 20
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").set_index("time")

d1 = m1.resample("1D").agg(close=("close", "last")).dropna()
d1["sma50"] = d1["close"].rolling(50).mean()
d1["downtrend_ok"] = d1["close"].shift(1) < d1["sma50"].shift(1)
downtrend_by_date = d1["downtrend_ok"].copy()
downtrend_by_date.index = downtrend_by_date.index.date

bars = m1.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last")).dropna()
bars = bars.reset_index()
bars["date"] = bars["time"].dt.date
bars["downtrend_ok"] = bars["date"].map(downtrend_by_date).fillna(False)

o = bars["open"].to_numpy()
h = bars["high"].to_numpy()
l = bars["low"].to_numpy()
c = bars["close"].to_numpy()
downtrend_ok = bars["downtrend_ok"].to_numpy()
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
donch_low = pd.Series(l).rolling(DONCH_ENTRY).min().shift(1).to_numpy()
high_exit = pd.Series(h).rolling(DONCH_EXIT).max().shift(1).to_numpy()

warmup = max(DONCH_ENTRY, ATR_LEN, DONCH_EXIT) + 5
signal_idxs = []
for i in range(warmup, n - 1):
    if np.isnan(donch_low[i]) or np.isnan(patr[i]) or patr[i] <= 0:
        continue
    if not (c[i] < donch_low[i]) or not downtrend_ok[i]:
        continue
    signal_idxs.append(i)


def backtest_short():
    trades = []
    last_exit_i = -1
    for sig_i in signal_idxs:
        if sig_i <= last_exit_i:
            continue
        entry_i = sig_i + 1
        if entry_i >= n:
            continue
        entry_price = float(o[entry_i])
        entry_time = times[entry_i]
        stop = entry_price + ATR_MULT * float(patr[sig_i])
        risk = stop - entry_price
        if risk <= 0.01:
            continue
        cur_stop = stop
        exit_i, exit_price = None, None
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrN[j - 1]) and atrN[j - 1] > 0:
                cur_stop = min(cur_stop, c[j - 1] + ATR_MULT * atrN[j - 1])
            if h[j] >= cur_stop:
                exit_i, exit_price = j, float(cur_stop)
                break
            if j > entry_i and not np.isnan(high_exit[j]) and c[j] > high_exit[j]:
                exit_i, exit_price = j, float(c[j])
                break
        if exit_i is None:
            exit_i, exit_price = n - 1, float(c[n - 1])
        pnl = (entry_price - exit_price) - FLAT_COST  # short: profit when price falls
        trades.append((dates[exit_i], entry_time, times[exit_i], entry_price, risk, pnl))
        last_exit_i = exit_i
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


def naive_sharpe(trades):
    if trades.empty:
        return np.nan
    daily = trades.groupby("date")["pnl"].sum()
    return daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan


t = backtest_short()
es = equity_stats(t)
wins = t[t["pnl"] > 0]["pnl"]
losses = t[t["pnl"] <= 0]["pnl"]
payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

print("H1 Donchian(20) breakdown + D1 SMA50 downtrend filter + 2xATR trail + Donch20 exit (SHORT):")
print(f"n={len(t)}  win={(t['pnl']>0).mean():.1%}  payoff={payoff:.2f}  pf={pf:.2f}  "
      f"naive_Sharpe={naive_sharpe(t):.2f}  compound_Sharpe={es['sharpe']:.2f}  "
      f"total_ret%={es['total_return_pct']:.1f}  CAGR%={(es['cagr']*100 if es['cagr']==es['cagr'] else float('nan')):.2f}  "
      f"max_dd%={es['max_dd_pct']:.1f}")

print("\n=== Year-by-year ===")
t["year"] = t["date"].apply(lambda d: d.year)
for year, g in t.groupby("year"):
    es_y = equity_stats(g)
    print(f"{year}: n={len(g):4d}  win={(g['pnl']>0).mean():.1%}  "
          f"sum_pnl={g['pnl'].sum():8.2f}  naive_Sh={naive_sharpe(g):+.2f}")

print("\n=== Walk-forward (fixed params, 1y rolling OOS) ===")
t_sorted = t.sort_values("entry_time").reset_index(drop=True)
start = pd.Timestamp(t_sorted["entry_time"].min())
end = pd.Timestamp(t_sorted["entry_time"].max())
cur = start + pd.DateOffset(years=3)
pos, neg = 0, 0
while cur < end:
    window_end = cur + pd.DateOffset(years=1)
    mask = (t_sorted["entry_time"] >= cur) & (t_sorted["entry_time"] < window_end)
    g = t_sorted[mask]
    if len(g) > 0:
        es_f = equity_stats(g)
        pos += es_f["total_return_pct"] > 0
        neg += es_f["total_return_pct"] <= 0
        print(f"{cur.date()} -> {window_end.date()}: n={len(g):3d}  "
              f"ret%={es_f['total_return_pct']:7.1f}  max_dd%={es_f['max_dd_pct']:6.1f}")
    cur = window_end
print(f"positive folds: {pos}  negative folds: {neg}")

t["entry_time"] = pd.to_datetime(t["entry_time"])
t["exit_time"] = pd.to_datetime(t["exit_time"])
t["hold_hours"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 3600
print(f"\nmedian hold hours: {t['hold_hours'].median():.1f}  "
      f"% held >24h (no swap cost since short): {(t['hold_hours']>24).mean():.1%}")

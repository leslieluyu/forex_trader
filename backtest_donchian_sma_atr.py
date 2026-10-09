"""
Independent replication of the pasted "H1 Donchian breakout + D1 SMA50
filter + ATR trailing stop, long-only" strategy, claimed +227% total /
-26% max DD / 37% win / 1.41 payoff over 2010-2026 on XAUUSD.

Rules (as specified):
  - D1 filter: PRIOR completed day's close > 50-day SMA (no lookahead).
  - H1 entry: close breaks above the high of the past 20 H1 bars (excluding
    current bar).
  - Entry: next H1 bar's open.
  - Initial stop: entry - 2*ATR(20), ATR computed on completed bars only.
  - Trailing stop: each bar, new_stop = max(old_stop, prev_close - 2*ATR20);
    ratchets up only.
  - Exit: stop hit, OR close < low of past 10 H1 bars (excluding current).
    Same-bar conflict -> stop takes priority.
  - Long only.

Cost: using our established proportional spread model (5.85bps round trip,
from live Exness demo quotes) for consistency with every other script in
this project, rather than the pasted strategy's flat $0.30/oz assumption --
a flat dollar spread implies a shrinking % cost as gold's price rose from
~1200 (2010) to ~4100 (2026), which would make the early, cheaper-gold years
look artificially better than they were.
"""
import numpy as np
import pandas as pd

SPREAD_BPS = 5.85e-4
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").set_index("time")

d1 = m1.resample("1D").agg(close=("close", "last")).dropna()
d1["sma50"] = d1["close"].rolling(50).mean()
d1["trend_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)  # prior day's close vs prior SMA
trend_ok_by_date = d1["trend_ok"].copy()
trend_ok_by_date.index = trend_ok_by_date.index.date

h1 = m1.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last")).dropna()
h1 = h1.reset_index()
h1["date"] = h1["time"].dt.date
h1["spread_px"] = h1["close"] * SPREAD_BPS
h1["trend_ok"] = h1["date"].map(trend_ok_by_date).fillna(False)

print(f"H1 bars: {len(h1)}, {h1['time'].min()} -> {h1['time'].max()}")

o = h1["open"].to_numpy()
h = h1["high"].to_numpy()
l = h1["low"].to_numpy()
c = h1["close"].to_numpy()
trend_ok = h1["trend_ok"].to_numpy()
spread = h1["spread_px"].to_numpy()
times = h1["time"].to_numpy()
dates = h1["date"].to_numpy()
n = len(h1)

prev_close = pd.Series(c).shift(1).to_numpy()
tr = np.maximum.reduce([
    h - l,
    np.abs(h - np.nan_to_num(prev_close, nan=h[0])),
    np.abs(l - np.nan_to_num(prev_close, nan=l[0])),
])
atr20 = pd.Series(tr).ewm(alpha=1 / 20, adjust=False).mean().shift(1).to_numpy()  # completed-data ATR

donchian_high20 = pd.Series(h).rolling(20).max().shift(1).to_numpy()
low10 = pd.Series(l).rolling(10).min().shift(1).to_numpy()


def backtest():
    trades = []
    position = 0
    entry_price = entry_time = stop = None
    for i in range(51, n - 1):
        if position == 1:
            # update trailing stop using PRIOR bar's close before checking this bar
            candidate_stop = c[i - 1] - 2 * atr20[i - 1] if not np.isnan(atr20[i - 1]) else stop
            stop = max(stop, candidate_stop)
            hit_stop = l[i] <= stop
            hit_low10 = c[i] < low10[i]
            if hit_stop or hit_low10:
                exit_price = stop if hit_stop else c[i]
                raw_pnl = exit_price - entry_price
                r = entry_price - trades_init_stop
                pnl_r = (raw_pnl - spread[i]) / r
                trades.append((dates[i], entry_time, times[i], pnl_r))
                position = 0
            continue

        if not trend_ok[i]:
            continue
        if np.isnan(donchian_high20[i]) or np.isnan(atr20[i]):
            continue
        if c[i] > donchian_high20[i]:
            entry_cand = o[i + 1]
            init_stop = entry_cand - 2 * atr20[i]
            r = entry_cand - init_stop
            if r > 0:
                position = 1
                entry_price, entry_time = entry_cand, times[i + 1]
                stop = init_stop
                trades_init_stop = init_stop

    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "pnl_r"])


trades = backtest()


def stats(trades):
    if trades.empty:
        return dict(n=0, win_rate=np.nan, total_r=0.0, sharpe=np.nan, max_dd=0.0)
    daily = trades.groupby("date")["pnl_r"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    cum = trades["pnl_r"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    avg_win = trades.loc[trades["pnl_r"] > 0, "pnl_r"].mean()
    avg_loss = trades.loc[trades["pnl_r"] <= 0, "pnl_r"].mean()
    payoff = abs(avg_win / avg_loss) if avg_loss else np.nan
    return dict(n=len(trades), win_rate=(trades["pnl_r"] > 0).mean(),
                total_r=trades["pnl_r"].sum(), sharpe=sharpe, max_dd=max_dd, payoff=payoff)


s = stats(trades)
print(f"\n=== Combined: n={s['n']}  win={s['win_rate']:.1%}  payoff={s['payoff']:.2f}  "
      f"total_R={s['total_r']:.1f}  sharpe={s['sharpe']:+.2f}  max_dd_R={s['max_dd']:.1f}")

print("\n=== Year-by-year ===")
trades["year"] = trades["date"].apply(lambda d: d.year)
for year, g in trades.groupby("year"):
    ss = stats(g)
    print(f"{year}: n={ss['n']:4d}  win={ss['win_rate']:.1%}  "
          f"total_R={ss['total_r']:7.1f}  sharpe={ss['sharpe']:+.2f}  dd_R={ss['max_dd']:7.1f}")

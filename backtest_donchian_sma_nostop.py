"""
Test the claim (from a secondhand, unverified source -- treat as a hypothesis
to check, not a fact): removing the ATR *trailing* stop (but keeping the
initial fixed ATR stop in place, un-trailed) should reduce sensitivity to
data-source noise on extreme-volatility days, since the exit no longer
depends on continuously-updated intrabar high/low prints.

NOTE: corrected to match research/backtest_cross_source.py's actual
"use_atr_trail=False" definition -- this is NOT a no-stop-loss strategy.
The initial stop (entry - 2*ATR) stays fixed for the life of the trade;
exit is whichever comes first: price trades through that fixed stop, or
close breaks the Donchian-10 exit channel. An earlier version of this
script had NO stop at all, which is a materially different (and much
riskier) strategy -- that was a bug, not a faithful replication.

Same entry rule as backtest_donchian_sma_atr.py (H1 Donchian-20 breakout +
D1 SMA50 filter). Cost model matches research exactly: flat $0.30 round-trip
cost (not a proportional spread), full compounding equity simulation +
walk-forward.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats, sim_equity

FLAT_COST = 0.30
ATR_N = 20
ATR_MULT = 2.0
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


def atr_series(h, l, c):
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(ATR_N, min_periods=ATR_N).mean().to_numpy()


def backtest_no_atr_trail(bars, donch_entry=20, donch_exit=10):
    """Fixed initial ATR stop (not trailed) + Donchian-10 close-exit, whichever hits first."""
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    trend_ok = bars["trend_ok"].to_numpy()
    times = bars["time"].to_numpy()
    dates = bars["date"].to_numpy()
    n = len(bars)

    donch_high = pd.Series(h).rolling(donch_entry).max().shift(1).to_numpy()
    low_exit = pd.Series(l).rolling(donch_exit).min().shift(1).to_numpy()
    atrv = atr_series(h, l, c)
    patr = np.r_[np.nan, atrv[:-1]]

    warmup = max(donch_entry, donch_exit, ATR_N) + 5
    trades = []
    i = warmup
    while i < n - 1:
        if np.isnan(donch_high[i]) or np.isnan(low_exit[i]) or np.isnan(patr[i]) or patr[i] <= 0:
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
        if risk <= 0.01:
            i += 1
            continue
        exit_i, exit_price = None, None
        for j in range(entry_i, n):
            if l[j] <= stop:
                exit_i, exit_price = j, float(stop)
                break
            if j > entry_i and not np.isnan(low_exit[j]) and c[j] < low_exit[j]:
                exit_i, exit_price = j, float(c[j])
                break
        if exit_i is None:
            exit_i, exit_price = n - 1, float(c[n - 1])
        pnl = (exit_price - entry_price) - FLAT_COST
        trades.append((dates[exit_i], entry_time, times[exit_i], entry_price, risk, pnl))
        i = exit_i + 1
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


def naive_sharpe(trades):
    if trades.empty:
        return np.nan
    daily = trades.groupby("date")["pnl"].sum()
    return daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan


bars = build_bars("1h")
t = backtest_no_atr_trail(bars)
es = equity_stats(t)
print(f"H1 Donchian+SMA50, fixed ATR stop (not trailed) + donch-10 exit:")
print(f"n={len(t)}  win={ (t['pnl']>0).mean():.1%}  naive_Sharpe={naive_sharpe(t):.2f}  "
      f"compound_Sharpe={es['sharpe']:.2f}  total_ret%={es['total_return_pct']:.1f}  "
      f"max_dd%={es['max_dd_pct']:.1f}")

print("\n=== Year-by-year ===")
t["year"] = t["date"].apply(lambda d: d.year)
for year, g in t.groupby("year"):
    es_y = equity_stats(g)
    print(f"{year}: n={len(g):4d}  win={(g['pnl']>0).mean():.1%}  "
          f"sum_pnl={g['pnl'].sum():8.2f}  naive_Sh={naive_sharpe(g):+.2f}")

print("\n=== Walk-forward (3y train window unused -- fixed params, 1y rolling OOS test) ===")
t_sorted = t.sort_values("entry_time").reset_index(drop=True)
start = pd.Timestamp(t_sorted["entry_time"].min())
end = pd.Timestamp(t_sorted["entry_time"].max())
cur = start + pd.DateOffset(years=3)
while cur < end:
    window_end = cur + pd.DateOffset(years=1)
    mask = (t_sorted["entry_time"] >= cur) & (t_sorted["entry_time"] < window_end)
    g = t_sorted[mask]
    if len(g) > 0:
        es_f = equity_stats(g)
        print(f"{cur.date()} -> {window_end.date()}: n={len(g):3d}  "
              f"ret%={es_f['total_return_pct']:7.1f}  max_dd%={es_f['max_dd_pct']:6.1f}")
    cur = window_end

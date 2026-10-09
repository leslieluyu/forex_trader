"""
Long-history (2010-2026) robustness check for the Donchian trend-breakout
candidate that showed weak positive edge (Sharpe ~0.43) on the 3.5-month
XAUUSDm screening run.

Same data/cost assumptions as backtest_orb_robustness.py.
"""
import numpy as np
import pandas as pd

SPREAD_BPS = 5.85e-4
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

df = pd.read_parquet(DATA_PATH).sort_values("time").reset_index(drop=True)
df["spread_px"] = df["close"] * SPREAD_BPS
df["date"] = df["time"].dt.date

print(f"Loaded {len(df)} M1 bars, {df['date'].nunique()} trading days, "
      f"{df['time'].min()} -> {df['time'].max()}")


def backtest_donchian(df, lookback=60, max_hold_minutes=240):
    hh = df["high"].rolling(lookback).max().shift(1).to_numpy()
    ll = df["low"].rolling(lookback).min().shift(1).to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    spread = df["spread_px"].to_numpy()
    dates = df["date"].to_numpy()

    trades = []
    position = 0
    entry_price = entry_time = None
    n = len(df)
    for i in range(n):
        if np.isnan(hh[i]):
            continue
        if position == 0:
            if close[i] > hh[i]:
                position, entry_price, entry_time = 1, close[i], times[i]
            elif close[i] < ll[i]:
                position, entry_price, entry_time = -1, close[i], times[i]
        else:
            held_min = (times[i] - entry_time) / np.timedelta64(1, "m")
            reverse = (position == 1 and close[i] < ll[i]) or \
                      (position == -1 and close[i] > hh[i])
            timed_out = held_min >= max_hold_minutes
            if reverse or timed_out:
                raw_pnl = (close[i] - entry_price) * position
                pnl = raw_pnl - spread[i]
                trades.append((dates[i], entry_time, times[i], position, pnl))
                if reverse:
                    position = -position
                    entry_price, entry_time = close[i], times[i]
                else:
                    position = 0
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "side", "pnl"])


def stats(trades):
    if trades.empty:
        return dict(n=0, win_rate=np.nan, total_pnl=0.0, sharpe=np.nan, max_dd=0.0)
    daily = trades.groupby("date")["pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    cum = trades["pnl"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return dict(
        n=len(trades),
        win_rate=(trades["pnl"] > 0).mean(),
        total_pnl=trades["pnl"].sum(),
        sharpe=sharpe,
        max_dd=max_dd,
    )


print("\n=== Year-by-year, fixed params (60min lookback/240min max hold) ===")
all_trades = backtest_donchian(df)
for year, g in all_trades.groupby(all_trades["date"].apply(lambda d: d.year)):
    s = stats(g)
    print(f"{year}: n={s['n']:4d}  win={s['win_rate']:.1%}  "
          f"pnl={s['total_pnl']:8.1f}  sharpe={s['sharpe']:+.2f}  dd={s['max_dd']:8.1f}")

overall = stats(all_trades)
print(f"\nFULL 2010-2026: n={overall['n']}  win={overall['win_rate']:.1%}  "
      f"pnl={overall['total_pnl']:.1f}  sharpe={overall['sharpe']:+.2f}  "
      f"dd={overall['max_dd']:.1f}")

print("\n=== Parameter sensitivity grid (full 2010-2026 history) ===")
print(f"{'lookback':>10} {'max_hold':>10} {'n':>6} {'win%':>7} {'sharpe':>8} {'pnl':>10}")
for lookback in [20, 40, 60, 120, 240]:
    for max_hold in [120, 240, 480, 960]:
        t = backtest_donchian(df, lookback=lookback, max_hold_minutes=max_hold)
        s = stats(t)
        print(f"{lookback:>10} {max_hold:>10} {s['n']:>6} {s['win_rate']:>6.1%} "
              f"{s['sharpe']:>+8.2f} {s['total_pnl']:>10.1f}")

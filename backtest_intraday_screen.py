"""
Screening backtest for a few classic intraday strategy ideas on XAUUSDm M1 bars.

Data: ~3.5 months of M1 (2026-06-29 -> now), from Exness demo account via MT5.
This is a SCREENING run only (short sample, single demo-account spread regime,
no walk-forward) -- meant to rule out obviously-dead ideas before investing more
time, not to certify anything as deployable.

Strategies tested:
  1. ORB  - opening range breakout (first N min of each UTC day)
  2. DONC - Donchian/ATR trend breakout
  3. MR   - mean reversion fade (rolling z-score)

Costs: actual recorded spread (points) from the data, charged as a full
round-trip cost on every trade (conservative: no slippage beyond spread,
no swap/overnight financing modeled).
"""
import numpy as np
import pandas as pd

POINT = 0.001
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSDm_M1.parquet"

df = pd.read_parquet(DATA_PATH).sort_values("time").reset_index(drop=True)
df["spread_px"] = df["spread"] * POINT
df["date"] = df["time"].dt.date

print(f"Loaded {len(df)} M1 bars, {df['date'].nunique()} trading days, "
      f"{df['time'].min()} -> {df['time'].max()}")
print(f"Median spread: {df['spread_px'].median():.3f}  "
      f"Mean spread: {df['spread_px'].mean():.3f}")


def summarize(trades: pd.DataFrame, label: str):
    if trades.empty:
        print(f"[{label}] no trades")
        return
    n = len(trades)
    win_rate = (trades["pnl"] > 0).mean()
    total_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()
    # daily return series for Sharpe-like stat
    daily = trades.groupby(trades["exit_time"].dt.date)["pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    cum = trades["pnl"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    print(f"[{label}] trades={n}  win_rate={win_rate:.1%}  "
          f"total_pnl={total_pnl:.2f}  avg_pnl={avg_pnl:.4f}  "
          f"sharpe~{sharpe:.2f}  max_dd={max_dd:.2f}")


# ---------------------------------------------------------------------------
# 1. Opening Range Breakout
# ---------------------------------------------------------------------------
def backtest_orb(df, range_minutes=30, max_hold_minutes=240, stop_mult=1.0):
    trades = []
    for date, g in df.groupby("date"):
        g = g.reset_index(drop=True)
        if len(g) < range_minutes + 10:
            continue
        opening = g.iloc[:range_minutes]
        or_high, or_low = opening["high"].max(), opening["low"].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue
        rest = g.iloc[range_minutes:]
        position = 0
        entry_price = entry_time = stop = None
        for _, row in rest.iterrows():
            if position == 0:
                if row["close"] > or_high:
                    position, entry_price, entry_time = 1, row["close"], row["time"]
                    stop = entry_price - stop_mult * or_range
                elif row["close"] < or_low:
                    position, entry_price, entry_time = -1, row["close"], row["time"]
                    stop = entry_price + stop_mult * or_range
            else:
                held_min = (row["time"] - entry_time).total_seconds() / 60
                hit_stop = (position == 1 and row["low"] <= stop) or \
                           (position == -1 and row["high"] >= stop)
                timed_out = held_min >= max_hold_minutes
                if hit_stop or timed_out:
                    exit_price = stop if hit_stop else row["close"]
                    raw_pnl = (exit_price - entry_price) * position
                    pnl = raw_pnl - row["spread_px"]
                    trades.append(dict(date=date, entry_time=entry_time,
                                        exit_time=row["time"], side=position,
                                        pnl=pnl))
                    position = 0
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# 2. Donchian channel trend breakout
# ---------------------------------------------------------------------------
def backtest_donchian(df, lookback=60, max_hold_minutes=240):
    d = df.copy()
    d["hh"] = d["high"].rolling(lookback).max().shift(1)
    d["ll"] = d["low"].rolling(lookback).min().shift(1)
    trades = []
    position = 0
    entry_price = entry_time = None
    for _, row in d.iterrows():
        if pd.isna(row["hh"]):
            continue
        if position == 0:
            if row["close"] > row["hh"]:
                position, entry_price, entry_time = 1, row["close"], row["time"]
            elif row["close"] < row["ll"]:
                position, entry_price, entry_time = -1, row["close"], row["time"]
        else:
            held_min = (row["time"] - entry_time).total_seconds() / 60
            reverse = (position == 1 and row["close"] < row["ll"]) or \
                      (position == -1 and row["close"] > row["hh"])
            timed_out = held_min >= max_hold_minutes
            if reverse or timed_out:
                raw_pnl = (row["close"] - entry_price) * position
                pnl = raw_pnl - row["spread_px"]
                trades.append(dict(entry_time=entry_time, exit_time=row["time"],
                                    side=position, pnl=pnl))
                if reverse:
                    position = -position
                    entry_price, entry_time = row["close"], row["time"]
                else:
                    position = 0
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# 3. Mean reversion fade
# ---------------------------------------------------------------------------
def backtest_mean_reversion(df, window=60, z_entry=2.0, max_hold_minutes=60):
    d = df.copy()
    d["ma"] = d["close"].rolling(window).mean()
    d["sd"] = d["close"].rolling(window).std()
    d["z"] = (d["close"] - d["ma"]) / d["sd"]
    trades = []
    position = 0
    entry_price = entry_time = None
    for _, row in d.iterrows():
        if pd.isna(row["z"]):
            continue
        if position == 0:
            if row["z"] > z_entry:
                position, entry_price, entry_time = -1, row["close"], row["time"]
            elif row["z"] < -z_entry:
                position, entry_price, entry_time = 1, row["close"], row["time"]
        else:
            held_min = (row["time"] - entry_time).total_seconds() / 60
            back_to_mean = (position == 1 and row["close"] >= row["ma"]) or \
                           (position == -1 and row["close"] <= row["ma"])
            timed_out = held_min >= max_hold_minutes
            if back_to_mean or timed_out:
                raw_pnl = (row["close"] - entry_price) * position
                pnl = raw_pnl - row["spread_px"]
                trades.append(dict(entry_time=entry_time, exit_time=row["time"],
                                    side=position, pnl=pnl))
                position = 0
    return pd.DataFrame(trades)


if __name__ == "__main__":
    orb = backtest_orb(df)
    summarize(orb, "ORB (30min range, 1x stop, 4h max hold)")

    donc = backtest_donchian(df)
    summarize(donc, "Donchian (60min lookback, 4h max hold)")

    mr = backtest_mean_reversion(df)
    summarize(mr, "MeanReversion (60min z-score, 1h max hold)")

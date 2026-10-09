"""
Fade failed breakout: mirror of the Donchian breakout logic, but instead of
riding the breakout, wait a short confirmation window to see if it fails
(price closes back inside the range) and fade it in the opposite direction.

Motivated by: (1) Al Brooks' stated heuristic that ~80% of range-breakout
attempts fail, and (2) our own 2010-2026 finding that naive breakout-and-hold
(ORB, Donchian) is systematically unprofitable on gold -- consistent with
breakouts failing more often than they continue.

Rules:
  - Donchian channel (lookback bars) defines the range (hh/ll, computed on
    bars prior to the current one).
  - A breakout is "pending" when close crosses outside the channel.
  - If, within confirm_bars, close crosses back inside the original breakout
    level, the breakout is "failed" -> enter fade (opposite direction) at
    that close.
  - Stop: the extreme (highest high / lowest low) reached during the failed
    breakout attempt.
  - Exit: stop hit, price reaches the channel midline, or max_hold_minutes
    timeout.
  - If the breakout does NOT fail within confirm_bars, no trade (we are not
    testing breakout-continuation here, already shown dead).

Same cost model as the other long-history scripts: proportional spread
~5.85bps round trip, approximated from live Exness demo quotes.
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


def backtest_fade_failed_breakout(df, lookback=60, confirm_bars=15, max_hold_minutes=240):
    hh = df["high"].rolling(lookback).max().shift(1).to_numpy()
    ll = df["low"].rolling(lookback).min().shift(1).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    spread = df["spread_px"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)

    trades = []
    # pending breakout watch state
    watch_dir = 0       # +1 = watching an upside breakout, -1 = downside, 0 = none
    watch_level = None  # the channel level that was broken
    watch_start = None  # index when breakout first observed
    watch_extreme = None

    # open fade position state
    position = 0
    entry_price = entry_time = stop = mid = None

    for i in range(n):
        if np.isnan(hh[i]):
            continue

        # --- manage open fade position first ---
        if position != 0:
            held_min = (times[i] - entry_time) / np.timedelta64(1, "m")
            hit_stop = (position == 1 and low[i] <= stop) or \
                       (position == -1 and high[i] >= stop)
            hit_mid = (position == 1 and high[i] >= mid) or \
                      (position == -1 and low[i] <= mid)
            timed_out = held_min >= max_hold_minutes
            if hit_stop or hit_mid or timed_out:
                exit_price = stop if hit_stop else (mid if hit_mid else close[i])
                raw_pnl = (exit_price - entry_price) * position
                pnl = raw_pnl - spread[i]
                trades.append((dates[i], entry_time, times[i], position, pnl))
                position = 0
            continue  # don't evaluate new breakouts while in a trade

        # --- manage pending breakout watch ---
        if watch_dir == 0:
            if close[i] > hh[i]:
                watch_dir, watch_level, watch_start, watch_extreme = 1, hh[i], i, high[i]
            elif close[i] < ll[i]:
                watch_dir, watch_level, watch_start, watch_extreme = -1, ll[i], i, low[i]
        else:
            bars_since = i - watch_start
            if watch_dir == 1:
                watch_extreme = max(watch_extreme, high[i])
            else:
                watch_extreme = min(watch_extreme, low[i])

            failed = (watch_dir == 1 and close[i] < watch_level) or \
                     (watch_dir == -1 and close[i] > watch_level)
            if failed:
                position = -watch_dir
                entry_price, entry_time = close[i], times[i]
                stop = watch_extreme
                mid = (hh[i] + ll[i]) / 2
                watch_dir = 0
            elif bars_since >= confirm_bars:
                watch_dir = 0  # give up watching, breakout continued or stalled

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


print("\n=== Year-by-year, fixed params (60min lookback/15bar confirm/240min max hold) ===")
all_trades = backtest_fade_failed_breakout(df)
for year, g in all_trades.groupby(all_trades["date"].apply(lambda d: d.year)):
    s = stats(g)
    print(f"{year}: n={s['n']:4d}  win={s['win_rate']:.1%}  "
          f"pnl={s['total_pnl']:8.1f}  sharpe={s['sharpe']:+.2f}  dd={s['max_dd']:8.1f}")

overall = stats(all_trades)
print(f"\nFULL 2010-2026: n={overall['n']}  win={overall['win_rate']:.1%}  "
      f"pnl={overall['total_pnl']:.1f}  sharpe={overall['sharpe']:+.2f}  "
      f"dd={overall['max_dd']:.1f}")

print("\n=== Parameter sensitivity grid (full 2010-2026 history) ===")
print(f"{'lookback':>10} {'confirm':>8} {'max_hold':>10} {'n':>6} {'win%':>7} {'sharpe':>8} {'pnl':>10}")
for lookback in [20, 40, 60, 120]:
    for confirm_bars in [5, 15, 30]:
        for max_hold in [120, 240, 480]:
            t = backtest_fade_failed_breakout(df, lookback=lookback,
                                               confirm_bars=confirm_bars,
                                               max_hold_minutes=max_hold)
            s = stats(t)
            print(f"{lookback:>10} {confirm_bars:>8} {max_hold:>10} {s['n']:>6} "
                  f"{s['win_rate']:>6.1%} {s['sharpe']:>+8.2f} {s['total_pnl']:>10.1f}")

"""
Same regime-filtered test as backtest_regime_filtered.py, but classify
trend/range regime from WEEKLY scale (prior completed week's ADX), not daily.
Every day within a given week inherits that week's classification -- this
matches "I read the weekly trend before deciding how to trade this week"
rather than an intraday or lagged-daily judgment.
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

# ---------------------------------------------------------------------------
# Weekly bars + ADX(14), classify each week using the PRIOR completed week
# ---------------------------------------------------------------------------
d1 = df.groupby("date").agg(open=("open", "first"), high=("high", "max"),
                             low=("low", "min"), close=("close", "last"))
d1.index = pd.to_datetime(d1.index)
d1 = d1.sort_index()

wk = d1.resample("W-FRI").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last")).dropna()

high, low, close = wk["high"], wk["low"], wk["close"]
prev_close = close.shift(1)
plus_dm = (high - high.shift(1)).clip(lower=0)
minus_dm = (low.shift(1) - low).clip(lower=0)
plus_dm[(plus_dm <= minus_dm)] = 0
minus_dm[(minus_dm <= plus_dm)] = 0
tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

period = 14
atr = tr.ewm(alpha=1 / period, adjust=False).mean()
plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
adx_w = dx.ewm(alpha=1 / period, adjust=False).mean()

# shift by 1 week: this week's regime uses LAST week's closed ADX
prior_adx_w = adx_w.shift(1)
TREND_TH, RANGE_TH = 25, 20
trend_weeks = prior_adx_w[prior_adx_w > TREND_TH].index  # W-FRI timestamps
range_weeks = prior_adx_w[prior_adx_w < RANGE_TH].index

# map every calendar day to the week-ending-Friday label, then expand to date sets
day_week = pd.Series(d1.index, index=d1.index).apply(
    lambda d: d + pd.offsets.Week(weekday=4) if d.weekday() != 4 else d
)
trend_days = set(day_week[day_week.isin(trend_weeks)].index.date)
range_days = set(day_week[day_week.isin(range_weeks)].index.date)
print(f"\nWeeks classified: trend={len(trend_weeks)}  range={len(range_weeks)}  "
      f"total_weeks={len(wk)}")
print(f"Days covered: trend={len(trend_days)}  range={len(range_days)}  "
      f"total_days={len(d1)}")


def backtest_donchian_filtered(df, allowed_dates, lookback=60, max_hold_minutes=240):
    hh = df["high"].rolling(lookback).max().shift(1).to_numpy()
    ll = df["low"].rolling(lookback).min().shift(1).to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    spread = df["spread_px"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)

    trades = []
    position = 0
    entry_price = entry_time = None
    for i in range(n):
        if np.isnan(hh[i]):
            continue
        if position == 0:
            if dates[i] not in allowed_dates:
                continue
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
                if reverse and dates[i] in allowed_dates:
                    position = -position
                    entry_price, entry_time = close[i], times[i]
                else:
                    position = 0
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "side", "pnl"])


def backtest_fade_filtered(df, allowed_dates, lookback=60, confirm_bars=15, max_hold_minutes=240):
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
    watch_dir = 0
    watch_level = None
    watch_start = None
    watch_extreme = None
    position = 0
    entry_price = entry_time = stop = mid = None

    for i in range(n):
        if np.isnan(hh[i]):
            continue
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
            continue

        if dates[i] not in allowed_dates:
            watch_dir = 0
            continue

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
                watch_dir = 0

    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "side", "pnl"])


def stats(trades):
    if trades.empty:
        return dict(n=0, win_rate=np.nan, total_pnl=0.0, sharpe=np.nan, max_dd=0.0)
    daily = trades.groupby("date")["pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    cum = trades["pnl"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return dict(n=len(trades), win_rate=(trades["pnl"] > 0).mean(),
                total_pnl=trades["pnl"].sum(), sharpe=sharpe, max_dd=max_dd)


print("\n=== Donchian breakout, TREND WEEKS ONLY (weekly ADX>25, prior week) ===")
t1 = backtest_donchian_filtered(df, trend_days)
s1 = stats(t1)
print(f"n={s1['n']}  win={s1['win_rate']:.1%}  pnl={s1['total_pnl']:.1f}  "
      f"sharpe={s1['sharpe']:+.2f}  dd={s1['max_dd']:.1f}")

print("\n=== Fade failed breakout, RANGE WEEKS ONLY (weekly ADX<20, prior week) ===")
t2 = backtest_fade_filtered(df, range_days)
s2 = stats(t2)
print(f"n={s2['n']}  win={s2['win_rate']:.1%}  pnl={s2['total_pnl']:.1f}  "
      f"sharpe={s2['sharpe']:+.2f}  dd={s2['max_dd']:.1f}")

print("\n=== Combined ===")
combined = pd.concat([t1, t2]).sort_values("entry_time")
sc = stats(combined)
print(f"n={sc['n']}  win={sc['win_rate']:.1%}  pnl={sc['total_pnl']:.1f}  "
      f"sharpe={sc['sharpe']:+.2f}  dd={sc['max_dd']:.1f}")

print("\n=== Year-by-year combined ===")
combined["year"] = combined["date"].apply(lambda d: d.year)
for year, g in combined.groupby("year"):
    s = stats(g)
    print(f"{year}: n={s['n']:4d}  win={s['win_rate']:.1%}  "
          f"pnl={s['total_pnl']:8.1f}  sharpe={s['sharpe']:+.2f}  dd={s['max_dd']:8.1f}")

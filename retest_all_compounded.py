"""
Re-test every strategy from this conversation's research line using proper
compounding equity-curve simulation (1% equity risk/trade, 5x leverage cap),
matching the methodology that validated the Donchian+SMA50 winner -- instead
of the naive non-compounding R-sum Sharpe used in the original tests.

Goal: check whether any of the 7 previously-rejected candidates were false
negatives caused by the weaker Sharpe methodology, not by a real lack of edge.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

SPREAD_BPS = 5.85e-5
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").reset_index(drop=True)
m1["spread_px"] = m1["close"] * SPREAD_BPS
m1["date"] = m1["time"].dt.date

h1 = m1.set_index("time").resample("1h").agg(
    open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
).dropna().reset_index()
h1["spread_px"] = h1["close"] * SPREAD_BPS
h1["date"] = h1["time"].dt.date

print(f"M1 bars: {len(m1)}  H1 bars: {len(h1)}")


def naive_sharpe(trades, pnl_col="pnl"):
    if trades.empty:
        return np.nan
    daily = trades.groupby("date")[pnl_col].sum()
    return daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan


results = []


# ---------------------------------------------------------------------------
# 1. ORB (30min range, 1.0x stop, 240min max hold) -- M1
# ---------------------------------------------------------------------------
def backtest_orb(df, range_minutes=30, max_hold_minutes=240, stop_mult=1.0):
    trades = []
    for date, g in df.groupby("date", sort=False):
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
        for row in rest.itertuples():
            if position == 0:
                if row.close > or_high:
                    position, entry_price, entry_time = 1, row.close, row.time
                    stop = entry_price - stop_mult * or_range
                elif row.close < or_low:
                    position, entry_price, entry_time = -1, row.close, row.time
                    stop = entry_price + stop_mult * or_range
            else:
                held_min = (row.time - entry_time).total_seconds() / 60
                hit_stop = (position == 1 and row.low <= stop) or \
                           (position == -1 and row.high >= stop)
                timed_out = held_min >= max_hold_minutes
                if hit_stop or timed_out:
                    exit_price = stop if hit_stop else row.close
                    raw_pnl = (exit_price - entry_price) * position
                    pnl = raw_pnl - row.spread_px
                    risk = abs(entry_price - stop)
                    trades.append((date, entry_time, row.time, entry_price, risk, pnl))
                    position = 0
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


t = backtest_orb(m1)
results.append(("ORB (30/1.0x/240)", naive_sharpe(t), equity_stats(t)))
print("ORB done", len(t))


# ---------------------------------------------------------------------------
# 2. Donchian trend breakout (60min lookback, 240min max hold) -- M1
#    No native stop (stop-and-reverse system); risk proxied as distance
#    from entry to the opposite channel band at entry time.
# ---------------------------------------------------------------------------
def backtest_donchian(df, lookback=60, max_hold_minutes=240):
    hh = df["high"].rolling(lookback).max().shift(1).to_numpy()
    ll = df["low"].rolling(lookback).min().shift(1).to_numpy()
    close = df["close"].to_numpy()
    times = df["time"].to_numpy()
    spread = df["spread_px"].to_numpy()
    dates = df["date"].to_numpy()
    n = len(df)
    trades = []
    position = 0
    entry_price = entry_time = entry_risk = None
    for i in range(n):
        if np.isnan(hh[i]):
            continue
        if position == 0:
            if close[i] > hh[i]:
                position, entry_price, entry_time = 1, close[i], times[i]
                entry_risk = max(entry_price - ll[i], entry_price * 0.001)
            elif close[i] < ll[i]:
                position, entry_price, entry_time = -1, close[i], times[i]
                entry_risk = max(hh[i] - entry_price, entry_price * 0.001)
        else:
            held_min = (times[i] - entry_time) / np.timedelta64(1, "m")
            reverse = (position == 1 and close[i] < ll[i]) or \
                      (position == -1 and close[i] > hh[i])
            timed_out = held_min >= max_hold_minutes
            if reverse or timed_out:
                raw_pnl = (close[i] - entry_price) * position
                pnl = raw_pnl - spread[i]
                trades.append((dates[i], entry_time, times[i], entry_price, entry_risk, pnl))
                if reverse:
                    position = -position
                    entry_price, entry_time = close[i], times[i]
                    entry_risk = max(abs(entry_price - (ll[i] if position == 1 else hh[i])), entry_price * 0.001)
                else:
                    position = 0
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


t = backtest_donchian(m1)
results.append(("Donchian (60/240)", naive_sharpe(t), equity_stats(t)))
print("Donchian done", len(t))


# ---------------------------------------------------------------------------
# 3. Mean reversion fade (60min window, z=2.0, 60min max hold) -- M1
#    Never tested on long history before -- doing it now for completeness.
#    Risk proxied as |entry - stop-loss-equivalent| = entry's distance from
#    the z=3 extreme (a nominal risk unit), since the original had no hard stop.
# ---------------------------------------------------------------------------
def backtest_mean_reversion(df, window=60, z_entry=2.0, max_hold_minutes=60):
    d = df.copy()
    d["ma"] = d["close"].rolling(window).mean()
    d["sd"] = d["close"].rolling(window).std()
    d["z"] = (d["close"] - d["ma"]) / d["sd"]
    trades = []
    position = 0
    entry_price = entry_time = entry_risk = None
    for row in d.itertuples():
        if pd.isna(row.z):
            continue
        if position == 0:
            if row.z > z_entry:
                position, entry_price, entry_time = -1, row.close, row.time
                entry_risk = max(row.sd * 1.0, row.close * 0.001)
            elif row.z < -z_entry:
                position, entry_price, entry_time = 1, row.close, row.time
                entry_risk = max(row.sd * 1.0, row.close * 0.001)
        else:
            held_min = (row.time - entry_time).total_seconds() / 60
            back_to_mean = (position == 1 and row.close >= row.ma) or \
                           (position == -1 and row.close <= row.ma)
            timed_out = held_min >= max_hold_minutes
            if back_to_mean or timed_out:
                raw_pnl = (row.close - entry_price) * position
                pnl = raw_pnl - row.spread_px
                trades.append((row.date, entry_time, row.time, entry_price, entry_risk, pnl))
                position = 0
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


t = backtest_mean_reversion(m1)
results.append(("MeanReversion (60/z2/60) [first long-history run]", naive_sharpe(t), equity_stats(t)))
print("MeanReversion done", len(t))


# ---------------------------------------------------------------------------
# 4. Fade failed breakout (60min lookback, 15 confirm, 240 max hold) -- M1
# ---------------------------------------------------------------------------
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
    watch_dir, watch_level, watch_start, watch_extreme = 0, None, None, None
    position = 0
    entry_price = entry_time = stop = mid = None
    for i in range(n):
        if np.isnan(hh[i]):
            continue
        if position != 0:
            held_min = (times[i] - entry_time) / np.timedelta64(1, "m")
            hit_stop = (position == 1 and low[i] <= stop) or (position == -1 and high[i] >= stop)
            hit_mid = (position == 1 and high[i] >= mid) or (position == -1 and low[i] <= mid)
            timed_out = held_min >= max_hold_minutes
            if hit_stop or hit_mid or timed_out:
                exit_price = stop if hit_stop else (mid if hit_mid else close[i])
                raw_pnl = (exit_price - entry_price) * position
                pnl = raw_pnl - spread[i]
                risk = abs(entry_price - stop)
                trades.append((dates[i], entry_time, times[i], entry_price, risk, pnl))
                position = 0
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
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


t = backtest_fade_failed_breakout(m1)
results.append(("FadeFailedBreakout (60/15/240)", naive_sharpe(t), equity_stats(t)))
print("FadeFailedBreakout done", len(t))


# ---------------------------------------------------------------------------
# 5. Brooks H1/H2 -- H1 bars, already R-normalized; convert to price-based
#    pnl + risk=1 unit scaled by actual stop distance for the shared simulator.
# ---------------------------------------------------------------------------
h1["ema20"] = h1["close"].ewm(span=20, adjust=False).mean()
o, h, l, c = h1["open"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy(), h1["close"].to_numpy()
rng = h - l
body = np.abs(c - o)
ema20 = h1["ema20"].to_numpy()
spread_h1 = h1["spread_px"].to_numpy()
times_h1 = h1["time"].to_numpy()
n_h1 = len(h1)

with np.errstate(invalid="ignore", divide="ignore"):
    strong_bull = (c > o) & ((c - l) >= 0.5 * rng) & (body >= 0.5 * rng) & (rng > 0)
    strong_bear = (c < o) & ((h - c) >= 0.5 * rng) & (body >= 0.5 * rng) & (rng > 0)
    bear_bar = (c < o)
    doji = (rng > 0) & (body < 0.3 * rng)
    pullback_bar = bear_bar | doji
    pullback_bar_short = (c > o) | doji

five_bar_high = pd.Series(h).rolling(5).max().shift(1).to_numpy()
five_bar_low = pd.Series(l).rolling(5).min().shift(1).to_numpy()


def run_brooks_side(direction):
    trades = []
    position = 0
    entry_price = entry_time = stop = target = None
    has_push = False
    pullback_seen = False
    for i in range(6, n_h1 - 1):
        regime_ok = (c[i] > ema20[i]) if direction == 1 else (c[i] < ema20[i])
        if position != 0:
            if direction == 1:
                hit_stop, hit_tgt = l[i] <= stop, h[i] >= target
            else:
                hit_stop, hit_tgt = h[i] >= stop, l[i] <= target
            if hit_stop or hit_tgt:
                exit_price = stop if hit_stop else target
                raw_pnl = (exit_price - entry_price) * direction
                pnl = raw_pnl - spread_h1[i]
                risk = abs(entry_price - stop)
                trades.append((h1["time"].iloc[i].date(), entry_time, times_h1[i], entry_price, risk, pnl))
                position = 0
            continue
        if not regime_ok:
            has_push, pullback_seen = False, False
            continue
        if direction == 1:
            push_now = (h[i] > five_bar_high[i]) or strong_bull[i]
            if has_push and pullback_seen:
                if strong_bull[i]:
                    qualify = c[i] >= l[i] + (2 / 3) * rng[i] or c[i] > h[i - 1]
                    if qualify:
                        stop_cand, entry_cand = l[i], o[i + 1]
                        stop_dist = entry_cand - stop_cand
                        if stop_dist > 0 and stop_dist / entry_cand >= 0.0005:
                            position, entry_price, entry_time = 1, entry_cand, times_h1[i + 1]
                            stop, target = stop_cand, entry_price + 2 * stop_dist
                        has_push, pullback_seen = False, False
                    else:
                        has_push, pullback_seen = True, False
            elif has_push and not pullback_seen:
                if pullback_bar[i]:
                    pullback_seen = True
            else:
                if push_now:
                    has_push, pullback_seen = True, False
        else:
            push_now = (l[i] < five_bar_low[i]) or strong_bear[i]
            if has_push and pullback_seen:
                if strong_bear[i]:
                    qualify = c[i] <= h[i] - (2 / 3) * rng[i] or c[i] < l[i - 1]
                    if qualify:
                        stop_cand, entry_cand = h[i], o[i + 1]
                        stop_dist = stop_cand - entry_cand
                        if stop_dist > 0 and stop_dist / entry_cand >= 0.0005:
                            position, entry_price, entry_time = -1, entry_cand, times_h1[i + 1]
                            stop, target = stop_cand, entry_price - 2 * stop_dist
                        has_push, pullback_seen = False, False
                    else:
                        has_push, pullback_seen = True, False
            elif has_push and not pullback_seen:
                if pullback_bar_short[i]:
                    pullback_seen = True
            else:
                if push_now:
                    has_push, pullback_seen = True, False
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


t_long = run_brooks_side(1)
t_short = run_brooks_side(-1)
t = pd.concat([t_long, t_short]).sort_values("entry_time").reset_index(drop=True)
results.append(("Brooks H1/H2", naive_sharpe(t), equity_stats(t)))
print("Brooks H1/H2 done", len(t))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'strategy':<50} {'n':>6} {'naive_Sharpe':>13} {'compound_Sharpe':>16} "
      f"{'total_ret%':>12} {'max_dd%':>10}")
for name, naive_sh, es in results:
    print(f"{name:<50} {es['n']:>6} {naive_sh:>13.2f} {es['sharpe']:>16.2f} "
          f"{es['total_return_pct']:>12.1f} {es['max_dd_pct']:>10.1f}")

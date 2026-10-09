"""
Al Brooks H1/H2 strong-signal-bar strategy, on XAUUSD H1 bars (resampled from
HistData M1, 2010-2026).

Simplification vs the user's spec: H1 and H2 are economically the same
entry trigger (strong signal bar after push+pullback, in regime direction);
the spec's H1-fails-then-H2 narrative doesn't change entry/stop/target
mechanics, so this implementation takes the first qualifying signal bar
after any push+pullback cycle, rather than separately bookkeeping "H1 failed,
now watching for H2". Flagged explicitly -- this is a simplification, not a
literal transcription.

PnL is computed in R-multiples (R = initial stop distance) rather than raw
price points, since position sizing is 1% equity risk per trade -- this
normalizes across gold's price level changing ~1200 -> ~4100 over the period.

Rules:
  - EMA20 on H1 close defines regime: close>EMA20 -> long only, else short only.
  - Strong bull bar: close>open, close in upper half of range, body >= 0.5*range.
    Strong bear bar: mirror.
  - Push up: current bar makes a new 5-bar high, OR current bar is a strong bull bar.
  - Pullback: >=1 bear bar or doji after the push.
  - Signal bar: strong bull bar after pullback, with close in top third of its
    range OR close > previous bar's high.
  - Entry: next bar's open. Stop: signal bar's low (long) / high (short).
  - Target: 2R. Skip if stop distance < 0.05% of entry price.
  - Only one open position at a time; ignore new signals while in a trade.
"""
import numpy as np
import pandas as pd

SPREAD_BPS = 5.85e-4
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").set_index("time")
h1 = m1.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                            low=("low", "min"), close=("close", "last")).dropna()
h1 = h1.reset_index()
h1["spread_px"] = h1["close"] * SPREAD_BPS
h1["ema20"] = h1["close"].ewm(span=20, adjust=False).mean()

print(f"H1 bars: {len(h1)}, {h1['time'].min()} -> {h1['time'].max()}")

o, h, l, c = h1["open"].to_numpy(), h1["high"].to_numpy(), h1["low"].to_numpy(), h1["close"].to_numpy()
rng = h - l
body = np.abs(c - o)
ema20 = h1["ema20"].to_numpy()
spread = h1["spread_px"].to_numpy()
times = h1["time"].to_numpy()
n = len(h1)

with np.errstate(invalid="ignore", divide="ignore"):
    strong_bull = (c > o) & ((c - l) >= 0.5 * rng) & (body >= 0.5 * rng) & (rng > 0)
    strong_bear = (c < o) & ((h - c) >= 0.5 * rng) & (body >= 0.5 * rng) & (rng > 0)
    bear_bar = (c < o)
    doji = (rng > 0) & (body < 0.3 * rng)
    pullback_bar = bear_bar | doji
    pullback_bar_short = (c > o) | doji  # bull bar or doji = pullback for short side

five_bar_high = pd.Series(h).rolling(5).max().shift(1).to_numpy()
five_bar_low = pd.Series(l).rolling(5).min().shift(1).to_numpy()


def run_side(direction):
    """direction: +1 for long setups (H1/H2), -1 for short setups (L1/L2)."""
    trades = []
    position = 0
    entry_price = entry_time = stop = target = None
    has_push = False
    pullback_seen = False

    for i in range(6, n - 1):
        regime_ok = (c[i] > ema20[i]) if direction == 1 else (c[i] < ema20[i])

        if position != 0:
            # check this bar for stop/target (conservative: stop checked first)
            if direction == 1:
                hit_stop = l[i] <= stop
                hit_tgt = h[i] >= target
            else:
                hit_stop = h[i] >= stop
                hit_tgt = l[i] <= target
            if hit_stop or hit_tgt:
                exit_price = stop if hit_stop else target
                raw_pnl = (exit_price - entry_price) * direction
                r = abs(entry_price - stop)
                pnl_r = (raw_pnl - spread[i]) / r
                trades.append((h1["time"].iloc[i].date(), entry_time, times[i], direction, pnl_r))
                position = 0
            continue

        if not regime_ok:
            has_push, pullback_seen = False, False
            continue

        if direction == 1:
            push_now = (h[i] > five_bar_high[i]) or strong_bull[i]
            if has_push and pullback_seen:
                if strong_bull[i]:
                    top_third = c[i] >= l[i] + (2 / 3) * rng[i]
                    above_prior_high = c[i] > h[i - 1]
                    if top_third or above_prior_high:
                        stop_cand = l[i]
                        entry_cand = o[i + 1]
                        stop_dist = entry_cand - stop_cand
                        if stop_dist > 0 and stop_dist / entry_cand >= 0.0005:
                            position = 1
                            entry_price, entry_time = entry_cand, times[i + 1]
                            stop = stop_cand
                            target = entry_price + 2 * stop_dist
                        has_push, pullback_seen = False, False
                    else:
                        # strong bull bar but didn't qualify as a signal -- treat
                        # as a fresh push, keep watching for a new pullback
                        has_push, pullback_seen = True, False
                # else: still in pullback / neutral bar, keep waiting
            elif has_push and not pullback_seen:
                if pullback_bar[i]:
                    pullback_seen = True
                # a non-qualifying push extension just keeps has_push True
            else:
                if push_now:
                    has_push, pullback_seen = True, False
        else:
            push_now = (l[i] < five_bar_low[i]) or strong_bear[i]
            if has_push and pullback_seen:
                if strong_bear[i]:
                    bottom_third = c[i] <= h[i] - (2 / 3) * rng[i]
                    below_prior_low = c[i] < l[i - 1]
                    if bottom_third or below_prior_low:
                        stop_cand = h[i]
                        entry_cand = o[i + 1]
                        stop_dist = stop_cand - entry_cand
                        if stop_dist > 0 and stop_dist / entry_cand >= 0.0005:
                            position = -1
                            entry_price, entry_time = entry_cand, times[i + 1]
                            stop = stop_cand
                            target = entry_price - 2 * stop_dist
                        has_push, pullback_seen = False, False
                    else:
                        has_push, pullback_seen = True, False
            elif has_push and not pullback_seen:
                if pullback_bar_short[i]:
                    pullback_seen = True
            else:
                if push_now:
                    has_push, pullback_seen = True, False

    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "side", "pnl_r"])


long_trades = run_side(1)
short_trades = run_side(-1)
all_trades = pd.concat([long_trades, short_trades]).sort_values("entry_time").reset_index(drop=True)


def stats(trades):
    if trades.empty:
        return dict(n=0, win_rate=np.nan, total_r=0.0, sharpe=np.nan, max_dd=0.0)
    daily = trades.groupby("date")["pnl_r"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    cum = trades["pnl_r"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return dict(n=len(trades), win_rate=(trades["pnl_r"] > 0).mean(),
                total_r=trades["pnl_r"].sum(), sharpe=sharpe, max_dd=max_dd)


print(f"\nLong (H1/H2) trades: {len(long_trades)}")
print(f"Short (L1/L2) trades: {len(short_trades)}")

s = stats(all_trades)
print(f"\n=== Combined: n={s['n']}  win={s['win_rate']:.1%}  "
      f"total_R={s['total_r']:.1f}  sharpe={s['sharpe']:+.2f}  max_dd_R={s['max_dd']:.1f}")

print("\n=== Year-by-year ===")
all_trades["year"] = all_trades["date"].apply(lambda d: d.year)
for year, g in all_trades.groupby("year"):
    ss = stats(g)
    print(f"{year}: n={ss['n']:4d}  win={ss['win_rate']:.1%}  "
          f"total_R={ss['total_r']:7.1f}  sharpe={ss['sharpe']:+.2f}  dd_R={ss['max_dd']:7.1f}")

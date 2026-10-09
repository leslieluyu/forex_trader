"""
Long-history (2010-2026) robustness check for the ORB (opening range breakout)
candidate that looked promising on the 3.5-month XAUUSDm screening run
(backtest_intraday_screen.py), which showed a suspiciously high Sharpe ~3.54.

Data: HistData.com XAUUSD M1 bars, 2010-01 -> 2026-10 (5.87M rows).
No recorded spread in this feed (HistData gives bid-only OHLC), so cost is
modeled as a proportional spread estimated from the live Exness demo spread
(median 0.24 on ~4100 price ~= 5.85bps round trip) applied to historical price
levels. This is an approximation, not the real historical spread -- flagged
explicitly because it changes with gold's price level over 16 years.

Two checks:
  1. Year-by-year performance of the ORIGINAL fixed params (30min range,
     1x stop, 4h max hold) -- is it consistently positive, or was the
     3.5-month result a lucky regime?
  2. Full-history parameter sensitivity grid -- is performance a stable
     plateau across neighboring parameters, or a knife-edge single point?
"""
import numpy as np
import pandas as pd

SPREAD_BPS = 5.85e-4  # ~0.24 / 4100, round-trip, applied proportionally
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

df = pd.read_parquet(DATA_PATH).sort_values("time").reset_index(drop=True)
df["spread_px"] = df["close"] * SPREAD_BPS
df["date"] = df["time"].dt.date
df["year"] = df["time"].dt.year

print(f"Loaded {len(df)} M1 bars, {df['date'].nunique()} trading days, "
      f"{df['time'].min()} -> {df['time'].max()}")


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
                    trades.append((date, entry_time, row.time, position, pnl))
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


# ---------------------------------------------------------------------------
# 1. Year-by-year with original fixed params
# ---------------------------------------------------------------------------
print("\n=== Year-by-year, fixed params (30min/1.0x/240min) ===")
all_trades = backtest_orb(df)
for year, g in all_trades.groupby(all_trades["date"].apply(lambda d: d.year)):
    s = stats(g)
    print(f"{year}: n={s['n']:4d}  win={s['win_rate']:.1%}  "
          f"pnl={s['total_pnl']:8.1f}  sharpe={s['sharpe']:+.2f}  dd={s['max_dd']:8.1f}")

overall = stats(all_trades)
print(f"\nFULL 2010-2026: n={overall['n']}  win={overall['win_rate']:.1%}  "
      f"pnl={overall['total_pnl']:.1f}  sharpe={overall['sharpe']:+.2f}  "
      f"dd={overall['max_dd']:.1f}")

# ---------------------------------------------------------------------------
# 2. Parameter sensitivity grid (full history)
# ---------------------------------------------------------------------------
print("\n=== Parameter sensitivity grid (full 2010-2026 history) ===")
print(f"{'range_min':>10} {'stop_mult':>10} {'max_hold':>10} "
      f"{'n':>6} {'win%':>7} {'sharpe':>8} {'pnl':>10}")
for range_minutes in [15, 30, 60]:
    for stop_mult in [0.5, 1.0, 1.5, 2.0]:
        for max_hold in [120, 240, 480]:
            t = backtest_orb(df, range_minutes=range_minutes,
                              max_hold_minutes=max_hold, stop_mult=stop_mult)
            s = stats(t)
            print(f"{range_minutes:>10} {stop_mult:>10.1f} {max_hold:>10} "
                  f"{s['n']:>6} {s['win_rate']:>6.1%} {s['sharpe']:>+8.2f} "
                  f"{s['total_pnl']:>10.1f}")

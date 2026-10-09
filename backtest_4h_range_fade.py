"""
Systematic test of the strategy in bilibili video BV1UG1YBpEnY ("史上最强5分钟
短线交易策略" / Chinese-dubbed repost of "The BEST 5 Minute Scalping Strategy
Ever" by Data Trader). Video has no subtitle track; content extracted via
downloaded mp4 + ffmpeg frame sampling (burned-in Chinese captions).

Video's stated 3-step method (applies to BTC/USDT, EUR/USD, XAU/USD in the
video's own examples):
  Step 1: take the day's FIRST 4-hour candle -> its high/low defines the
          day's "range".
  Step 2: watch subsequent 5-minute candles; a breakout is confirmed when a
          5m candle CLOSES outside the range (either side).
  Step 3: wait for price to close back INSIDE the range (false-breakout
          confirmation) -> enter a fade trade in that direction at the
          re-entry close. Stop placed at the range's far boundary (the side
          that was broken); target = at least 2x the risk (video explicitly
          states "risk/reward >= 1:2").

This is architecturally the same "fade a failed range-breakout" idea already
coded generically in backtest_fade_failed_breakout.py (rolling Donchian
channel, arbitrary lookback), but this script matches the video's specific
framing: the range is anchored to the day's first 4H candle (not a rolling
window), and stop/target follow the video's literal rule (stop at range
boundary, target = 2R) instead of a timeout/midline exit.

Simplification disclosed: HistData timestamps are fixed EST (no DST), so
"first 4H candle of the day" = 00:00-04:00 in the data's own clock, which is
a reasonable proxy for the video's "first 4H NY-time candle" (within ~1h of
true NY time depending on DST) -- exact session-boundary calibration is not
critical to the strategy's core logic (sweep-and-fade of a session range).

Cost model: same proportional spread as backtest_fade_failed_breakout.py
(5.85bps round trip, from live Exness demo quotes) -- no swap cost modeled
since trades are intended to be intraday (added as a sanity check on holding
time, not assumed).
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

SPREAD_BPS = 5.85e-4
MIN_RR = 2.0


def build_bars(symbol):
    path = f"/Volumes/My Passport/working_data/forex_trader/data/{symbol}_histdata_M1.parquet"
    m1 = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    m1["date"] = m1["time"].dt.date
    m1["tod_min"] = m1["time"].dt.hour * 60 + m1["time"].dt.minute

    # Step 1: first 4H candle (00:00-03:59) of each day -> range
    first4h = m1[m1["tod_min"] < 240]
    rng = first4h.groupby("date").agg(range_high=("high", "max"), range_low=("low", "min"))

    # Step 2/3 operate on 5-minute bars for the REST of the day (>=04:00)
    rest = m1[m1["tod_min"] >= 240].copy()
    bars = (rest.set_index("time")
            .groupby("date" if False else pd.Grouper(freq="5min"))
            .agg(open=("open", "first"), high=("high", "max"),
                 low=("low", "min"), close=("close", "last"))
            .dropna().reset_index())
    bars["date"] = bars["time"].dt.date
    bars = bars.merge(rng, on="date", how="inner")
    return bars


def run(symbol):
    bars = build_bars(symbol)
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    rhi = bars["range_high"].to_numpy()
    rlo = bars["range_low"].to_numpy()
    times = bars["time"].to_numpy()
    dates = bars["date"].to_numpy()
    spread = (bars["close"] * SPREAD_BPS).to_numpy()
    n = len(bars)

    trades = []
    watch_dir = 0  # +1 = watching upside breakout (-> fade short), -1 downside (-> fade long)
    position = 0
    entry_price = entry_time = stop = target = None

    for i in range(n):
        if position != 0:
            if dates[i] != dates[i - 1]:
                # day rolled over without resolving -> force close at prior close (EOD timeout)
                exit_price = close[i - 1]
                raw = (exit_price - entry_price) * position
                trades.append((dates[i - 1], entry_time, times[i - 1], position, raw - spread[i - 1]))
                position = 0
                watch_dir = 0
            else:
                hit_stop = (position == -1 and high[i] >= stop) or (position == 1 and low[i] <= stop)
                hit_tgt = (position == -1 and low[i] <= target) or (position == 1 and high[i] >= target)
                if hit_stop or hit_tgt:
                    exit_price = stop if hit_stop else target
                    raw = (exit_price - entry_price) * position
                    trades.append((dates[i], entry_time, times[i], position, raw - spread[i]))
                    position = 0
                continue

        if dates[i] != (dates[i - 1] if i > 0 else None):
            watch_dir = 0  # new day -> new range, reset watch

        if watch_dir == 0:
            if close[i] > rhi[i]:
                watch_dir = 1
            elif close[i] < rlo[i]:
                watch_dir = -1
        else:
            failed = (watch_dir == 1 and close[i] < rhi[i]) or (watch_dir == -1 and close[i] > rlo[i])
            if failed:
                position = -watch_dir
                entry_price, entry_time = close[i], times[i]
                if position == -1:  # short fade of upside breakout
                    stop = rhi[i]
                    risk = stop - entry_price
                    target = entry_price - MIN_RR * risk
                else:  # long fade of downside breakout
                    stop = rlo[i]
                    risk = entry_price - stop
                    target = entry_price + MIN_RR * risk
                watch_dir = 0
                if risk <= 1e-9:
                    position = 0  # degenerate range, skip

    t = pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "side", "pnl"])
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["risk"] = (t["pnl"] * 0 + 1)  # placeholder, replaced below
    # recompute risk properly per trade for equity_sim (needs entry_price, risk, pnl columns)
    return t


def run_with_risk(symbol):
    bars = build_bars(symbol)
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    rhi = bars["range_high"].to_numpy()
    rlo = bars["range_low"].to_numpy()
    times = bars["time"].to_numpy()
    dates = bars["date"].to_numpy()
    spread = (bars["close"] * SPREAD_BPS).to_numpy()
    n = len(bars)

    trades = []
    watch_dir = 0
    position = 0
    entry_price = entry_time = stop = target = risk0 = None

    for i in range(n):
        if position != 0:
            same_day = i > 0 and dates[i] == dates[i - 1]
            if not same_day:
                exit_price = close[i - 1]
                raw = (exit_price - entry_price) * position
                trades.append((entry_time, times[i - 1], entry_price, risk0, raw - spread[i - 1]))
                position = 0
                watch_dir = 0
            else:
                hit_stop = (position == -1 and high[i] >= stop) or (position == 1 and low[i] <= stop)
                hit_tgt = (position == -1 and low[i] <= target) or (position == 1 and high[i] >= target)
                if hit_stop or hit_tgt:
                    exit_price = stop if hit_stop else target
                    raw = (exit_price - entry_price) * position
                    trades.append((entry_time, times[i], entry_price, risk0, raw - spread[i]))
                    position = 0
                continue

        if i == 0 or dates[i] != dates[i - 1]:
            watch_dir = 0

        if watch_dir == 0:
            if close[i] > rhi[i]:
                watch_dir = 1
            elif close[i] < rlo[i]:
                watch_dir = -1
        else:
            failed = (watch_dir == 1 and close[i] < rhi[i]) or (watch_dir == -1 and close[i] > rlo[i])
            if failed:
                cand_pos = -watch_dir
                cand_entry = close[i]
                if cand_pos == -1:
                    cand_stop = rhi[i]
                    cand_risk = cand_stop - cand_entry
                    cand_target = cand_entry - MIN_RR * cand_risk
                else:
                    cand_stop = rlo[i]
                    cand_risk = cand_entry - cand_stop
                    cand_target = cand_entry + MIN_RR * cand_risk
                watch_dir = 0
                if cand_risk > 1e-9:
                    position, entry_price, entry_time = cand_pos, cand_entry, times[i]
                    stop, target, risk0 = cand_stop, cand_target, cand_risk

    t = pd.DataFrame(trades, columns=["entry_time", "exit_time", "entry_price", "risk", "pnl"])
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["hold_min"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 60
    return t


def report(symbol):
    t = run_with_risk(symbol)
    wins = t[t["pnl"] > 0]["pnl"]
    losses = t[t["pnl"] <= 0]["pnl"]
    payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    es = equity_stats(t)
    print(f"=== {symbol}: 4H-range breakout-fade (video BV1UG1YBpEnY's 3-step method) ===")
    print(f"n={len(t)}  win={(t['pnl']>0).mean():.1%}  payoff={payoff:.2f}  pf={pf:.2f}  "
          f"median hold(min)={t['hold_min'].median():.0f}")
    print(f"  ret%={es['total_return_pct']:8.1f}  dd%={es['max_dd_pct']:7.1f}  "
          f"Sharpe={es['sharpe']:.2f}  CAGR%={(es['cagr']*100 if es['cagr']==es['cagr'] else float('nan')):.2f}")

    # rolling 1-year walk-forward fold count (pos/neg) for honesty check
    t["year"] = t["entry_time"].dt.year
    pos, neg = 0, 0
    for y, g in t.groupby("year"):
        if len(g) < 5:
            continue
        r = equity_stats(g)["total_return_pct"]
        if r == r:
            (pos := pos + 1) if r > 0 else (neg := neg + 1)
    print(f"  yearly folds: positive={pos}  negative={neg}")
    print()
    return t


if __name__ == "__main__":
    for sym in ["XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"]:
        report(sym)

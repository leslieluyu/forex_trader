"""
Test of the MA-trend "一单一结" strategy shown in bilibili video BV1ade76tEXx
(QS顺势者, "一年近30倍，黄金XAU为例，分享一单一结小止损趋势EA方案").

Video shows MT5 Strategy Tester backtest of a closed-source EA "SS30.ex5" on
XAUUSDm M30, single test window 2025.09.01 -> 2026.09.11 (~1 year), starting
balance $1000, ending balance ~$27,377 (84 trades, win rate 45.2%) by the last
fully-read frame near the end of the video -- consistent with the title's
"近30倍" (~30x) claim.

Mechanism extracted from on-screen elements + burned-in captions (EA itself
is closed-source .ex5, so this is a MECHANICAL RECONSTRUCTION of the general
idea, not a literal replica):
  - Chart shows 4 moving averages: MA(3), MA(8), MA(21), MA(60) + ATR(14).
    A "white line" and a "yellow line" are highlighted in the UI; the yellow
    line visually tracks MA(60) across frames.
  - UI buttons are literally labeled "黄线做多" / "黄线做空" (go long/short
    "at the yellow line") -- the entry/trend signal is price's relationship
    to the yellow line (MA60).
  - Caption: "我们把这个止损设在黄线附近...只要这个黄线不破" -- stop loss is
    anchored near the yellow line, position held as long as the yellow line
    isn't broken -- MA60 acts as a trailing/ratcheting stop once a trade is
    open.
  - Caption: "它的止损啊固定止损是200啊" / the trade log's "最大浮亏" column
    clusters tightly around -$200 at the shown 0.2-lot size -- there is ALSO
    a fixed initial protective stop worth ~$200 per 0.2 lot (before the
    trailing-to-MA60 stop has anything meaningful to trail on).
  - Described as "波段持仓" (swing/position holding), low win rate (25%-45%
    across the growing trade sample) with a few large winners carrying the
    equity curve -- classic trend-following payoff profile.

Mechanical reconstruction v1 (sign(close-MA60), reverse on every cross) was
tried first and discarded: it produced ~700-750 trades/year on M30, vs. the
video's own trade log showing only 84 trades across the full ~1-year window
(~1 trade every 4+ days) -- a 10x+ overtrading mismatch, meaning a bare MA60
cross is not a faithful reconstruction (way too much whipsaw reversal on a
60-period MA at 30-min granularity).

Mechanical reconstruction v2 (what's actually run below) adds the
confirmation filter the 4-MA display on screen (MA3/MA8/MA21/MA60) most
plausibly exists for -- full stack alignment, entered only on a FRESH
alignment (not held continuously flat-to-aligned), which is the standard
reading of "why would you plot 4 MAs" if not to require them all stacked
before trusting a trend:
  - Long entry: close > MA3 > MA8 > MA21 > MA60, and this stack was NOT
    already fully aligned on the previous bar (fresh signal, not mid-trend
    re-trigger). Short entry: mirror image.
  - Reverse-and-flip: closes current position (if any) the moment a fresh
    opposite-direction alignment fires.
  - Initial stop = entry -/+ STOP_PRICE_DIST, calibrated so a 0.2-lot
    position has ~$200 max loss: STOP_PRICE_DIST = 200 / (0.2 * contract_size).
  - Trailing stop: once open, the stop ratchets toward (never away from)
    the current MA60 value each bar -- actual_stop = max(initial_stop, MA60)
    for longs / min(initial_stop, MA60) for shorts ("止损设在黄线附近...只
    要黄线不破").
  - Exit: close crosses the stop level that bar (approximated with M30
    close since the project only has HistData M1-derived bars, not true
    broker ticks), OR a fresh opposite alignment fires, whichever first.
  - No take-profit: winners held until the trailing MA60 stop or an
    opposite alignment closes them (matches "波段持仓" / let-winners-run).

This is still a RECONSTRUCTION, not the literal closed-source EA -- flagged
explicitly because changing the entry filter changed the trade count by an
order of magnitude, i.e. the result is sensitive to a guess. Treat the
numbers below as "does a 4-MA-stack trend-following idea in this spirit
survive 16 years", not as a replication of the exact 84-trades/+2638% curve
shown in the video.

Tested on BOTH the full ~16-year EURUSD/GBPUSD/AUDUSD/XAUUSD H1 history
(does the mechanism survive the long run) AND the video's own 1-year window
2025.09.01-2026.09.11 (does the headline number replicate on an independent
price series over the SAME calendar window the video used -- the video's own
test only ever ran on XAUUSDm M30 via the broker's own feed, never checked
against HistData).
"""
import numpy as np
import pandas as pd

DATA_DIR = "/Volumes/My Passport/working_data/forex_trader/data"
SYMBOLS = {
    "EURUSD": 100_000.0,
    "GBPUSD": 100_000.0,
    "AUDUSD": 100_000.0,
    "XAUUSD": 100.0,
}

INITIAL_BALANCE = 1_000.0     # matches video's shown starting balance
LOT = 0.2                     # matches video's shown lot size
MA_FAST, MA_MID, MA_SLOW = 3, 8, 21
MA_TREND = 60                 # the "yellow line"
FIXED_STOP_USD_PER_02LOT = 200.0

VIDEO_WINDOW = ("2025-09-01", "2026-09-11")


def load_bars(symbol, freq="30min"):
    m1 = pd.read_parquet(f"{DATA_DIR}/{symbol}_histdata_M1.parquet").sort_values("time").set_index("time")
    o = m1["open"].resample(freq).first()
    h = m1["high"].resample(freq).max()
    l = m1["low"].resample(freq).min()
    c = m1["close"].resample(freq).last()
    bars = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    return bars


def run_ma_trend(bars, contract_size, lot=LOT, ma_trend=MA_TREND):
    return run_ma_trend_from(bars, contract_size, lot=lot, ma_trend=ma_trend, start_ts=None)


ATR_PERIOD = 14
ATR_MULT = 1.5   # trailing-stop buffer below/above MA60, in ATRs (not a literal
                 # buffer -- a guess to reduce the whipsaw a bare MA60 touch causes)


def run_ma_trend_from(bars, contract_size, lot=LOT, ma_trend=MA_TREND, start_ts=None):
    """Same engine as run_ma_trend, but if start_ts is given, lets the MAs
    warm up on bars before start_ts (a position may already be open going
    in) and only resets balance/trade accounting to begin fresh exactly at
    start_ts -- i.e. isolates the P&L generated strictly within
    [start_ts, end]."""
    close = bars["close"]
    ma_f = close.rolling(MA_FAST).mean()
    ma_m = close.rolling(MA_MID).mean()
    ma_s = close.rolling(MA_SLOW).mean()
    ma_t = close.rolling(ma_trend).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()
    stop_dist = FIXED_STOP_USD_PER_02LOT / (0.2 * contract_size)  # price units, independent of lot

    px = close.to_numpy()
    hi = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    maf, mam, mas, mat = ma_f.to_numpy(), ma_m.to_numpy(), ma_s.to_numpy(), ma_t.to_numpy()
    atrv = atr.to_numpy()
    reset_idx = bars.index.get_loc(start_ts) if start_ts is not None else (ma_trend + ATR_PERIOD)

    start = ma_trend + ATR_PERIOD  # first bar with all MAs + ATR valid
    direction = 0            # 0 = flat, +1 long, -1 short
    entry_price = None
    stop_level = None
    balance = INITIAL_BALANCE
    trades = []
    peak_equity = INITIAL_BALANCE
    max_dd_pct = 0.0
    prev_long_aligned = prev_short_aligned = False

    for i in range(start, len(px)):
        if np.isnan(mat[i]) or np.isnan(atrv[i]):
            continue
        price = px[i]
        long_aligned = price > maf[i] > mam[i] > mas[i] > mat[i]
        short_aligned = price < maf[i] < mam[i] < mas[i] < mat[i]
        fresh_long = long_aligned and not prev_long_aligned
        fresh_short = short_aligned and not prev_short_aligned

        if i == reset_idx:
            # start fresh accounting here; if a position is already open
            # from the warmup period, treat its current price as a fresh
            # entry (marks-to-market, doesn't retroactively book warmup PnL)
            balance = INITIAL_BALANCE
            trades = []
            peak_equity = INITIAL_BALANCE
            max_dd_pct = 0.0
            if direction != 0:
                entry_price = price
                stop_level = price - direction * stop_dist

        if direction != 0:
            floating = lot * contract_size * (price - entry_price) * direction
            equity = balance + floating
            peak_equity = max(peak_equity, equity)
            max_dd_pct = min(max_dd_pct, (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else -100.0)

            # ratchet trailing stop toward MA60 (minus an ATR buffer so a
            # single wick touching MA60 doesn't instantly stop the trade
            # out), never loosen
            buffer = ATR_MULT * atrv[i]
            if direction == 1:
                stop_level = max(stop_level, mat[i] - buffer)
                stopped_out = lo[i] <= stop_level
            else:
                stop_level = min(stop_level, mat[i] + buffer)
                stopped_out = hi[i] >= stop_level

            flip_signal = fresh_short if direction == 1 else fresh_long

            if stopped_out:
                exit_price = stop_level
                pnl = lot * contract_size * (exit_price - entry_price) * direction
                balance += pnl
                trades.append(pnl)
                direction = 0
            elif flip_signal:
                pnl = lot * contract_size * (price - entry_price) * direction
                balance += pnl
                trades.append(pnl)
                direction = -direction
                entry_price = price
                stop_level = price - direction * stop_dist
        else:
            # flat: enter only on a fresh full-stack alignment
            if fresh_long:
                direction = 1
            elif fresh_short:
                direction = -1
            if direction != 0:
                entry_price = price
                stop_level = price - direction * stop_dist

        prev_long_aligned, prev_short_aligned = long_aligned, short_aligned

    if direction != 0:
        pnl = lot * contract_size * (px[-1] - entry_price) * direction
        balance += pnl
        trades.append(pnl)

    trades = np.array(trades)
    n = len(trades)
    win_rate = (trades > 0).mean() * 100 if n else 0.0
    return dict(
        n_trades=n,
        win_rate=win_rate,
        final_balance=balance,
        total_return_pct=(balance / INITIAL_BALANCE - 1) * 100,
        max_dd_pct=max_dd_pct,
        avg_win=trades[trades > 0].mean() if (trades > 0).any() else 0.0,
        avg_loss=trades[trades < 0].mean() if (trades < 0).any() else 0.0,
    )


if __name__ == "__main__":
    for symbol, contract_size in SYMBOLS.items():
        bars = load_bars(symbol)
        print(f"\n=== {symbol}  full history: {bars.index.min().date()} -> {bars.index.max().date()}  ({len(bars)} M30 bars) ===")

        r_full = run_ma_trend(bars, contract_size)
        print(f"  [16-year full history]  trades={r_full['n_trades']:4d}  win_rate={r_full['win_rate']:5.1f}%  "
              f"final=${r_full['final_balance']:,.0f} ({r_full['total_return_pct']:+8.1f}%)  "
              f"max_dd={r_full['max_dd_pct']:6.1f}%  avg_win=${r_full['avg_win']:,.1f}  avg_loss=${r_full['avg_loss']:,.1f}")

        window = bars.loc[VIDEO_WINDOW[0]:VIDEO_WINDOW[1]]
        if len(window) > MA_TREND:
            # need enough lookback before window start so MA60 is warmed up
            # AT the window's first bar (not mid-window) -- run the engine on
            # a lookback+window slice, same as it would behave if it had been
            # live and running continuously, then reset balance/trades to
            # start fresh exactly at the window boundary.
            lookback_start = bars.index.get_loc(window.index[0])
            warmed = bars.iloc[max(0, lookback_start - MA_TREND):]
            r_window = run_ma_trend_from(warmed, contract_size, start_ts=window.index[0])
            print(f"  [video's 1-yr window {VIDEO_WINDOW[0]}->{VIDEO_WINDOW[1]}, {len(window)} bars]  "
                  f"trades={r_window['n_trades']:4d}  win_rate={r_window['win_rate']:5.1f}%  "
                  f"final=${r_window['final_balance']:,.0f} ({r_window['total_return_pct']:+8.1f}%)  "
                  f"max_dd={r_window['max_dd_pct']:6.1f}%  avg_win=${r_window['avg_win']:,.1f}  avg_loss=${r_window['avg_loss']:,.1f}")
        else:
            print(f"  [video's 1-yr window] insufficient bars ({len(window)}) in local data for this symbol/window")

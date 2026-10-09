"""
Test of the "EMA trend + HILO filter + mild pyramiding grid" gold EA described
in the zhihu article (p/1972953303836025110) "经历了三次爆仓，终于把黄金EA调
顺了｜我的量化策略从0到稳定" (author zzNewM).

Unlike the bilibili videos tested elsewhere in this project, this article is
TEXT ONLY -- no screenshots, no MT5 Strategy Tester trade log, no backtest
curve, no win rate / return numbers of any kind. It only lists the EA's
parameter names and values and a few sentences of qualitative description
("trend moves make money, ranging gets small losses, extreme moves survive").
There is nothing to independently verify against -- this script reconstructs
the MECHANISM from the parameter list and tests whether it has any edge,
not a replication of a claimed result (there is no claimed result).

Mechanism as described:
  - Algo1 (direction filter): EMA(40) on M15. EMA rising -> only allow longs;
    EMA falling -> only allow shorts ("EMA上升=只做多，EMA下降=只做空").
  - Algo2 (entry filter): "HILO 通道" period 7-10 (used 9), described as
    filtering false breakouts / confirming the EMA direction. Reconstructed
    as the classic Gann HiLo Activator: hilo = prior N-bar low if close was
    above the previous hilo value, else prior N-bar high; direction flips
    when price crosses it. Reconstructed on the M5 execution timeframe
    (article says "M5图执行操作" but does not say which timeframe HILO itself
    runs on -- M5 is the natural guess since it's the only other named frame).
  - Entry: M5 bar where HILO direction is fresh-aligned with the M15 EMA
    direction (avoids re-firing every bar of an already-aligned regime).
  - Pyramiding ("顺势网格(温和型)"): starting lot 0.01 (not stated in the
    article; a standard micro-lot default), each further GridSize-point
    favorable move adds GridFactor(=1.25)x the last added lot, capped at
    MaxLot=2.0 total. This is pyramiding INTO winners, not grid-against-
    losers martingale (article explicitly contrasts it with "死扛" grid EAs).
  - Exit: TP = 90 points from average entry; BreakEven stop activates once
    price is 20 points in favor (moves stop to entry+13 points); Trailing
    stop activates once price is 90 points in favor (trails 60 points behind,
    ratchets only in favor). Whichever stop/TP is hit first on that bar's
    high/low closes the whole position.
  - Opposite fresh signal while a position is open closes it at market and
    flips.
  - Equity risk control: EquityStop=30% -- if floating equity drawdown from
    its running peak reaches -30%, force-close the open position (crystallize
    the loss) rather than let it compound further; trading continues after.

Points conversion is a GUESS, documented explicitly because the result is
sensitive to it: XAUUSD point = 0.01 (2-decimal gold quote), FX majors point
= 0.00001 (5-decimal quote). This makes TP/grid/trailing distances very tight
in price terms (e.g. XAUUSD TP = $0.90, grid step = $0.10) -- consistent with
a fast M5 scalping-style EA, but if the real EA used a different points
convention the trade frequency and payoff profile here would not match it.

Tested on the full ~16-year EURUSD/GBPUSD/AUDUSD/XAUUSD H1-equivalent
(constructed from HistData M1) history -- the article gives no specific test
window to replicate, so there is no "video window" counterpart test here,
only the general survive-16-years stress test used throughout this project.
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
POINT = {
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "AUDUSD": 0.00001,
    "XAUUSD": 0.01,
}

INITIAL_BALANCE = 1_000.0
BASE_LOT = 0.01
GRID_FACTOR = 1.25
MAX_LOT = 2.0
GRID_SIZE_PTS = 10
TP_PTS = 90
TRAIL_START_PTS = 90
TRAIL_STOP_PTS = 60
BE_START_PTS = 20
BE_STEP_PTS = 13
EQUITY_STOP_PCT = 30.0

EMA_PERIOD = 40
HILO_PERIOD = 9


def load_m1(symbol):
    return pd.read_parquet(f"{DATA_DIR}/{symbol}_histdata_M1.parquet").sort_values("time").set_index("time")


def resample_ohlc(m1, freq):
    o = m1["open"].resample(freq).first()
    h = m1["high"].resample(freq).max()
    l = m1["low"].resample(freq).min()
    c = m1["close"].resample(freq).last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def compute_ema_direction(m15):
    ema = m15["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    direction = np.where(ema.to_numpy() > np.concatenate([[np.nan], ema.to_numpy()[:-1]]), 1, -1)
    direction[: EMA_PERIOD] = 0  # not warmed up yet
    return pd.Series(direction, index=m15.index)


def compute_hilo_direction(m5):
    hi_n = m5["high"].rolling(HILO_PERIOD).max().shift(1).to_numpy()
    lo_n = m5["low"].rolling(HILO_PERIOD).min().shift(1).to_numpy()
    close = m5["close"].to_numpy()
    n = len(close)
    hilo_val = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    for i in range(HILO_PERIOD, n):
        if np.isnan(hi_n[i]) or np.isnan(lo_n[i]):
            continue
        prev_val = hilo_val[i - 1]
        if np.isnan(prev_val):
            direction[i] = 1 if close[i] > lo_n[i] else -1
            hilo_val[i] = lo_n[i] if direction[i] == 1 else hi_n[i]
            continue
        if close[i] > prev_val:
            direction[i] = 1
            hilo_val[i] = lo_n[i]
        elif close[i] < prev_val:
            direction[i] = -1
            hilo_val[i] = hi_n[i]
        else:
            direction[i] = direction[i - 1]
            hilo_val[i] = prev_val
    return pd.Series(direction, index=m5.index)


def run_ema_hilo_grid(symbol, contract_size):
    m1 = load_m1(symbol)
    m15 = resample_ohlc(m1, "15min")
    m5 = resample_ohlc(m1, "5min")
    point = POINT[symbol]

    ema_dir = compute_ema_direction(m15)
    # map each M15 direction value forward onto M5 bars, using only the
    # most recently CLOSED M15 bar strictly before the M5 bar's own close
    ema_on_m5 = ema_dir.reindex(m5.index, method="ffill").shift(1).fillna(0).astype(int)

    hilo_dir = compute_hilo_direction(m5)
    prev_hilo = hilo_dir.shift(1).fillna(0)

    close = m5["close"].to_numpy()
    high = m5["high"].to_numpy()
    low = m5["low"].to_numpy()
    ema_arr = ema_on_m5.to_numpy()
    hilo_arr = hilo_dir.to_numpy()
    prev_hilo_arr = prev_hilo.to_numpy()

    grid_size = GRID_SIZE_PTS * point
    tp_dist = TP_PTS * point
    trail_start_dist = TRAIL_START_PTS * point
    trail_stop_dist = TRAIL_STOP_PTS * point
    be_start_dist = BE_START_PTS * point
    be_step_dist = BE_STEP_PTS * point

    balance = INITIAL_BALANCE
    direction = 0
    legs_lot = []   # list of (lot, entry_price)
    last_add_price = None
    stop_level = None
    be_active = False
    trail_active = False
    peak_equity = INITIAL_BALANCE
    max_dd_pct = 0.0
    trades = []
    forced_stopouts = 0
    blown = False

    start = max(EMA_PERIOD + 5, HILO_PERIOD + 5)

    for i in range(start, len(close)):
        if blown:
            break
        price = close[i]

        if direction != 0:
            total_lot = sum(l for l, _ in legs_lot)
            avg_price = sum(l * p for l, p in legs_lot) / total_lot
            floating = total_lot * contract_size * (price - avg_price) * direction
            equity = balance + floating
            peak_equity = max(peak_equity, equity)
            dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else -100.0
            max_dd_pct = min(max_dd_pct, dd_pct)

            # equity-level kill switch: force-flat on -30% drawdown from peak
            if dd_pct <= -EQUITY_STOP_PCT:
                pnl = total_lot * contract_size * (price - avg_price) * direction
                balance += pnl
                trades.append(pnl)
                forced_stopouts += 1
                direction = 0
                legs_lot = []
                stop_level = None
                be_active = trail_active = False
                if balance <= 0:
                    blown = True
                    continue
                peak_equity = balance
                continue

            # grid add (pyramid into winners), capped at MAX_LOT
            if total_lot < MAX_LOT and direction * (price - last_add_price) >= grid_size:
                next_lot = min(legs_lot[-1][0] * GRID_FACTOR, MAX_LOT - total_lot)
                if next_lot > 1e-6:
                    legs_lot.append((next_lot, price))
                    last_add_price = price
                    total_lot = sum(l for l, _ in legs_lot)
                    avg_price = sum(l * p for l, p in legs_lot) / total_lot

            favor = direction * (price - avg_price)

            if not be_active and favor >= be_start_dist:
                be_active = True
                stop_level = avg_price + direction * be_step_dist
            if favor >= trail_start_dist:
                trail_active = True
                candidate = price - direction * trail_stop_dist
                if stop_level is None:
                    stop_level = candidate
                elif direction == 1:
                    stop_level = max(stop_level, candidate)
                else:
                    stop_level = min(stop_level, candidate)

            tp_level = avg_price + direction * tp_dist
            hit_tp = (direction == 1 and high[i] >= tp_level) or (direction == -1 and low[i] <= tp_level)
            hit_stop = stop_level is not None and (
                (direction == 1 and low[i] <= stop_level) or (direction == -1 and high[i] >= stop_level)
            )

            fresh_flip = (
                ema_arr[i] != 0 and hilo_arr[i] == ema_arr[i] and hilo_arr[i] != prev_hilo_arr[i]
                and hilo_arr[i] == -direction
            )

            if hit_tp or hit_stop or fresh_flip:
                exit_price = tp_level if hit_tp else (stop_level if hit_stop else price)
                pnl = total_lot * contract_size * (exit_price - avg_price) * direction
                balance += pnl
                trades.append(pnl)
                direction = 0
                legs_lot = []
                stop_level = None
                be_active = trail_active = False
                if balance <= 0:
                    blown = True
                    continue
                if fresh_flip and not hit_tp and not hit_stop:
                    direction = hilo_arr[i]
                    legs_lot = [(BASE_LOT, price)]
                    last_add_price = price
                continue

        else:
            if ema_arr[i] != 0 and hilo_arr[i] == ema_arr[i] and hilo_arr[i] != prev_hilo_arr[i]:
                direction = hilo_arr[i]
                legs_lot = [(BASE_LOT, price)]
                last_add_price = price
                stop_level = None
                be_active = trail_active = False
                peak_equity = balance

    if direction != 0 and legs_lot:
        total_lot = sum(l for l, _ in legs_lot)
        avg_price = sum(l * p for l, p in legs_lot) / total_lot
        pnl = total_lot * contract_size * (close[-1] - avg_price) * direction
        balance += pnl
        trades.append(pnl)

    trades = np.array(trades)
    n = len(trades)
    win_rate = (trades > 0).mean() * 100 if n else 0.0
    blown_at = m5.index[i] if blown else None
    years_to_blow = (blown_at - m5.index[start]).days / 365.25 if blown else None
    total_years = (m5.index[-1] - m5.index[start]).days / 365.25
    return dict(
        n_trades=n,
        win_rate=win_rate,
        final_balance=balance,
        total_return_pct=(balance / INITIAL_BALANCE - 1) * 100,
        max_dd_pct=max_dd_pct,
        forced_equity_stopouts=forced_stopouts,
        blown=blown,
        blown_at=blown_at,
        years_to_blow=years_to_blow,
        total_years=total_years,
    )


if __name__ == "__main__":
    print(f"{'SYMBOL':8s} {'trades':>7s} {'win%':>7s} {'ret%':>12s} {'maxDD%':>9s} {'eqStops':>8s} {'blown':>6s} {'yrsToBlow':>10s} {'ofYrs':>7s}")
    for symbol, contract_size in SYMBOLS.items():
        r = run_ema_hilo_grid(symbol, contract_size)
        ytb = f"{r['years_to_blow']:.1f}" if r["years_to_blow"] is not None else "-"
        print(
            f"{symbol:8s} {r['n_trades']:7d} {r['win_rate']:6.1f}% {r['total_return_pct']:11.1f}% "
            f"{r['max_dd_pct']:8.1f}% {r['forced_equity_stopouts']:8d} {str(r['blown']):>6s} {ytb:>10s} {r['total_years']:7.1f}"
        )

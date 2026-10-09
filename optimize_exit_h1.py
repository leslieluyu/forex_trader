"""
Exit/take-profit optimization for the validated H1 Donchian(20)+SMA50(D1)
structure. Entry rule is held fixed (that's what's validated); only the
exit side is varied, to see whether the current "2x ATR trail OR
Donchian-10 close-exit, whichever first, no fixed TP" rule can be beaten.

Variants tested, all on the same H1 bars / same entry signals:
  1. ATR trail multiplier grid (how tight the trail is)
  2. Donchian exit-channel window grid (how tight the channel-exit is)
  3. Fixed R-multiple take-profit (full exit, no trail) at various R
  4. Partial profit-taking: close part of the position at N*R, let the
     rest ride on the existing trail/channel exit

Cost model matches research: $0.30 flat round-trip. Compounding equity
sim (1% risk, 5x leverage cap) via equity_sim.py.
"""
import numpy as np
import pandas as pd

from equity_sim import equity_stats

FLAT_COST = 0.30
DATA_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

m1 = pd.read_parquet(DATA_PATH).sort_values("time").set_index("time")

d1 = m1.resample("1D").agg(close=("close", "last")).dropna()
d1["sma50"] = d1["close"].rolling(50).mean()
d1["trend_ok"] = d1["close"].shift(1) > d1["sma50"].shift(1)
trend_ok_by_date = d1["trend_ok"].copy()
trend_ok_by_date.index = trend_ok_by_date.index.date

bars = m1.resample("1h").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last")).dropna()
bars = bars.reset_index()
bars["date"] = bars["time"].dt.date
bars["trend_ok"] = bars["date"].map(trend_ok_by_date).fillna(False)

o = bars["open"].to_numpy()
h = bars["high"].to_numpy()
l = bars["low"].to_numpy()
c = bars["close"].to_numpy()
trend_ok = bars["trend_ok"].to_numpy()
times = bars["time"].to_numpy()
dates = bars["date"].to_numpy()
n = len(bars)

prev_close = pd.Series(c).shift(1).to_numpy()
tr = np.maximum.reduce([
    h - l,
    np.abs(h - np.nan_to_num(prev_close, nan=h[0])),
    np.abs(l - np.nan_to_num(prev_close, nan=l[0])),
])
ATR_LEN = 20
DONCH_ENTRY = 20
atrN = pd.Series(tr).rolling(ATR_LEN, min_periods=ATR_LEN).mean().to_numpy()
patr = pd.Series(atrN).shift(1).to_numpy()
donch_high = pd.Series(h).rolling(DONCH_ENTRY).max().shift(1).to_numpy()

warmup = max(DONCH_ENTRY, ATR_LEN) + 5
signal_idxs = []
for i in range(warmup, n - 1):
    if np.isnan(donch_high[i]) or np.isnan(patr[i]) or patr[i] <= 0:
        continue
    if not (c[i] > donch_high[i]) or not trend_ok[i]:
        continue
    signal_idxs.append(i)


def simulate(atr_mult, donch_exit, tp_r=None, partial_r=None, partial_frac=0.5):
    """
    tp_r: if set, full exit at entry + tp_r*risk (no trail at all beyond that).
    partial_r: if set, close partial_frac of size at entry + partial_r*risk,
               remainder continues under the normal trail/channel exit.
    If both None: pure trail/channel exit (baseline).
    """
    low_exit = pd.Series(l).rolling(donch_exit).min().shift(1).to_numpy()
    trades = []
    i = 0
    last_exit_i = -1
    for sig_i in signal_idxs:
        if sig_i <= last_exit_i:
            continue
        entry_i = sig_i + 1
        if entry_i >= n:
            continue
        entry_price = float(o[entry_i])
        entry_time = times[entry_i]
        stop = entry_price - atr_mult * float(patr[sig_i])
        risk = entry_price - stop
        if risk <= 0.01:
            continue
        tp_price = entry_price + tp_r * risk if tp_r is not None else None
        partial_price = entry_price + partial_r * risk if partial_r is not None else None
        partial_done = False
        remaining_frac = 1.0
        cur_stop = stop
        exit_i, exit_price = None, None
        pnl_frac_total = 0.0  # weighted avg exit price contribution tracked via explicit pnl calc
        realized_pnl = 0.0
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrN[j - 1]) and atrN[j - 1] > 0:
                cur_stop = max(cur_stop, c[j - 1] - atr_mult * atrN[j - 1])
            # full TP (overrides trail entirely)
            if tp_price is not None and h[j] >= tp_price:
                exit_i, exit_price = j, float(tp_price)
                break
            # partial profit-take (once), reduces remaining size, continue trailing rest
            if (not partial_done) and partial_price is not None and h[j] >= partial_price:
                realized_pnl += partial_frac * ((partial_price - entry_price) - 0)
                remaining_frac -= partial_frac
                partial_done = True
                continue
            if l[j] <= cur_stop:
                exit_i, exit_price = j, float(cur_stop)
                break
            if j > entry_i and not np.isnan(low_exit[j]) and c[j] < low_exit[j]:
                exit_i, exit_price = j, float(c[j])
                break
        if exit_i is None:
            exit_i, exit_price = n - 1, float(c[n - 1])
        realized_pnl += remaining_frac * (exit_price - entry_price)
        pnl = realized_pnl - FLAT_COST
        trades.append((dates[exit_i], entry_time, times[exit_i], entry_price, risk, pnl))
        last_exit_i = exit_i
    return pd.DataFrame(trades, columns=["date", "entry_time", "exit_time", "entry_price", "risk", "pnl"])


def report(label, t):
    if t.empty:
        print(f"{label:<28} no trades")
        return
    es = equity_stats(t)
    win = (t["pnl"] > 0).mean()
    wins = t[t["pnl"] > 0]["pnl"]
    losses = t[t["pnl"] <= 0]["pnl"]
    payoff = wins.mean() / abs(losses.mean()) if len(losses) and len(wins) else np.nan
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    print(f"{label:<28} n={len(t):4d} win={win:5.1%} payoff={payoff:5.2f} pf={pf:5.2f} "
          f"ret%={es['total_return_pct']:8.1f} dd%={es['max_dd_pct']:7.1f} Sharpe={es['sharpe']:5.2f}")


print("=== Baseline (current production rule) ===")
report("ATR2.0x trail + Donch10 exit", simulate(atr_mult=2.0, donch_exit=10))

print("\n=== ATR trail multiplier grid (Donch exit fixed at 10) ===")
for m in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    report(f"ATR{m}x trail", simulate(atr_mult=m, donch_exit=10))

print("\n=== Donchian exit-channel window grid (ATR mult fixed at 2.0) ===")
for dx in [5, 10, 15, 20, 30]:
    report(f"Donch{dx} exit", simulate(atr_mult=2.0, donch_exit=dx))

print("\n=== Fixed R-multiple take-profit (full exit, replaces trail) ===")
for r in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    report(f"TP={r}R fixed", simulate(atr_mult=2.0, donch_exit=10, tp_r=r))

print("\n=== Partial profit-taking (take 50% off at N*R, rest rides trail) ===")
for r in [1.0, 1.5, 2.0, 3.0]:
    report(f"50% off @{r}R + trail", simulate(atr_mult=2.0, donch_exit=10, partial_r=r, partial_frac=0.5))

print("\n=== Partial profit-taking (take 33% off at N*R, rest rides trail) ===")
for r in [1.0, 1.5, 2.0, 3.0]:
    report(f"33% off @{r}R + trail", simulate(atr_mult=2.0, donch_exit=10, partial_r=r, partial_frac=0.33))

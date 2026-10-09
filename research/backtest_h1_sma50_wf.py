#!/usr/bin/env python3
"""
Deep dive: H1 Donchian long-only + prior daily SMA50 filter on Dukascopy XAU/USD.
Identical exit/entry rules to backtest_freq.py backtest_h1_donchian (long-only + filter).
Walk-forward with FIXED params (no re-optimization).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = "/workspace/backtest_gold"
DATA_H1 = os.path.join(ROOT, "xauusd_h1_2010_2026.parquet")
DATA_M5 = os.path.join(ROOT, "xauusd_m5.parquet")  # fallback fill if needed
RESULTS_CSV = os.path.join(ROOT, "results_h1_sma50_wf.csv")
TRADES_CSV = os.path.join(ROOT, "trades_h1_sma50_wf.csv")
EQUITY_PNG = os.path.join(ROOT, "equity_h1_sma50_wf.png")
CODE_PATH = os.path.join(ROOT, "backtest_h1_sma50_wf.py")

INITIAL = 10_000.0
RISK_PCT = 0.01
MAX_LEV = 5.0
COST_MAIN = 0.30
COST_SENS = [0.0, 0.30, 0.50]
FINANCE_ANN = 0.05

# Default params (headline)
DONCH_ENTRY = 20
DONCH_EXIT = 10
ATR_LEN = 20
ATR_STOP_MULT = 2.0
SMA_LEN = 50


@dataclass
class Trade:
    strategy: str
    variant: str
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    stop_px: float
    pnl_oz: float
    r_multiple: float
    risk_oz: float
    reason: str
    bars_held: int
    fold: str = ""


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def session_daily_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """22:00 UTC session daily OHLC."""
    tmp = h1.copy()
    tmp["sess"] = (tmp.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    g = tmp.groupby("sess", sort=True)
    daily = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum() if "volume" in tmp.columns else 0,
    }).dropna(subset=["open", "close"])
    idx = pd.to_datetime(daily.index) + pd.Timedelta(hours=22)
    daily.index = idx.tz_localize("UTC")
    return daily.sort_index()


def prior_daily_sma_map(daily: pd.DataFrame, n: int = 50) -> Dict[str, Tuple[float, float]]:
    d = daily.copy()
    d["sma"] = d["close"].rolling(n, min_periods=n).mean()
    d["pc"] = d["close"].shift(1)
    d["ps"] = d["sma"].shift(1)
    sess = (d.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    out = {}
    for sid, pc, ps in zip(sess, d["pc"].values, d["ps"].values):
        out[sid] = (
            float(pc) if not np.isnan(pc) else np.nan,
            float(ps) if not np.isnan(ps) else np.nan,
        )
    return out


def load_h1() -> pd.DataFrame:
    if not os.path.exists(DATA_H1):
        raise FileNotFoundError(DATA_H1)
    df = pd.read_parquet(DATA_H1)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    # Optionally stitch recent M5->H1 if H1 ends early
    if os.path.exists(DATA_M5):
        m5 = pd.read_parquet(DATA_M5)
        m5.index = pd.to_datetime(m5.index, utc=True)
        m5 = m5.sort_index()
        h1_from_m5 = m5[["open", "high", "low", "close", "volume"]].resample(
            "1h", label="left", closed="left"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
        # fill gaps / extend
        df = pd.concat([df, h1_from_m5]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    df["session_id"] = (df.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    return df


def apply_financing(pnl: float, entry_px: float, entry_t, exit_t, finance: bool) -> float:
    if not finance:
        return pnl
    days = max((exit_t - entry_t).total_seconds() / 86400.0, 0.0)
    return pnl - entry_px * FINANCE_ANN * (days / 365.25)


def backtest_h1_long_sma(
    h1: pd.DataFrame,
    daily_bias: Dict[str, Tuple[float, float]],
    cost: float,
    donch_entry: int = DONCH_ENTRY,
    donch_exit: int = DONCH_EXIT,
    atr_len: int = ATR_LEN,
    atr_mult: float = ATR_STOP_MULT,
    use_sma_filter: bool = True,
    finance: bool = False,
    variant: str = "h1_donch_long_sma50",
    fold: str = "",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> List[Trade]:
    """
    Exact logic (long-only) matching backtest_freq.py:
      Entry: close > prior donch_entry high, next open
      Filter: prior daily close > SMA50 (if enabled)
      Initial stop: entry - atr_mult * prior ATR(atr_len)
      Trail: each bar after entry, stop = max(stop, prev_close - atr_mult * ATR[j-1])
      Also exit: close < prior donch_exit low
      Stop checked before channel exit; same-bar stop first.
    """
    d = h1.copy()
    if start is not None:
        d = d.loc[d.index >= start]
    if end is not None:
        d = d.loc[d.index < end]
    if len(d) < donch_entry + 30:
        return []

    d["hiN"] = d["high"].rolling(donch_entry, min_periods=donch_entry).max().shift(1)
    d["loM"] = d["low"].rolling(donch_exit, min_periods=donch_exit).min().shift(1)
    d["atrN"] = atr(d, atr_len)
    d["patr"] = d["atrN"].shift(1)

    trades: List[Trade] = []
    n = len(d)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    times = d.index
    sids = d["session_id"].values
    hiN, loM = d["hiN"].values, d["loM"].values
    patr, atrv = d["patr"].values, d["atrN"].values

    warmup = max(donch_entry, atr_len, donch_exit) + 5
    i = warmup
    while i < n - 1:
        if np.isnan(hiN[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > hiN[i]):
            i += 1
            continue
        if use_sma_filter:
            pc, ps = daily_bias.get(sids[i], (np.nan, np.nan))
            if np.isnan(pc) or np.isnan(ps) or not (pc > ps):
                i += 1
                continue

        entry_i = i + 1
        entry_px = float(o[entry_i])
        entry_t = times[entry_i]
        stop = entry_px - atr_mult * float(patr[i])
        risk = entry_px - stop
        if risk <= 0.01:
            i += 1
            continue

        exit_i, exit_px, reason = None, None, ""
        cur_stop = stop
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrv[j - 1]) and atrv[j - 1] > 0:
                cur_stop = max(cur_stop, c[j - 1] - atr_mult * atrv[j - 1])
            if l[j] <= cur_stop:
                exit_i, exit_px, reason = j, float(cur_stop), "trail_stop"
                break
            if j > entry_i and not np.isnan(loM[j]) and c[j] < loM[j]:
                exit_i, exit_px, reason = j, float(c[j]), "donch_exit"
                break
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, float(c[n - 1]), "data_end"

        pnl = (exit_px - entry_px) - cost
        pnl = apply_financing(pnl, entry_px, entry_t, times[exit_i], finance)
        trades.append(Trade(
            strategy="H1_DONCH_SMA", variant=variant, side=1,
            entry_time=entry_t, exit_time=times[exit_i],
            entry_px=entry_px, exit_px=float(exit_px), stop_px=float(stop),
            pnl_oz=float(pnl), r_multiple=float(pnl / risk),
            risk_oz=float(risk), reason=reason,
            bars_held=int(exit_i - entry_i + 1), fold=fold,
        ))
        i = exit_i + 1
    return trades


# ---- metrics ----
def sim_equity(trades: List[Trade], initial: float = INITIAL) -> Tuple[pd.Series, float]:
    if not trades:
        return pd.Series(dtype=float), initial
    eq = initial
    rows = []
    for t in trades:
        if eq <= 0:
            break
        size = (eq * RISK_PCT) / max(t.risk_oz, 1e-9)
        size = min(size, (MAX_LEV * eq) / max(t.entry_px, 1e-9))
        eq = eq + size * t.pnl_oz
        rows.append((t.exit_time, eq))
    s = pd.Series({ts: e for ts, e in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")], float(eq)


def max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def sharpe(eq: pd.Series) -> float:
    if eq.empty or len(eq) < 2:
        return float("nan")
    eq = eq.copy()
    eq.index = pd.to_datetime(eq.index, utc=True)
    idx = pd.date_range(eq.index.min().normalize(), eq.index.max().normalize() + pd.Timedelta(days=1), freq="D", tz="UTC")
    d = eq.reindex(idx.union(eq.index)).sort_index().ffill().reindex(idx).dropna()
    r = d.pct_change().dropna()
    if len(r) < 5 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))


def years_span(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return max((b - a).total_seconds() / (365.25 * 24 * 3600), 1e-9)


def exposure_pct(trades: List[Trade], start: pd.Timestamp, end: pd.Timestamp) -> float:
    if not trades or end <= start:
        return 0.0
    total = (end - start).total_seconds()
    held = 0.0
    for t in trades:
        a, b = max(t.entry_time, start), min(t.exit_time, end)
        if b > a:
            held += (b - a).total_seconds()
    return 100.0 * held / total


def summarize(
    trades: List[Trade],
    row_type: str,
    label: str,
    cost: float,
    finance: str,
    span_start: pd.Timestamp,
    span_end: pd.Timestamp,
    extra: Optional[Dict] = None,
) -> Dict:
    n = len(trades)
    yrs = years_span(span_start, span_end)
    base = dict(
        row_type=row_type, label=label, cost=cost, financing=finance,
        n_trades=n,
        trades_per_year=(n / yrs) if n else 0.0,
        win_rate=np.nan, avg_R=np.nan, avg_pnl_oz=np.nan, profit_factor=np.nan,
        total_return_pct=0.0, cagr=np.nan, max_dd=0.0, sharpe_daily=np.nan,
        exposure_pct=0.0, avg_hold_hours=np.nan,
        period_start=str(span_start), period_end=str(span_end),
    )
    if extra:
        base.update(extra)
    if n == 0:
        return base
    wins = [t for t in trades if t.pnl_oz > 0]
    losses = [t for t in trades if t.pnl_oz <= 0]
    gp = sum(t.pnl_oz for t in wins)
    gl = abs(sum(t.pnl_oz for t in losses))
    pf = gp / gl if gl > 1e-12 else (999.0 if gp > 0 else np.nan)
    eq, final = sim_equity(trades)
    hold_h = float(np.mean([t.bars_held for t in trades]))  # H1 bars ~= hours
    cagr = (final / INITIAL) ** (1 / yrs) - 1 if final > 0 else float("nan")
    base.update(
        win_rate=len(wins) / n,
        avg_R=float(np.mean([t.r_multiple for t in trades])),
        avg_pnl_oz=float(np.mean([t.pnl_oz for t in trades])),
        profit_factor=float(pf) if np.isfinite(pf) else np.nan,
        total_return_pct=(final / INITIAL - 1) * 100,
        cagr=float(cagr) if np.isfinite(cagr) else np.nan,
        max_dd=max_dd(eq),
        sharpe_daily=sharpe(eq),
        exposure_pct=exposure_pct(trades, span_start, span_end),
        avg_hold_hours=hold_h,
    )
    return base


def buyhold(daily: pd.DataFrame, cost: float, label: str = "BuyHold") -> Dict:
    px0, px1 = float(daily["close"].iloc[0]), float(daily["close"].iloc[-1])
    rets = daily["close"].pct_change().dropna()
    sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    peak = daily["close"].cummax()
    dd = float(((daily["close"] - peak) / peak).min())
    yrs = years_span(daily.index[0], daily.index[-1])
    cagr = (px1 / px0) ** (1 / yrs) - 1
    return dict(
        row_type="buyhold", label=label, cost=cost, financing="none",
        n_trades=1, trades_per_year=0.0, win_rate=1.0, avg_R=np.nan,
        avg_pnl_oz=(px1 - px0) - cost, profit_factor=np.nan,
        total_return_pct=(px1 / px0 - 1) * 100, cagr=float(cagr),
        max_dd=dd, sharpe_daily=sh, exposure_pct=100.0, avg_hold_hours=np.nan,
        period_start=str(daily.index[0]), period_end=str(daily.index[-1]),
        start_px=px0, end_px=px1,
    )


def drawdown_periods(eq: pd.Series, min_dd: float = -0.05) -> List[Dict]:
    """Find drawdown episodes where DD <= min_dd."""
    if eq.empty:
        return []
    peak = eq.cummax()
    dd = (eq - peak) / peak
    periods = []
    in_dd = False
    start = None
    peak_eq = None
    trough = None
    trough_t = None
    for t, v in dd.items():
        if not in_dd and v < 0:
            in_dd = True
            start = t
            peak_eq = peak.loc[t]
            trough, trough_t = v, t
        elif in_dd:
            if v < trough:
                trough, trough_t = v, t
            if v >= -1e-12:  # recovered
                if trough <= min_dd:
                    periods.append(dict(
                        start=str(start), trough_time=str(trough_t),
                        end=str(t), depth=float(trough),
                    ))
                in_dd = False
    if in_dd and trough is not None and trough <= min_dd:
        periods.append(dict(
            start=str(start), trough_time=str(trough_t),
            end=str(eq.index[-1]), depth=float(trough),
        ))
    return periods


def rolling_folds(start: pd.Timestamp, end: pd.Timestamp, train: pd.DateOffset, test: pd.DateOffset, step: pd.DateOffset):
    """Yield (train_start, train_end, test_start, test_end) with optional 7d embargo."""
    embargo = pd.Timedelta(days=7)
    t0 = start
    folds = []
    while True:
        train_start = t0
        train_end = train_start + train
        test_start = train_end + embargo
        test_end = test_start + test
        if test_end > end + pd.Timedelta(days=1):
            break
        if train_end <= start or test_start >= end:
            t0 = t0 + step
            continue
        folds.append((train_start, train_end, test_start, min(test_end, end)))
        t0 = t0 + step
        if t0 + train + embargo + test > end + pd.Timedelta(days=32):
            # allow last partial? skip incomplete
            if test_end > end and len(folds) > 0:
                break
    return folds


def run_all():
    print("Loading H1...")
    h1 = load_h1()
    print(f"H1 bars={len(h1)} | {h1.index.min()} -> {h1.index.max()} UTC")
    print(f"Source: Dukascopy XAU/USD BID INTERVAL_HOUR_1 (+ M5 stitch if newer)")
    daily = session_daily_from_h1(h1)
    print(f"Session daily (22:00 UTC) bars={len(daily)} | {daily.index.min()} -> {daily.index.max()}")
    bias = prior_daily_sma_map(daily, SMA_LEN)

    span_start, span_end = h1.index[0], h1.index[-1]
    # Need SMA50 warmup: start usable after ~50 session days
    usable_start = daily.index[min(60, len(daily) - 1)]
    print(f"Usable after SMA warmup ~ {usable_start}")

    results = []
    curves = {}

    # ===== Full sample headline =====
    for cost in COST_SENS:
        for finance in [False, True]:
            fin = "fin5pct" if finance else "nofin"
            trades = backtest_h1_long_sma(
                h1, bias, cost, use_sma_filter=True, finance=finance,
                variant=f"headline_sma50_{fin}", fold="full",
            )
            row = summarize(trades, "full", f"headline_sma50_{fin}", cost, fin, span_start, span_end)
            results.append(row)
            print(
                f"FULL sma50 {fin} c={cost}: n={row['n_trades']} ({row['trades_per_year']:.1f}/y) "
                f"WR={row['win_rate']:.1%} avgR={row['avg_R']:.3f} PF={row['profit_factor']:.2f} "
                f"ret={row['total_return_pct']:.1f}% CAGR={row['cagr']:.1%} DD={row['max_dd']:.1%} "
                f"Sharpe={row['sharpe_daily']:.2f} exp={row['exposure_pct']:.1f}% hold={row['avg_hold_hours']:.1f}h"
            )
            if cost == COST_MAIN and not finance:
                eq, _ = sim_equity(trades)
                curves["headline_sma50"] = eq
                # save trades
                pd.DataFrame([asdict(t) for t in trades]).to_csv(TRADES_CSV, index=False)
                print(f"Wrote trades {TRADES_CSV} n={len(trades)}")
                # drawdowns
                dds = drawdown_periods(eq, -0.05)
                for dd in dds[:15]:
                    results.append(dict(
                        row_type="drawdown", label="headline_dd", cost=cost, financing=fin,
                        n_trades=0, trades_per_year=0, win_rate=np.nan, avg_R=np.nan,
                        avg_pnl_oz=np.nan, profit_factor=np.nan, total_return_pct=np.nan,
                        cagr=np.nan, max_dd=dd["depth"], sharpe_daily=np.nan,
                        exposure_pct=np.nan, avg_hold_hours=np.nan,
                        period_start=dd["start"], period_end=dd["end"],
                        trough_time=dd["trough_time"],
                    ))

    # Unfiltered comparison
    for cost in [COST_MAIN]:
        trades_u = backtest_h1_long_sma(
            h1, bias, cost, use_sma_filter=False, finance=False,
            variant="h1_donch_long_nofilter", fold="full",
        )
        row = summarize(trades_u, "full", "h1_donch_long_nofilter", cost, "nofin", span_start, span_end)
        results.append(row)
        eq_u, _ = sim_equity(trades_u)
        curves["nofilter"] = eq_u
        print(f"FULL nofilter c={cost}: n={row['n_trades']} ret={row['total_return_pct']:.1f}% DD={row['max_dd']:.1%} Sharpe={row['sharpe_daily']:.2f}")

    # Buy & hold
    bh = buyhold(daily.loc[daily.index >= usable_start], COST_MAIN)
    results.append(bh)
    # BH equity curve (mark-to-market 1oz scaled to start 10k notionally via price ratio)
    d_bh = daily.loc[daily.index >= usable_start]
    curves["buyhold"] = INITIAL * (d_bh["close"] / float(d_bh["close"].iloc[0]))
    print(f"BH: {bh.get('start_px',0):.1f}->{bh.get('end_px',0):.1f} ret={bh['total_return_pct']:.1f}% DD={bh['max_dd']:.1%}")

    # ===== 70/30 IS/OOS =====
    split_i = int(len(h1) * 0.70)
    split_t = h1.index[split_i]
    trades_full = backtest_h1_long_sma(h1, bias, COST_MAIN, use_sma_filter=True, finance=False, variant="headline_sma50_nofin", fold="full")
    is_tr = [t for t in trades_full if t.entry_time < split_t]
    oos_tr = [t for t in trades_full if t.entry_time >= split_t]
    results.append(summarize(is_tr, "is_oos", "IS_70", COST_MAIN, "nofin", span_start, split_t))
    results.append(summarize(oos_tr, "is_oos", "OOS_30", COST_MAIN, "nofin", split_t, span_end))
    print(f"IS70: n={len(is_tr)} ret={results[-2]['total_return_pct']:.1f}% | OOS30: n={len(oos_tr)} ret={results[-1]['total_return_pct']:.1f}%")

    # ===== Walk-forward rolling =====
    # Scheme A: train 3y / test 1y / step 1y
    # Scheme B: train 2y / test 6m / step 6m
    schemes = [
        ("WF_3y1y", pd.DateOffset(years=3), pd.DateOffset(years=1), pd.DateOffset(years=1)),
        ("WF_2y6m", pd.DateOffset(years=2), pd.DateOffset(months=6), pd.DateOffset(months=6)),
    ]
    oos_concat_trades = []  # from primary scheme
    for scheme_name, train_off, test_off, step_off in schemes:
        folds = rolling_folds(usable_start, span_end, train_off, test_off, step_off)
        print(f"\n{scheme_name}: {len(folds)} folds")
        fold_oos_trades = []
        for fi, (tr_s, tr_e, te_s, te_e) in enumerate(folds):
            # FIXED params — we still "train" only to define window; no opt
            # Optional: report train metrics for reference
            tr_trades = backtest_h1_long_sma(
                h1, bias, COST_MAIN, use_sma_filter=True, finance=False,
                variant=f"{scheme_name}_train", fold=f"{scheme_name}_F{fi}_train",
                start=tr_s, end=tr_e,
            )
            te_trades = backtest_h1_long_sma(
                h1, bias, COST_MAIN, use_sma_filter=True, finance=False,
                variant=f"{scheme_name}_test", fold=f"{scheme_name}_F{fi}_test",
                start=te_s, end=te_e,
            )
            # Only count trades entered in test window (already filtered by start/end)
            results.append(summarize(
                tr_trades, "wf_train", f"{scheme_name}_F{fi}", COST_MAIN, "nofin", tr_s, tr_e,
                extra=dict(fold=fi, train_start=str(tr_s), train_end=str(tr_e),
                           test_start=str(te_s), test_end=str(te_e)),
            ))
            te_row = summarize(
                te_trades, "wf_test", f"{scheme_name}_F{fi}", COST_MAIN, "nofin", te_s, te_e,
                extra=dict(fold=fi, train_start=str(tr_s), train_end=str(tr_e),
                           test_start=str(te_s), test_end=str(te_e)),
            )
            results.append(te_row)
            print(
                f"  F{fi} TEST {te_s.date()}->{te_e.date()}: n={te_row['n_trades']} "
                f"WR={te_row['win_rate']:.1%} avgR={te_row['avg_R']:.3f} PF={te_row['profit_factor']:.2f} "
                f"ret={te_row['total_return_pct']:.1f}% DD={te_row['max_dd']:.1%} Sharpe={te_row['sharpe_daily']:.2f}"
            )
            fold_oos_trades.extend(te_trades)
        # Concatenated OOS equity for this scheme
        if fold_oos_trades:
            # sort by entry
            fold_oos_trades = sorted(fold_oos_trades, key=lambda t: t.entry_time)
            # dedupe overlapping if any (shouldn't with embargo)
            oos_eq, oos_final = sim_equity(fold_oos_trades)
            results.append(summarize(
                fold_oos_trades, "wf_oos_concat", f"{scheme_name}_OOS_concat",
                COST_MAIN, "nofin",
                fold_oos_trades[0].entry_time, fold_oos_trades[-1].exit_time,
            ))
            curves[f"{scheme_name}_oos"] = oos_eq
            print(f"  {scheme_name} OOS concat: n={len(fold_oos_trades)} ret={results[-1]['total_return_pct']:.1f}% DD={results[-1]['max_dd']:.1%}")
            if scheme_name == "WF_3y1y":
                oos_concat_trades = fold_oos_trades

    # ===== Calendar year-by-year =====
    years = sorted(set(h1.index.year))
    for y in years:
        y_start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        y_end = pd.Timestamp(f"{y+1}-01-01", tz="UTC")
        y_trades = backtest_h1_long_sma(
            h1, bias, COST_MAIN, use_sma_filter=True, finance=False,
            variant="yearly", fold=f"Y{y}", start=y_start, end=y_end,
        )
        # Also BH for year
        d_y = daily.loc[(daily.index >= y_start) & (daily.index < y_end)]
        row = summarize(y_trades, "yearly", f"Y{y}", COST_MAIN, "nofin", y_start, min(y_end, span_end))
        if len(d_y) >= 2:
            bh_y = (float(d_y["close"].iloc[-1]) / float(d_y["close"].iloc[0]) - 1) * 100
            row["bh_return_pct"] = bh_y
        results.append(row)
        print(f"YEAR {y}: n={row['n_trades']} ret={row['total_return_pct']:.1f}% DD={row['max_dd']:.1%} BH={row.get('bh_return_pct', float('nan')):.1f}%")

    # ===== Sensitivity grid (report only; headline stays default) =====
    print("\nSensitivity grid (not used for headline)...")
    for de in [15, 20, 25]:
        for am in [1.5, 2.0, 2.5]:
            trades_s = backtest_h1_long_sma(
                h1, bias, COST_MAIN, donch_entry=de, donch_exit=max(5, de // 2),
                atr_len=20, atr_mult=am, use_sma_filter=True, finance=False,
                variant=f"sens_d{de}_atr{am}", fold="sens",
            )
            row = summarize(trades_s, "sensitivity", f"d{de}_atr{am}", COST_MAIN, "nofin", span_start, span_end,
                            extra=dict(donch_entry=de, atr_mult=am, donch_exit=max(5, de // 2)))
            results.append(row)
            print(f"  d{de} atr{am}: n={row['n_trades']} avgR={row['avg_R']:.3f} PF={row['profit_factor']:.2f} ret={row['total_return_pct']:.1f}% DD={row['max_dd']:.1%}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"Wrote {RESULTS_CSV}")

    # ===== Chart =====
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    ax = axes[0]
    for k, label in [("headline_sma50", "H1 Donch+SMA50"), ("nofilter", "H1 Donch no filter"), ("buyhold", "Buy&Hold")]:
        eq = curves.get(k)
        if eq is not None and not eq.empty:
            ax.plot(eq.index.tz_convert("Asia/Shanghai"), eq.values, label=label, lw=1.2)
    ax.axhline(INITIAL, color="gray", ls="--", lw=0.8)
    ax.set_title("Full-sample equity (10k, 1% risk, cost 0.30 USD/oz)")
    ax.set_ylabel("Equity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for k, label in [("WF_3y1y_oos", "WF 3y/1y OOS concat"), ("WF_2y6m_oos", "WF 2y/6m OOS concat")]:
        eq = curves.get(k)
        if eq is not None and not eq.empty:
            ax2.plot(eq.index.tz_convert("Asia/Shanghai"), eq.values, label=label, lw=1.2)
    ax2.axhline(INITIAL, color="gray", ls="--", lw=0.8)
    ax2.set_title("Walk-forward concatenated OOS equity (fixed params)")
    ax2.set_ylabel("Equity")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    # yearly returns bar
    yearly = res_df[res_df.row_type == "yearly"].copy()
    if len(yearly):
        years_l = [int(x[1:]) for x in yearly["label"]]
        ax3.bar(years_l, yearly["total_return_pct"].values, color="steelblue", alpha=0.8, label="Strategy")
        if "bh_return_pct" in yearly.columns:
            ax3.plot(years_l, yearly["bh_return_pct"].values, "o-", color="orange", label="BuyHold")
        ax3.axhline(0, color="gray", lw=0.8)
        ax3.set_title("Calendar year returns (strategy vs buy-hold)")
        ax3.set_ylabel("Return %")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(EQUITY_PNG, dpi=140)
    print(f"Wrote {EQUITY_PNG}")

    # Print key tables
    print("\n=== FULL / IS-OOS / WF concat ===")
    show = res_df[res_df.row_type.isin(["full", "is_oos", "wf_oos_concat", "buyhold"])]
    cols = ["row_type", "label", "cost", "financing", "n_trades", "trades_per_year",
            "win_rate", "avg_R", "profit_factor", "total_return_pct", "cagr", "max_dd",
            "sharpe_daily", "exposure_pct", "avg_hold_hours"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== WF_3y1y TEST folds ===")
    wf = res_df[(res_df.row_type == "wf_test") & (res_df.label.str.startswith("WF_3y1y"))]
    print(wf[["label", "period_start", "period_end", "n_trades", "win_rate", "avg_R",
              "profit_factor", "total_return_pct", "max_dd", "sharpe_daily"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Yearly ===")
    print(yearly[["label", "n_trades", "total_return_pct", "max_dd", "bh_return_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Sensitivity ===")
    sens = res_df[res_df.row_type == "sensitivity"]
    print(sens[["label", "n_trades", "avg_R", "profit_factor", "total_return_pct", "max_dd", "sharpe_daily"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return res_df


if __name__ == "__main__":
    run_all()

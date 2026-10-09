#!/usr/bin/env python3
"""
Cross-data-source robustness: Dukascopy vs HistData XAU/USD.
Fixed-param strategies on both feeds; price-diff diagnostics.
"""
from __future__ import annotations

import os, warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = "/workspace/backtest_gold"
DUKA_H1 = os.path.join(ROOT, "xauusd_h1_2010_2026.parquet")
HIST_H1 = os.path.join(ROOT, "xauusd_h1_histdata_2010_2026.parquet")
RESULTS = os.path.join(ROOT, "results_cross_source.csv")
EQUITY = os.path.join(ROOT, "equity_cross_source.png")
DIFF_CSV = os.path.join(ROOT, "price_diff_daily_cross_source.csv")

INITIAL = 10_000.0
RISK_PCT = 0.01
MAX_LEV = 5.0
COST = 0.30


@dataclass
class Trade:
    strategy: str
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


def load_h1(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    need = ["open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            raise ValueError(path)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["session_id"] = (df.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df[["open", "high", "low", "close", "volume"]].resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "close"])
    out["session_id"] = (out.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    return out


def session_daily(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["sess"] = (tmp.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    g = tmp.groupby("sess", sort=True)
    d = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna(subset=["open", "close"])
    d.index = (pd.to_datetime(d.index) + pd.Timedelta(hours=22)).tz_localize("UTC")
    return d.sort_index()


def atr(df, n):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def sma_map(daily: pd.DataFrame, n: int = 50) -> Dict[str, Tuple[float, float]]:
    d = daily.copy()
    d["sma"] = d["close"].rolling(n, min_periods=n).mean()
    d["pc"] = d["close"].shift(1)
    d["ps"] = d["sma"].shift(1)
    sess = (d.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    return {
        sid: (float(pc) if not np.isnan(pc) else np.nan, float(ps) if not np.isnan(ps) else np.nan)
        for sid, pc, ps in zip(sess, d["pc"].values, d["ps"].values)
    }


def sim_eq(trades: List[Trade], initial=INITIAL):
    if not trades:
        return pd.Series(dtype=float), initial
    eq = initial
    rows = []
    for t in trades:
        if eq <= 0:
            break
        size = min((eq * RISK_PCT) / max(t.risk_oz, 1e-9), (MAX_LEV * eq) / max(t.entry_px, 1e-9))
        eq = eq + size * t.pnl_oz
        rows.append((t.exit_time, eq))
    s = pd.Series({a: b for a, b in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")], float(eq)


def max_dd(eq):
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    return float(((eq - peak) / peak).min())


def sharpe(eq):
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


def yrs(a, b):
    return max((b - a).total_seconds() / (365.25 * 24 * 3600), 1e-9)


def summarize(trades, name, source, span_start, span_end) -> Dict:
    n = len(trades)
    y = yrs(span_start, span_end)
    base = dict(
        strategy=name, source=source, n_trades=n, trades_per_year=n / y if n else 0.0,
        win_rate=np.nan, avg_R=np.nan, profit_factor=np.nan,
        total_return_pct=0.0, cagr=np.nan, max_dd=0.0, sharpe_daily=np.nan,
        period_start=str(span_start.date()), period_end=str(span_end.date()),
    )
    if n == 0:
        return base
    wins = [t for t in trades if t.pnl_oz > 0]
    losses = [t for t in trades if t.pnl_oz <= 0]
    gp = sum(t.pnl_oz for t in wins)
    gl = abs(sum(t.pnl_oz for t in losses))
    pf = gp / gl if gl > 1e-12 else (999.0 if gp > 0 else np.nan)
    eq, final = sim_eq(trades)
    cagr = (final / INITIAL) ** (1 / y) - 1 if final > 0 else np.nan
    base.update(
        win_rate=len(wins) / n,
        avg_R=float(np.mean([t.r_multiple for t in trades])),
        profit_factor=float(pf) if np.isfinite(pf) else np.nan,
        total_return_pct=(final / INITIAL - 1) * 100,
        cagr=float(cagr) if np.isfinite(cagr) else np.nan,
        max_dd=max_dd(eq),
        sharpe_daily=sharpe(eq),
    )
    return base


# ---- strategies ----
def bt_donchian(bars, bias, cost=COST, entry_n=20, exit_n=10, atr_n=20, atr_mult=2.0,
                use_atr_trail=True, name="donch"):
    d = bars.copy()
    d["hiE"] = d["high"].rolling(entry_n, min_periods=entry_n).max().shift(1)
    d["loX"] = d["low"].rolling(exit_n, min_periods=exit_n).min().shift(1)
    d["atr"] = atr(d, atr_n)
    d["patr"] = d["atr"].shift(1)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    times = d.index
    sids = d["session_id"].values
    hiE, loX = d["hiE"].values, d["loX"].values
    patr, atrv = d["patr"].values, d["atr"].values
    trades = []
    n = len(d)
    i = max(entry_n, atr_n, exit_n) + 5
    while i < n - 1:
        if np.isnan(hiE[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > hiE[i]):
            i += 1
            continue
        pc, ps = bias.get(sids[i], (np.nan, np.nan))
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
        exit_i = exit_px = reason = None
        cur = stop
        for j in range(entry_i, n):
            if use_atr_trail and j > entry_i and not np.isnan(atrv[j - 1]) and atrv[j - 1] > 0:
                cur = max(cur, c[j - 1] - atr_mult * atrv[j - 1])
            if l[j] <= cur:
                exit_i, exit_px, reason = j, float(cur), "stop"
                break
            if j > entry_i and not np.isnan(loX[j]) and c[j] < loX[j]:
                exit_i, exit_px, reason = j, float(c[j]), "donch_exit"
                break
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, float(c[-1]), "end"
        pnl = (exit_px - entry_px) - cost
        trades.append(Trade(name, 1, entry_t, times[exit_i], entry_px, float(exit_px), float(stop),
                            float(pnl), float(pnl / risk), float(risk), reason, int(exit_i - entry_i + 1)))
        i = exit_i + 1
    return trades


def bt_atr_channel(bars, bias, cost=COST, ma_n=20, atr_n=20, k=2.0, atr_mult=2.0, name="atrchan"):
    d = bars.copy()
    d["ma"] = d["close"].rolling(ma_n, min_periods=ma_n).mean()
    d["atr"] = atr(d, atr_n)
    d["up"] = (d["ma"] + k * d["atr"]).shift(1)
    d["patr"] = d["atr"].shift(1)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    times = d.index
    sids = d["session_id"].values
    up, patr = d["up"].values, d["patr"].values
    ma, atrv = d["ma"].values, d["atr"].values
    trades = []
    n = len(d)
    i = max(ma_n, atr_n) + 5
    while i < n - 1:
        if np.isnan(up[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > up[i]):
            i += 1
            continue
        pc, ps = bias.get(sids[i], (np.nan, np.nan))
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
        exit_i = exit_px = reason = None
        cur = stop
        for j in range(entry_i, n):
            if j > entry_i and not np.isnan(atrv[j - 1]) and atrv[j - 1] > 0:
                cur = max(cur, c[j - 1] - atr_mult * atrv[j - 1])
            if l[j] <= cur:
                exit_i, exit_px, reason = j, float(cur), "trail"
                break
            if j > entry_i and not np.isnan(ma[j]) and c[j] < ma[j]:
                exit_i, exit_px, reason = j, float(c[j]), "ma_exit"
                break
        if exit_i is None:
            exit_i, exit_px, reason = n - 1, float(c[-1]), "end"
        pnl = (exit_px - entry_px) - cost
        trades.append(Trade(name, 1, entry_t, times[exit_i], entry_px, float(exit_px), float(stop),
                            float(pnl), float(pnl / risk), float(risk), reason, int(exit_i - entry_i + 1)))
        i = exit_i + 1
    return trades


def bt_bb(bars, bias, cost=COST, n=20, k=2.0, atr_n=20, atr_mult=2.0, name="bb"):
    d = bars.copy()
    mid = d["close"].rolling(n, min_periods=n).mean()
    std = d["close"].rolling(n, min_periods=n).std(ddof=0)
    d["up"] = (mid + k * std).shift(1)
    d["mid"] = mid
    d["atr"] = atr(d, atr_n)
    d["patr"] = d["atr"].shift(1)
    o, h, l, c = d["open"].values, d["high"].values, d["low"].values, d["close"].values
    times = d.index
    sids = d["session_id"].values
    up, midv = d["up"].values, d["mid"].values
    patr, atrv = d["patr"].values, d["atr"].values
    trades = []
    nbar = len(d)
    i = n + 5
    while i < nbar - 1:
        if np.isnan(up[i]) or np.isnan(patr[i]) or patr[i] <= 0:
            i += 1
            continue
        if not (c[i] > up[i]):
            i += 1
            continue
        pc, ps = bias.get(sids[i], (np.nan, np.nan))
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
        exit_i = exit_px = reason = None
        cur = stop
        for j in range(entry_i, nbar):
            if j > entry_i and not np.isnan(atrv[j - 1]) and atrv[j - 1] > 0:
                cur = max(cur, c[j - 1] - atr_mult * atrv[j - 1])
            if l[j] <= cur:
                exit_i, exit_px, reason = j, float(cur), "trail"
                break
            if j > entry_i and not np.isnan(midv[j]) and c[j] < midv[j]:
                exit_i, exit_px, reason = j, float(c[j]), "mid_exit"
                break
        if exit_i is None:
            exit_i, exit_px, reason = nbar - 1, float(c[-1]), "end"
        pnl = (exit_px - entry_px) - cost
        trades.append(Trade(name, 1, entry_t, times[exit_i], entry_px, float(exit_px), float(stop),
                            float(pnl), float(pnl / risk), float(risk), reason, int(exit_i - entry_i + 1)))
        i = exit_i + 1
    return trades


def compare_prices(duka_h1: pd.DataFrame, hist_h1: pd.DataFrame) -> pd.DataFrame:
    common = duka_h1.index.intersection(hist_h1.index)
    d = duka_h1.loc[common]
    h = hist_h1.loc[common]
    h1_diff = pd.DataFrame({
        "d_close": d["close"], "h_close": h["close"],
        "abs_close": (d["close"] - h["close"]).abs(),
        "abs_high": (d["high"] - h["high"]).abs(),
        "abs_low": (d["low"] - h["low"]).abs(),
    }, index=common)
    # daily session
    dd = session_daily(duka_h1)
    hd = session_daily(hist_h1)
    # align by calendar date of session_id
    dd2 = dd.copy()
    hd2 = hd.copy()
    dd2["date"] = (dd2.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    hd2["date"] = (hd2.index - pd.Timedelta(hours=22)).strftime("%Y-%m-%d")
    m = dd2.reset_index().merge(hd2.reset_index(), on="date", suffixes=("_duka", "_hist"))
    daily_diff = pd.DataFrame({
        "date": m["date"],
        "abs_high": (m["high_duka"] - m["high_hist"]).abs(),
        "abs_low": (m["low_duka"] - m["low_hist"]).abs(),
        "abs_close": (m["close_duka"] - m["close_hist"]).abs(),
        "close_duka": m["close_duka"],
        "close_hist": m["close_hist"],
    })
    return h1_diff, daily_diff


def run_all():
    print("=" * 60)
    print("SOURCE 1: Dukascopy XAU/USD BID H1")
    print(f"  file: {DUKA_H1}")
    duka = load_h1(DUKA_H1)
    print(f"  bars={len(duka)} | {duka.index.min()} -> {duka.index.max()} UTC")

    print("SOURCE 2: HistData.com Generic ASCII XAUUSD M1 (BID quotes) resampled to H1")
    print(f"  file: {HIST_H1}")
    print("  note: HistData timestamps treated as America/New_York (DST-aware) -> UTC")
    print("        (empirically aligns with Dukascopy; FAQ claims EST no-DST but median error much larger)")
    hist = load_h1(HIST_H1)
    print(f"  bars={len(hist)} | {hist.index.min()} -> {hist.index.max()} UTC")

    # Overlap window
    start = max(duka.index.min(), hist.index.min())
    end = min(duka.index.max(), hist.index.max())
    duka_o = duka.loc[(duka.index >= start) & (duka.index <= end)]
    hist_o = hist.loc[(hist.index >= start) & (hist.index <= end)]
    print(f"OVERLAP: {start} -> {end}")
    print(f"  Duka bars={len(duka_o)} Hist bars={len(hist_o)}")

    h1_diff, daily_diff = compare_prices(duka_o, hist_o)
    print("\nH1 close abs diff (overlapping hours):")
    for q in [0.5, 0.9, 0.95, 0.99]:
        print(f"  p{int(q*100)}: {h1_diff['abs_close'].quantile(q):.4f}")
    print(f"  max: {h1_diff['abs_close'].max():.4f} at {h1_diff['abs_close'].idxmax()}")
    print("Daily H/L/C abs diff:")
    for col in ["abs_high", "abs_low", "abs_close"]:
        print(f"  {col}: median={daily_diff[col].median():.4f} p95={daily_diff[col].quantile(0.95):.4f} max={daily_diff[col].max():.4f}")

    # Flag extreme days
    daily_diff["max_hlc"] = daily_diff[["abs_high", "abs_low", "abs_close"]].max(axis=1)
    extreme = daily_diff.nlargest(15, "max_hlc")
    print("\nTop 15 extreme daily divergence days:")
    for _, r in extreme.iterrows():
        print(f"  {r['date']}: maxHLCdiff={r['max_hlc']:.2f} H={r['abs_high']:.2f} L={r['abs_low']:.2f} C={r['abs_close']:.2f}")

    # Brexit
    brexit = daily_diff[daily_diff["date"] == "2016-06-23"]
    if len(brexit):
        r = brexit.iloc[0]
        print(f"\nBrexit 2016-06-23: Hdiff={r['abs_high']:.2f} Ldiff={r['abs_low']:.2f} Cdiff={r['abs_close']:.2f}")
    daily_diff.to_csv(DIFF_CSV, index=False)
    print(f"Wrote {DIFF_CSV}")

    results = []
    curves = {}

    def run_source(tag, h1):
        daily = session_daily(h1)
        bias = sma_map(daily, 50)
        h2 = resample(h1, "2h")
        ss, se = h1.index[0], h1.index[-1]
        specs = [
            ("H1_donch20_sma50", lambda: bt_donchian(h1, bias, name="H1_donch20_sma50")),
            ("H2_donch20_sma50", lambda: bt_donchian(h2, bias, name="H2_donch20_sma50")),
            ("H2_atrchan_sma50", lambda: bt_atr_channel(h2, bias, name="H2_atrchan_sma50")),
            ("H1_bb20_2_sma50", lambda: bt_bb(h1, bias, name="H1_bb20_2_sma50")),
            ("H1_donch20_noATR_trail", lambda: bt_donchian(h1, bias, use_atr_trail=False, name="H1_donch20_noATR_trail")),
            ("H2_donch20_noATR_trail", lambda: bt_donchian(h2, bias, use_atr_trail=False, name="H2_donch20_noATR_trail")),
        ]
        for name, fn in specs:
            trades = fn()
            row = summarize(trades, name, tag, ss, se)
            results.append(row)
            eq, _ = sim_eq(trades)
            curves[f"{tag}__{name}"] = eq
            print(f"  [{tag}] {name}: n={row['n_trades']} ret={row['total_return_pct']:.1f}% "
                  f"DD={row['max_dd']:.1%} Sh={row['sharpe_daily']:.2f} PF={row['profit_factor']:.2f}")

    print("\n--- Overlap-window backtests ---")
    run_source("Duka_overlap", duka_o)
    run_source("Hist_overlap", hist_o)

    # Also Dukascopy full sample for reference
    print("\n--- Dukascopy FULL sample (for reference) ---")
    run_source("Duka_full", duka)

    # BuyHold both on overlap
    for tag, h1 in [("Duka_overlap", duka_o), ("Hist_overlap", hist_o)]:
        daily = session_daily(h1)
        px0, px1 = float(daily["close"].iloc[0]), float(daily["close"].iloc[-1])
        rets = daily["close"].pct_change().dropna()
        sh = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
        peak = daily["close"].cummax()
        dd = float(((daily["close"] - peak) / peak).min())
        y = yrs(daily.index[0], daily.index[-1])
        results.append(dict(
            strategy="BuyHold", source=tag, n_trades=1, trades_per_year=0,
            win_rate=1.0, avg_R=np.nan, profit_factor=np.nan,
            total_return_pct=(px1 / px0 - 1) * 100,
            cagr=(px1 / px0) ** (1 / y) - 1,
            max_dd=dd, sharpe_daily=sh,
            period_start=str(daily.index[0].date()), period_end=str(daily.index[-1].date()),
        ))
        curves[f"{tag}__BuyHold"] = INITIAL * (daily["close"] / px0)

    res = pd.DataFrame(results)
    res.to_csv(RESULTS, index=False)
    print(f"\nWrote {RESULTS}")

    # Side-by-side pivot
    ov = res[res.source.isin(["Duka_overlap", "Hist_overlap"])].copy()
    print("\n=== SIDE-BY-SIDE (overlap window) ===")
    piv = ov.pivot_table(index="strategy", columns="source",
                         values=["total_return_pct", "max_dd", "sharpe_daily", "profit_factor", "n_trades"],
                         aggfunc="first")
    # flatten
    print(ov.sort_values(["strategy", "source"])[
        ["strategy", "source", "n_trades", "total_return_pct", "max_dd", "sharpe_daily", "profit_factor"]
    ].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Survival check
    print("\n=== EDGE SURVIVAL ===")
    for strat in ["H1_donch20_sma50", "H2_donch20_sma50", "H2_atrchan_sma50", "H1_bb20_2_sma50",
                  "H1_donch20_noATR_trail", "H2_donch20_noATR_trail"]:
        drow = ov[(ov.strategy == strat) & (ov.source == "Duka_overlap")]
        hrow = ov[(ov.strategy == strat) & (ov.source == "Hist_overlap")]
        if len(drow) == 0 or len(hrow) == 0:
            continue
        d, h = drow.iloc[0], hrow.iloc[0]
        both_pos = d["total_return_pct"] > 0 and h["total_return_pct"] > 0
        both_pf = d["profit_factor"] > 1.1 and h["profit_factor"] > 1.1
        print(f"  {strat}: Duka ret={d['total_return_pct']:.1f}% Sh={d['sharpe_daily']:.2f} | "
              f"Hist ret={h['total_return_pct']:.1f}% Sh={h['sharpe_daily']:.2f} | "
              f"both_pos={both_pos} both_PF>1.1={both_pf}")

    # Chart
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    for name in ["H1_donch20_sma50", "H2_donch20_sma50", "H2_atrchan_sma50", "H1_bb20_2_sma50"]:
        for src, ls in [("Duka_overlap", "-"), ("Hist_overlap", "--")]:
            key = f"{src}__{name}"
            eq = curves.get(key)
            if eq is not None and len(eq):
                ax.plot(eq.index.tz_convert("Asia/Shanghai"), eq.values,
                        label=f"{src[:4]} {name}", ls=ls, lw=1.1)
    ax.axhline(INITIAL, color="gray", ls=":", lw=0.8)
    ax.set_title("Cross-source equity (overlap window, cost 0.30)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Equity")

    ax2 = axes[1]
    for name in ["H1_donch20_sma50", "H1_donch20_noATR_trail", "H2_donch20_sma50", "H2_donch20_noATR_trail"]:
        for src, ls in [("Duka_overlap", "-"), ("Hist_overlap", "--")]:
            key = f"{src}__{name}"
            eq = curves.get(key)
            if eq is not None and len(eq):
                ax2.plot(eq.index.tz_convert("Asia/Shanghai"), eq.values,
                         label=f"{src[:4]} {name}", ls=ls, lw=1.0)
    ax2.axhline(INITIAL, color="gray", ls=":", lw=0.8)
    ax2.set_title("ATR-trail vs Donchian-only exit (cross-source)")
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("Time (Beijing)")
    ax2.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(EQUITY, dpi=140)
    print(f"Wrote {EQUITY}")
    return res


if __name__ == "__main__":
    run_all()

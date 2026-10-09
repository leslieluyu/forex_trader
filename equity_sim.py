"""Shared compounding equity-curve simulator, matching the methodology used
in research/backtest_h1_sma50_wf.py (1% equity risk per trade, 5x leverage
cap), so every strategy in this project is judged by the same yardstick.
"""
import numpy as np
import pandas as pd

INITIAL = 10_000.0
RISK_PCT = 0.01
MAX_LEV = 5.0


def sim_equity(trades: pd.DataFrame, initial=INITIAL, risk_pct=RISK_PCT, max_lev=MAX_LEV):
    """trades needs columns: exit_time, entry_price, risk (price distance to
    stop, >0), pnl (price-denominated, cost-inclusive)."""
    if trades.empty:
        return pd.Series(dtype=float), initial
    eq = initial
    rows = []
    for row in trades.itertuples():
        if eq <= 0:
            break
        size = (eq * risk_pct) / max(row.risk, 1e-9)
        size = min(size, (max_lev * eq) / max(row.entry_price, 1e-9))
        eq = eq + size * row.pnl
        rows.append((row.exit_time, eq))
    s = pd.Series({t: e for t, e in rows}).sort_index()
    return s[~s.index.duplicated(keep="last")], float(eq)


def equity_stats(trades: pd.DataFrame, initial=INITIAL):
    eq, final = sim_equity(trades, initial=initial)
    if eq.empty:
        return dict(n=0, total_return_pct=0.0, cagr=np.nan, max_dd_pct=0.0, sharpe=np.nan)
    years = max((eq.index.max() - eq.index.min()).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (final / initial) ** (1 / years) - 1 if final > 0 else np.nan
    peak = eq.cummax()
    max_dd_pct = float(((eq - peak) / peak).min()) * 100
    idx = pd.date_range(eq.index.min().normalize(), eq.index.max().normalize() + pd.Timedelta(days=1), freq="D")
    daily = eq.reindex(idx.union(eq.index)).sort_index().ffill().reindex(idx).dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    return dict(n=len(trades), total_return_pct=(final / initial - 1) * 100,
                cagr=cagr, max_dd_pct=max_dd_pct, sharpe=sharpe)

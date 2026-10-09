"""
Test of the "grid martingale" strategy shown in bilibili video BV1Em421N7k3
(AlgoLee, "像一个十足的傻瓜一样去交易..."). Video content extracted via
downloaded mp4 + ffmpeg frame sampling (burned-in Chinese captions) + whisper
not needed (captions were readable directly).

Video shows two parts:
  1. A manual demo on NZD/CAD daily: buy at a support zone; if price keeps
     falling, add another buy at the next lower support zone (no stop loss);
     exit the whole basket when price recovers. Caption explicitly shows a
     -10,019 CAD floating loss at one point ("要的就是这个效果" / "this IS
     the effect we want") -- i.e. the video is demonstrating how deep this
     can go under water, not hiding it.
  2. An MT5 EA named "EvenWhenIAmWrong" backtested on NZDCAD H1 2012-2024,
     $10,000 start -> ~$36,000-41,000 equity, visually a grid/martingale EA:
     settings shown on-screen include 网格距离起点(grid spacing)=600 points,
     止损点数(stop-loss points)=0 (NO stop loss), 初始手数=0.2,
     加仓次数开始保本=10 (switch-to-breakeven after 10 adds). The equity
     curve is a near-straight line with several sharp drawdown spikes.

NO NZDCAD history is available locally (project only has
EURUSD/GBPUSD/AUDUSD/XAUUSD HistData M1, 2010-2026). This script
reconstructs the GENERAL MECHANISM (grid/martingale averaging with zero
stop-loss) -- not a literal replica of the closed-source EA's exact entry
signal or take-profit formula, which can't be recovered from a video alone
-- and tests it on the 4 available 16-year histories. This is deliberately
NOT an attempt to reproduce the video's exact equity curve; it is a stress
test of the one thing a video demo can't show: whether an unbounded,
no-stop-loss grid blows up somewhere in a long enough history, on
instruments other than the one cherry-picked in the demo.

Mechanism tested (same for every symbol/direction/multiplier combination):
  - Start a cycle with 1 position, lot = INITIAL_LOT, fixed direction
    (LONG-only grid and SHORT-only grid both tested separately, since with
    no stop loss the two are not symmetric over a 16-year secular trend).
  - If price moves GRID_DIST_PCT against the most recent (deepest) entry,
    add another position at the current price with lot = INITIAL_LOT *
    MULTIPLIER ** level (classic martingale lot progression).
  - NO stop loss anywhere (matches the video's explicit setting).
  - Close the entire basket (realize PnL, start a new cycle) once floating
    basket profit >= TP_USD.
  - Account "equity" = cash balance + floating PnL of all open grid
    positions, tracked bar-by-bar (H1 bars, resampled from M1). A run is
    flagged BLOWN the first bar equity <= 0 (simulation stops there --
    this is a lower-bound/conservative blowup trigger; real brokers
    stop out earlier at a positive margin-level threshold).

Position PnL uses $-per-point = price_diff * CONTRACT_SIZE (100,000 for the
FX majors, matching standard lot convention; 100 for XAUUSD, matching the
project's other gold scripts) -- no spread/commission modeled (irrelevant
to the question being tested here: does unbounded averaging blow up a fixed
starting balance).
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

INITIAL_BALANCE = 10_000.0
INITIAL_LOT = 0.2            # matches video's shown initial lot
GRID_DIST_PCT = 0.0060       # ~600 points on NZDCAD (~0.85-0.92) -> ~0.65%; round to 0.60%
TP_USD = 50.0                # fixed basket take-profit, USD


def load_h1(symbol):
    m1 = pd.read_parquet(f"{DATA_DIR}/{symbol}_histdata_M1.parquet").sort_values("time")
    h1 = (m1.set_index("time")["close"].resample("1h").last().dropna())
    return h1


def run_grid(h1, contract_size, direction, multiplier, grid_dist_pct=GRID_DIST_PCT, tp_usd=TP_USD):
    """direction: +1 = long-only grid, -1 = short-only grid."""
    balance = INITIAL_BALANCE
    positions = []  # list of (lot, entry_price)
    level = 0
    blown_at = None
    peak_equity = INITIAL_BALANCE
    max_dd_pct = 0.0
    max_level_reached = 0
    n_cycles = 0

    prices = h1.to_numpy()
    times = h1.index

    for i in range(len(prices)):
        px = prices[i]
        if not positions:
            positions.append((INITIAL_LOT, px))
            level = 0
            continue

        floating = sum(lot * contract_size * (px - entry) * direction for lot, entry in positions)
        equity = balance + floating
        peak_equity = max(peak_equity, equity)
        dd_pct = (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else -100.0
        max_dd_pct = min(max_dd_pct, dd_pct)
        max_level_reached = max(max_level_reached, level)

        if equity <= 0:
            blown_at = times[i]
            balance = equity
            positions = []
            break

        if floating >= tp_usd:
            balance += floating
            positions = []
            n_cycles += 1
            continue

        last_entry = positions[-1][1]
        adverse_move = (last_entry - px) * direction  # positive = against us
        if adverse_move >= last_entry * grid_dist_pct:
            level += 1
            lot = INITIAL_LOT * (multiplier ** level)
            positions.append((lot, px))

    final_floating = sum(lot * contract_size * (prices[-1] - entry) * direction for lot, entry in positions) if positions else 0.0
    final_equity = balance + final_floating
    return dict(
        blown=blown_at is not None,
        blown_at=blown_at,
        final_equity=final_equity,
        total_return_pct=(final_equity / INITIAL_BALANCE - 1) * 100,
        max_dd_pct=max_dd_pct,
        max_level_reached=max_level_reached,
        n_cycles=n_cycles,
    )


if __name__ == "__main__":
    for symbol, contract_size in SYMBOLS.items():
        h1 = load_h1(symbol)
        print(f"\n=== {symbol}  ({h1.index.min().date()} -> {h1.index.max().date()}, {len(h1)} H1 bars) ===")
        for direction, dname in [(1, "LONG-grid"), (-1, "SHORT-grid")]:
            for multiplier in [1.0, 1.5, 2.0]:
                r = run_grid(h1, contract_size, direction, multiplier)
                status = f"BLOWN at {r['blown_at']}" if r["blown"] else "survived"
                print(f"  {dname:11s} mult={multiplier:.1f}  {status:28s}  "
                      f"final_equity=${r['final_equity']:,.0f} ({r['total_return_pct']:+.1f}%)  "
                      f"max_dd={r['max_dd_pct']:.1f}%  max_grid_level={r['max_level_reached']}  "
                      f"cycles_closed={r['n_cycles']}")

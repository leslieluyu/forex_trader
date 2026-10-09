"""
Follow-up to backtest_grid_martingale.py, triggered by the user's proposal:
instead of hedging/locking a losing grid basket (open an opposite position
to freeze floating PnL, discussed but not implemented), what if the losing
grid basket is closed and FLIPPED to the opposite direction once price has
moved far enough against it -- i.e. treat a large-enough adverse move as a
signal that the trend has actually reversed, ride it with a single
trend-following position, then flip back to the default direction once
price rebounds.

User's own words: "在单向下跌一定阈值后，意味着开空单可以赚钱，可以直接转为
空单，直到开始反弹，这时将空单平掉，等待多单上涨" -- i.e. a stop-and-reverse
(SAR) overlay on top of the grid, not a simultaneous hedge.

Mechanism (state machine, continuous operation over the full history):
  - Default direction is LONG. A grid basket behaves exactly like
    backtest_grid_martingale.py: add a position every GRID_DIST_PCT adverse
    move, lot size growing by MULTIPLIER each level, until MAX_GRID_LEVELS
    is reached.
  - FLIP trigger: once the grid has added MAX_GRID_LEVELS positions and
    price moves another GRID_DIST_PCT against the basket, the ENTIRE basket
    is closed at market (realizing whatever loss/profit it has), and a
    single new position is opened in the OPPOSITE direction, sized at the
    basket's last (largest) grid lot -- i.e. "go with the move" at full
    conviction, no further averaging while flipped.
  - While flipped: track the best (most favorable) price reached since the
    flip. If price retraces back by REBOUND_PCT from that best price
    (trailing-stop logic), OR floating profit on the flipped position hits
    TP_USD, close it and reset: direction goes back to the default (LONG),
    and a fresh grid basket opens immediately.
  - No hard stop-loss is added anywhere beyond this structural flip/trail
    logic (keeps the comparison to backtest_grid_martingale.py meaningful:
    same no-stop-loss starting premise, same account simulation).
  - Blown == equity (balance + floating PnL of whatever is open) <= 0 at
    any bar.

This is NOT the same thing as the hedge/lock idea discussed earlier (which
keeps both legs open simultaneously). A flip only ever holds one direction
at a time, so margin usage doesn't double the way a simultaneous hedge
would -- but it also means the realized loss at each flip point is fully
booked immediately (no chance to recover without the new leg working).
"""
import numpy as np
import pandas as pd

from backtest_grid_martingale import DATA_DIR, SYMBOLS, INITIAL_BALANCE, INITIAL_LOT, GRID_DIST_PCT, TP_USD, load_h1

MAX_GRID_LEVELS = 3
REBOUND_PCT = GRID_DIST_PCT  # trailing-stop distance on the flipped leg, same magnitude as grid spacing


def run_grid_flip(h1, contract_size, start_direction=1, multiplier=1.5,
                   max_grid_levels=MAX_GRID_LEVELS, grid_dist_pct=GRID_DIST_PCT,
                   rebound_pct=REBOUND_PCT, tp_usd=TP_USD):
    balance = INITIAL_BALANCE
    direction = start_direction          # +1 long, -1 short
    mode = "grid"                        # "grid" or "flip"
    positions = []                       # grid mode: list of (lot, entry_price); flip mode: single (lot, entry_price)
    level = 0
    best_price_since_flip = None         # trailing reference while in flip mode
    peak_equity = INITIAL_BALANCE
    max_dd_pct = 0.0
    max_level_reached = 0
    n_cycles = 0
    n_flips = 0
    blown_at = None

    prices = h1.to_numpy()
    times = h1.index

    for i in range(len(prices)):
        px = prices[i]

        if not positions:
            positions = [(INITIAL_LOT, px)]
            level = 0
            mode = "grid"
            if direction == start_direction and i > 0:
                pass  # default direction grid restart (post-TP or post-flip-close)
            continue

        floating = sum(lot * contract_size * (px - entry) * direction for lot, entry in positions)
        equity = balance + floating
        peak_equity = max(peak_equity, equity)
        max_dd_pct = min(max_dd_pct, (equity - peak_equity) / peak_equity * 100 if peak_equity > 0 else -100.0)
        max_level_reached = max(max_level_reached, level)

        if equity <= 0:
            blown_at = times[i]
            balance = equity
            positions = []
            break

        if mode == "grid":
            if floating >= tp_usd:
                balance += floating
                positions = []
                n_cycles += 1
                continue

            last_entry = positions[-1][1]
            adverse_move = (last_entry - px) * direction
            if adverse_move >= last_entry * grid_dist_pct:
                if level < max_grid_levels:
                    level += 1
                    lot = INITIAL_LOT * (multiplier ** level)
                    positions.append((lot, px))
                else:
                    # flip: close the whole basket, open opposite-direction single leg
                    balance += floating
                    flip_lot = positions[-1][0]
                    direction = -direction
                    positions = [(flip_lot, px)]
                    mode = "flip"
                    best_price_since_flip = px
                    n_flips += 1
            continue

        # mode == "flip"
        lot, entry = positions[0]
        if direction == 1:
            best_price_since_flip = max(best_price_since_flip, px)
        else:
            best_price_since_flip = min(best_price_since_flip, px)

        if floating >= tp_usd:
            balance += floating
            positions = []
            n_cycles += 1
            direction = start_direction
            continue

        retrace = (best_price_since_flip - px) * direction
        if retrace >= best_price_since_flip * rebound_pct:
            balance += floating
            positions = []
            n_cycles += 1
            direction = start_direction
            continue

    if positions:
        final_floating = sum(lot * contract_size * (prices[-1] - entry) * direction for lot, entry in positions)
    else:
        final_floating = 0.0
    final_equity = balance + final_floating

    return dict(
        blown=blown_at is not None,
        blown_at=blown_at,
        final_equity=final_equity,
        total_return_pct=(final_equity / INITIAL_BALANCE - 1) * 100,
        max_dd_pct=max_dd_pct,
        max_level_reached=max_level_reached,
        n_cycles=n_cycles,
        n_flips=n_flips,
    )


if __name__ == "__main__":
    for symbol, contract_size in SYMBOLS.items():
        h1 = load_h1(symbol)
        print(f"\n=== {symbol}  ({h1.index.min().date()} -> {h1.index.max().date()}, {len(h1)} H1 bars) ===")
        for max_levels in [2, 3, 5]:
            for multiplier in [1.0, 1.5, 2.0]:
                r = run_grid_flip(h1, contract_size, start_direction=1, multiplier=multiplier,
                                   max_grid_levels=max_levels)
                status = f"BLOWN at {r['blown_at']}" if r["blown"] else "survived"
                print(f"  max_lvl={max_levels}  mult={multiplier:.1f}  {status:28s}  "
                      f"final_equity=${r['final_equity']:,.0f} ({r['total_return_pct']:+.1f}%)  "
                      f"max_dd={r['max_dd_pct']:.1f}%  cycles={r['n_cycles']}  flips={r['n_flips']}")

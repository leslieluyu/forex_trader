import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

import config_local as cfg

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = "XAUUSDm"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

if not mt5.initialize(
    path=TERMINAL_PATH,
    login=cfg.MT5_LOGIN,
    password=cfg.MT5_PASSWORD,
    server=cfg.MT5_SERVER,
):
    print("initialize() failed, error code =", mt5.last_error())
    raise SystemExit(1)

mt5.symbol_select(SYMBOL, True)
now = datetime.now()

# D1: full available history
d1 = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_D1, now, 20000)
df_d1 = pd.DataFrame(d1)
df_d1["time"] = pd.to_datetime(df_d1["time"], unit="s")
df_d1.to_parquet(os.path.join(OUT_DIR, f"{SYMBOL}_D1.parquet"), index=False)
print(f"D1: {len(df_d1)} rows, {df_d1['time'].min()} -> {df_d1['time'].max()}")

def fetch_chunked(fetch_fn, start, end, window_days):
    """copy_rates_range/copy_ticks_range reject windows wider than ~65 days;
    fetch in chunks and concatenate."""
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        chunk = fetch_fn(cur, nxt)
        if chunk is not None and len(chunk) > 0:
            frames.append(pd.DataFrame(chunk))
        cur = nxt
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"] if "time" in df.columns else None)
    return df


def fetch_chunked_to_disk(fetch_fn, start, end, window_days, out_dir):
    """Same as fetch_chunked but writes each chunk straight to its own parquet
    file instead of accumulating in memory (tick history can be tens of GB)."""
    os.makedirs(out_dir, exist_ok=True)
    cur = start
    total_rows = 0
    first_ts, last_ts = None, None
    while cur < end:
        nxt = min(cur + timedelta(days=window_days), end)
        chunk = fetch_fn(cur, nxt)
        if chunk is not None and len(chunk) > 0:
            df = pd.DataFrame(chunk)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            fname = f"part-{cur.date()}_{nxt.date()}.parquet"
            df.to_parquet(os.path.join(out_dir, fname), index=False)
            total_rows += len(df)
            first_ts = df["time"].min() if first_ts is None else first_ts
            last_ts = df["time"].max()
            print(f"  chunk {cur.date()}->{nxt.date()}: {len(df)} rows")
        cur = nxt
    return total_rows, first_ts, last_ts


# M1: full available history (probed depth ~2026-06-29)
m1_start = datetime(2026, 6, 29)
df_m1 = fetch_chunked(
    lambda a, b: mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, a, b),
    m1_start, now, window_days=30,
)
df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
df_m1.to_parquet(os.path.join(OUT_DIR, f"{SYMBOL}_M1.parquet"), index=False)
print(f"M1: {len(df_m1)} rows, {df_m1['time'].min()} -> {df_m1['time'].max()}")

# Ticks: full available history (probed depth ~2026-01-02), written per-chunk
# to disk to avoid holding tens of millions of rows in the VM's ~4GB RAM.
tick_start = datetime(2026, 1, 2)
tick_dir = os.path.join(OUT_DIR, f"{SYMBOL}_ticks")
total_rows, first_ts, last_ts = fetch_chunked_to_disk(
    lambda a, b: mt5.copy_ticks_range(SYMBOL, a, b, mt5.COPY_TICKS_ALL),
    tick_start, now, window_days=14, out_dir=tick_dir,
)
print(f"ticks: {total_rows} rows total, {first_ts} -> {last_ts}")

mt5.shutdown()

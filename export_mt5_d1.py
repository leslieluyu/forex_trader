import os
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

import config_local as cfg

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = "XAUUSDm"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", f"{SYMBOL}_D1_full.parquet")

if not mt5.initialize(path=TERMINAL_PATH, login=cfg.MT5_LOGIN,
                       password=cfg.MT5_PASSWORD, server=cfg.MT5_SERVER):
    print("initialize() failed, error code =", mt5.last_error())
    raise SystemExit(1)

mt5.symbol_select(SYMBOL, True)
d1 = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_D1, datetime.now(), 20000)
df = pd.DataFrame(d1)
df["time"] = pd.to_datetime(df["time"], unit="s")
df.to_parquet(OUT_PATH, index=False)
print(f"D1: {len(df)} rows, {df['time'].min()} -> {df['time'].max()}, saved to {OUT_PATH}")
mt5.shutdown()

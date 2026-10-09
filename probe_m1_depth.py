import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
from datetime import datetime

import config_local as cfg

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = "XAUUSDm"

if not mt5.initialize(
    path=TERMINAL_PATH,
    login=cfg.MT5_LOGIN,
    password=cfg.MT5_PASSWORD,
    server=cfg.MT5_SERVER,
):
    print("initialize() failed, error code =", mt5.last_error())
    raise SystemExit(1)

mt5.symbol_select(SYMBOL, True)

candidates = [
    datetime(2010, 1, 1),
    datetime(2014, 1, 1),
    datetime(2018, 1, 1),
    datetime(2020, 1, 1),
    datetime(2022, 1, 1),
    datetime(2024, 1, 1),
    datetime(2025, 1, 1),
]

for d in candidates:
    bars = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_M1, d, 5)
    if bars is None or len(bars) == 0:
        print(f"{d.date()} -> no M1 bars returned, error={mt5.last_error()}")
    else:
        first_ts = datetime.fromtimestamp(bars[0]["time"])
        print(f"{d.date()} -> earliest returned M1 bar at {first_ts}")

mt5.shutdown()

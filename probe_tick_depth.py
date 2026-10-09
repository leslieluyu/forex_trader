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

# copy_ticks_from pulls ticks starting at/after the given date; probing several
# candidate start dates tells us roughly where real tick history begins.
candidates = [
    datetime(2010, 1, 1),
    datetime(2014, 1, 1),
    datetime(2016, 1, 1),
    datetime(2018, 1, 1),
    datetime(2020, 1, 1),
    datetime(2022, 1, 1),
    datetime(2024, 1, 1),
]

for d in candidates:
    ticks = mt5.copy_ticks_from(SYMBOL, d, 5, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        print(f"{d.date()} -> no ticks returned, error={mt5.last_error()}")
    else:
        first_ts = datetime.fromtimestamp(ticks[0]["time"])
        print(f"{d.date()} -> earliest returned tick at {first_ts}")

mt5.shutdown()

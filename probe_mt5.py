import MetaTrader5 as mt5
from datetime import datetime, timedelta

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

print("terminal_info:", mt5.terminal_info())
print("account_info:", mt5.account_info())

if not mt5.symbol_select(SYMBOL, True):
    print("symbol_select failed for", SYMBOL, mt5.last_error())
else:
    tick = mt5.symbol_info_tick(SYMBOL)
    print("latest tick:", tick)

    now = datetime.now()
    ticks = mt5.copy_ticks_range(SYMBOL, now - timedelta(minutes=5), now, mt5.COPY_TICKS_ALL)
    print(f"ticks in last 5min: {0 if ticks is None else len(ticks)}")

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M1, now - timedelta(days=1), now)
    print(f"M1 bars in last 1 day: {0 if rates is None else len(rates)}")

    oldest = mt5.copy_rates_from(SYMBOL, mt5.TIMEFRAME_D1, now, 20000)
    if oldest is not None and len(oldest) > 0:
        print(f"D1 history depth: {len(oldest)} bars, earliest = {datetime.fromtimestamp(oldest[0]['time'])}")
    else:
        print("D1 history: no data")

mt5.shutdown()

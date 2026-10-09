import MetaTrader5 as mt5
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
info = mt5.symbol_info(SYMBOL)
if info is None:
    print("symbol_info failed", mt5.last_error())
else:
    print("swap_long:", info.swap_long)
    print("swap_short:", info.swap_short)
    print("swap_mode:", info.swap_mode)
    print("swap_rollover3days:", info.swap_rollover3days)
    print("trade_contract_size:", info.trade_contract_size)
    print("point:", info.point)
    print("digits:", info.digits)
    print("volume_min:", info.volume_min)
    print("session info / currency_base:", info.currency_base, info.currency_profit)

mt5.shutdown()

import MetaTrader5 as mt5
import config_local as cfg

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
CANDIDATES = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "USDCADm",
              "NZDUSDm", "USDCHFm", "EURJPYm", "GBPJPYm", "EURGBPm"]

if not mt5.initialize(
    path=TERMINAL_PATH,
    login=cfg.MT5_LOGIN,
    password=cfg.MT5_PASSWORD,
    server=cfg.MT5_SERVER,
):
    print("initialize() failed, error code =", mt5.last_error())
    raise SystemExit(1)

print(f"{'symbol':<10} {'swap_long':>10} {'swap_short':>11} {'point':>8} {'digits':>7} {'contract':>9} {'min_vol':>8}")
for sym in CANDIDATES:
    if not mt5.symbol_select(sym, True):
        print(f"{sym:<10} symbol_select failed: {mt5.last_error()}")
        continue
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"{sym:<10} symbol_info failed")
        continue
    print(f"{sym:<10} {info.swap_long:>10.2f} {info.swap_short:>11.2f} {info.point:>8.5f} "
          f"{info.digits:>7d} {info.trade_contract_size:>9.0f} {info.volume_min:>8.2f}")

mt5.shutdown()

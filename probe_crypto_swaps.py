import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5
import config_local as cfg

TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
CANDIDATES = ["BTCUSDm", "BTCUSD", "BTCUSDTm", "BTCUSDT", "ETHUSDm", "ETHUSD"]

if not mt5.initialize(
    path=TERMINAL_PATH,
    login=cfg.MT5_LOGIN,
    password=cfg.MT5_PASSWORD,
    server=cfg.MT5_SERVER,
):
    print("initialize() failed, error code =", mt5.last_error())
    raise SystemExit(1)

all_syms = [s.name for s in mt5.symbols_get()]
crypto_syms = [s for s in all_syms if "BTC" in s.upper() or "ETH" in s.upper()]
print("crypto-ish symbols found on this account:", crypto_syms)
print()

print(f"{'symbol':<12} {'swap_long':>10} {'swap_short':>11} {'swap_mode':>10} {'point':>10} "
      f"{'digits':>7} {'contract':>10} {'min_vol':>8} {'base':>6} {'profit_ccy':>10}")
for sym in sorted(set(CANDIDATES) | set(crypto_syms)):
    if not mt5.symbol_select(sym, True):
        print(f"{sym:<12} symbol_select failed: {mt5.last_error()}")
        continue
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"{sym:<12} symbol_info failed")
        continue
    print(f"{sym:<12} {info.swap_long:>10.4f} {info.swap_short:>11.4f} {info.swap_mode:>10d} "
          f"{info.point:>10.6f} {info.digits:>7d} {info.trade_contract_size:>10.4f} "
          f"{info.volume_min:>8.4f} {info.currency_base:>6} {info.currency_profit:>10}")

mt5.shutdown()

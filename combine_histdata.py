import glob
import os

import pandas as pd

SRC_DIR = "/tmp/histdata_extract"
OUT_PATH = "/Volumes/My Passport/working_data/forex_trader/data/XAUUSD_histdata_M1.parquet"

files = sorted(glob.glob(os.path.join(SRC_DIR, "DAT_ASCII_XAUUSD_M1_*.csv")))
print(f"found {len(files)} files")

frames = []
for f in files:
    df = pd.read_csv(
        f, sep=";", header=None,
        names=["time", "open", "high", "low", "close", "volume"],
    )
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d %H%M%S")
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
full = full.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)

# HistData timestamps are true New York local wall-clock time, WITH DST adjustment
# (verified empirically: NFP release at 8:30am NY time lands on local hour 8:30 in
# both winter and summer samples; weekly Sunday open sits at local hour 17 in all
# 12 calendar months -- both would drift by ~1h across seasons if this were fixed
# EST/no-DST, which it is not). Not converting here -- keep as-is.
print(f"total rows: {len(full)}")
print(f"range: {full['time'].min()} -> {full['time'].max()}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
full.to_parquet(OUT_PATH, index=False)
print(f"saved to {OUT_PATH}")

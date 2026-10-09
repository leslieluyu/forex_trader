import glob
import os
import sys

import pandas as pd

SYMBOL = sys.argv[1].upper()  # e.g. EURUSD
SRC_DIR = f"/tmp/histdata_extract_{SYMBOL.lower()}"
OUT_PATH = f"/Volumes/My Passport/working_data/forex_trader/data/{SYMBOL}_histdata_M1.parquet"

files = sorted(glob.glob(os.path.join(SRC_DIR, f"DAT_ASCII_{SYMBOL}_M1_*.csv")))
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

print(f"total rows: {len(full)}")
print(f"range: {full['time'].min()} -> {full['time'].max()}")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
full.to_parquet(OUT_PATH, index=False)
print(f"saved to {OUT_PATH}")

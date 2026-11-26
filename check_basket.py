from pathlib import Path

import pandas as pd

print("=== Rows and date range per ticker ===")
for p in sorted(Path("data/basket").glob("*.parquet")):
    df = pd.read_parquet(p)
    print(f"{p.stem:6s} rows={len(df):5d}  {df.index.min().date()} -> {df.index.max().date()}")

# Compare AAPL's buy-and-hold over the post-warm-up window, basket file vs the earlier file
print("\n=== AAPL buy-and-hold check (warm-up = 200 rows) ===")
for label, path in [("basket file", "data/basket/AAPL.parquet"),
                    ("earlier clean file", "data/AAPL_clean.parquet")]:
    close = pd.read_parquet(path)["Close"]
    ret = close.pct_change()
    window = ret.iloc[200:]
    print(f"{label:20s} window starts {window.index[0].date()}, "
          f"ends {window.index[-1].date()}, "
          f"buy-and-hold total = {(1 + window).prod() - 1:.3f}")
import pandas as pd

TICKER = "AAPL"

df = pd.read_parquet(f"data/{TICKER}.parquet")

print("=== 1. Shape and types ===")
print(df.shape)
print(df.dtypes)

print("\n=== 2. Index checks ===")
print("Timezone:", df.index.tz)
print("Sorted ascending:", df.index.is_monotonic_increasing)
print("Duplicate dates:", df.index.duplicated().sum())
print("First date:", df.index.min())
print("Last date:", df.index.max())

print("\n=== 3. Missing values ===")
print(df.isna().sum())

print("\n=== 4. Sanity checks ===")
print("Rows with zero volume:", (df["Volume"] == 0).sum())
print("Rows with Close <= 0:", (df["Close"] <= 0).sum())
print("Rows where High < Low:", (df["High"] < df["Low"]).sum())

print("\n=== 5. Biggest daily moves ===")
daily_change = df["Close"].pct_change()
print(daily_change.abs().sort_values(ascending=False).head(5))

print("\n=== 6. Summary stats ===")
print(df[["Open", "High", "Low", "Close", "Volume"]].describe())


# --- Cleaning ---
# Keep only the columns we need, drop timezone, keep the date only
clean = df[["Open", "High", "Low", "Close", "Volume"]].copy()
clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
clean.index = clean.index.tz_localize(None).normalize()
clean.index.name = "Date"

print("=== 7. Shape after filtering ===")
print(clean.shape)

clean.to_parquet(f"data/{TICKER}_clean.parquet")
print(f"\nSaved cleaned data to data/{TICKER}_clean.parquet")
print(clean.head(3))
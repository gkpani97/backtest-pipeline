import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save to file, no window needed
import matplotlib.pyplot as plt

TICKER = "AAPL"

df = pd.read_parquet(f"data/{TICKER}_clean.parquet")

# --- Compute the indicators ---
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()

# --- Check 1: warm-up period ---
print("=== Non-null counts ===")
print(df[["Close", "SMA20", "SMA50"]].count())

print("\n=== First valid row of each ===")
print("SMA20 starts:", df["SMA20"].first_valid_index())
print("SMA50 starts:", df["SMA50"].first_valid_index())

# --- Check 2: verify by hand ---
manual_sma20 = df["Close"].iloc[-20:].mean()
print("\n=== Manual check on the last row ===")
print("Pandas SMA20:", df["SMA20"].iloc[-1])
print("Manual SMA20:", manual_sma20)
print("Match:", abs(df["SMA20"].iloc[-1] - manual_sma20) < 1e-9)

print("\n=== Last 5 rows ===")
print(df[["Close", "SMA20", "SMA50"]].tail())

# --- Plot the last year ---
recent = df.iloc[-250:]
plt.figure(figsize=(12, 6))
plt.plot(recent.index, recent["Close"], label="Close", linewidth=1)
plt.plot(recent.index, recent["SMA20"], label="SMA 20")
plt.plot(recent.index, recent["SMA50"], label="SMA 50")
plt.title(f"{TICKER}: Close with SMA 20 and SMA 50 (last 250 days)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("data/sma_check.png", dpi=120, bbox_inches="tight")
print("\nSaved chart to data/sma_check.png")

df.to_parquet(f"data/{TICKER}_indicators.parquet")
print(f"Saved data/{TICKER}_indicators.parquet")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "AAPL"

df = pd.read_parquet(f"data/{TICKER}_indicators.parquet")

# Keep only rows where both averages exist
df = df.dropna(subset=["SMA20", "SMA50"]).copy()

# Signal: 1 if SMA20 is above SMA50 (trend up), else 0
df["Signal"] = (df["SMA20"] > df["SMA50"]).astype(int)

# Cross: the day-to-day change in the signal
#  +1 = bullish cross (buy), -1 = bearish cross (sell), 0 = no change
df["Cross"] = df["Signal"].diff()

# --- Summary ---
buys = (df["Cross"] == 1).sum()
sells = (df["Cross"] == -1).sum()
print("=== Summary ===")
print("Rows:", len(df))
print("First date:", df.index.min().date(), "| starting signal:", df["Signal"].iloc[0])
print("Buy crosses:", buys)
print("Sell crosses:", sells)
print("Share of days with signal = 1:", round(df["Signal"].mean() * 100, 1), "%")

# --- Every crossover ---
trades = df[df["Cross"].isin([1, -1])][["Close", "SMA20", "SMA50", "Cross"]].copy()
trades["Action"] = trades["Cross"].map({1: "BUY", -1: "SELL"})
trades["Days_since_last"] = trades.index.to_series().diff().dt.days
print("\n=== All crossovers ===")
print(trades[["Close", "Action", "Days_since_last"]].round(2).to_string())

# --- Chart ---
plt.figure(figsize=(13, 6))
plt.plot(df.index, df["Close"], label="Close", linewidth=0.9)
plt.plot(df.index, df["SMA20"], label="SMA 20")
plt.plot(df.index, df["SMA50"], label="SMA 50")
b = df[df["Cross"] == 1]
s = df[df["Cross"] == -1]
plt.scatter(b.index, b["Close"], marker="^", color="green", s=90, label="Buy", zorder=5)
plt.scatter(s.index, s["Close"], marker="v", color="red", s=90, label="Sell", zorder=5)
plt.title(f"{TICKER}: crossover signals")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("data/signals.png", dpi=120, bbox_inches="tight")
print("\nSaved chart to data/signals.png")

df.to_parquet(f"data/{TICKER}_signals.parquet")
print(f"Saved data/{TICKER}_signals.parquet")
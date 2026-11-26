import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "AAPL"
df = pd.read_parquet(f"data/{TICKER}_returns.parquet")

# Drawdown = how far below the previous peak we are
df["DD_strategy"] = df["Equity_strategy"] / df["Equity_strategy"].cummax() - 1
df["DD_buyhold"] = df["Equity_buyhold"] / df["Equity_buyhold"].cummax() - 1

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

# Top panel: equity curves
ax1.plot(df.index, df["Equity_buyhold"], label="Buy and hold")
ax1.plot(df.index, df["Equity_strategy"], label="SMA 20/50 strategy")
ax1.fill_between(
    df.index, 0, 1,
    where=(df["Position"] == 1),
    transform=ax1.get_xaxis_transform(),
    color="green", alpha=0.08, label="Strategy in the market",
)
ax1.set_title(f"{TICKER}: growth of $1")
ax1.legend()
ax1.grid(alpha=0.3)

# Bottom panel: drawdowns
ax2.plot(df.index, df["DD_buyhold"] * 100, label="Buy and hold")
ax2.plot(df.index, df["DD_strategy"] * 100, label="Strategy")
ax2.set_title("Drawdown from previous peak (%)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.savefig("data/backtest.png", dpi=120, bbox_inches="tight")
print("Saved chart to data/backtest.png")

# Calendar-year breakdown
by_year = df.index.year
yearly = pd.DataFrame({
    "Strategy": df["Strategy"].groupby(by_year).apply(lambda r: (1 + r).prod() - 1),
    "Buy and hold": df["Return"].groupby(by_year).apply(lambda r: (1 + r).prod() - 1),
    "Days in market %": df["Position"].groupby(by_year).mean() * 100,
})
print("\n=== Returns by calendar year (0.10 means 10%) ===")
print(yearly.round(3))
print("\nNote: 2022 starts in February and 2026 ends on Nov 19, so both are partial years.")
import pandas as pd

TICKER = "AAPL"

df = pd.read_parquet(f"data/{TICKER}_signals.parquet")

# 1. Stock's daily return
df["Return"] = df["Close"].pct_change()

# 2. Position = yesterday's signal (we can only act after seeing the close)
df["Position"] = df["Signal"].shift(1)

# 3. Strategy return
df["Strategy"] = df["Position"] * df["Return"]

# The lookahead "cheat": trading on the same day's signal (NOT realistic)
df["Strategy_cheat"] = df["Signal"] * df["Return"]

# First row has no previous close/signal, so drop it
df = df.dropna(subset=["Return", "Position"]).copy()

# Equity curves: growth of $1
df["Equity_strategy"] = (1 + df["Strategy"]).cumprod()
df["Equity_cheat"] = (1 + df["Strategy_cheat"]).cumprod()
df["Equity_buyhold"] = (1 + df["Return"]).cumprod()

# --- Check 1: the shift around the first buy signal ---
print("=== Around the first BUY (2022-04-05) ===")
cols = ["Close", "Signal", "Position", "Return", "Strategy"]
print(df.loc["2022-04-01":"2022-04-08", cols].round(4))

# --- Check 2: days in the market ---
print("\n=== Exposure ===")
print("Days held:", int(df["Position"].sum()), "of", len(df))

# --- Check 3: growth of $1 ---
print("\n=== Growth of $1 ===")
print("Strategy (honest):       ", round(df["Equity_strategy"].iloc[-1], 3))
print("Strategy (lookahead):    ", round(df["Equity_cheat"].iloc[-1], 3))
print("Buy and hold:            ", round(df["Equity_buyhold"].iloc[-1], 3))

df.to_parquet(f"data/{TICKER}_returns.parquet")
print(f"\nSaved data/{TICKER}_returns.parquet")
import numpy as np
import pandas as pd

TICKER = "AAPL"
TRADING_DAYS = 252
COST = 0.001
WARMUP = 200   # common start for every combination

df = pd.read_parquet(f"data/{TICKER}_clean.parquet")
ret = df["Close"].pct_change()

FASTS = [5, 10, 20, 30, 40]
SLOWS = [50, 100, 150, 200]


def backtest(fast, slow):
    signal = (df["Close"].rolling(fast).mean() > df["Close"].rolling(slow).mean()).astype(int)
    position = signal.shift(1).fillna(0)
    trades = position.diff().abs().fillna(0)
    strat = (position * ret - trades * COST).iloc[WARMUP:]
    equity = (1 + strat).cumprod()
    return {
        "total": equity.iloc[-1] - 1,
        "sharpe": strat.mean() / strat.std() * np.sqrt(TRADING_DAYS),
        "trades": int(trades.iloc[WARMUP:].sum()),
    }


total = pd.DataFrame(index=FASTS, columns=SLOWS, dtype=float)
sharpe = total.copy()
trades = total.copy()

for f in FASTS:
    for s in SLOWS:
        if f < s:
            r = backtest(f, s)
            total.loc[f, s] = r["total"]
            sharpe.loc[f, s] = r["sharpe"]
            trades.loc[f, s] = r["trades"]

bh = ret.iloc[WARMUP:]
print("Same test window:", df.index[WARMUP].date(), "to", df.index[-1].date())
print("Buy and hold: total", round((1 + bh).prod() - 1, 3),
      "| Sharpe", round(bh.mean() / bh.std() * np.sqrt(TRADING_DAYS), 3))

print("\n=== Total return (rows = fast window, columns = slow window) ===")
print(total.round(3))
print("\n=== Sharpe ratio ===")
print(sharpe.round(2))
print("\n=== Trade legs ===")
print(trades)
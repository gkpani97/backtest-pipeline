import numpy as np
import pandas as pd

TICKER = "AAPL"
TRADING_DAYS = 252

df = pd.read_parquet(f"data/{TICKER}_returns.parquet")


def drawdown_series(returns: pd.Series):
    equity = (1 + returns).cumprod()
    return equity / equity.cummax() - 1


def metrics(returns: pd.Series):
    equity = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR (per year)": equity.iloc[-1] ** (1 / years) - 1,
        "Volatility (annual)": returns.std() * np.sqrt(TRADING_DAYS),
        "Sharpe (rf=0)": returns.mean() / returns.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": drawdown_series(returns).min(),
    }


table = pd.DataFrame({
    "Strategy": metrics(df["Strategy"]),
    "Buy and hold": metrics(df["Return"]),
})
print("=== Metrics (0.277 means 27.7%) ===")
print(table.round(3))

print("\n=== Worst drawdowns ===")
for name, r in [("Strategy", df["Strategy"]), ("Buy and hold", df["Return"])]:
    dd = drawdown_series(r)
    trough = dd.idxmin()
    peak = (1 + r).cumprod().loc[:trough].idxmax()
    print(f"{name}: {dd.min():.1%} from peak {peak.date()} to through {trough.date()}")

# --- Trade-level stats ---
# Label each continuous holding stretch with an id
held = df["Position"] == 1
trade_id = (held != held.shift()).cumsum()
trade_returns = (1 + df["Strategy"][held]).groupby(trade_id[held]).prod() - 1

print("\n=== Trades ===")
print("Number of trades:", len(trade_returns))
print("Win rate:", round((trade_returns > 0).mean() * 100, 1), "%")
print("Average win: ", round(trade_returns[trade_returns > 0].mean() * 100, 1), "%")
print("Average loss:", round(trade_returns[trade_returns <= 0].mean() * 100, 1), "%")
print("Last trade (still open):", round(trade_returns.iloc[-1] * 100, 1), "%")
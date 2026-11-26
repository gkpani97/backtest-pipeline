import numpy as np
import pandas as pd

TICKER = "AAPL"
TRADING_DAYS = 252
COST_PER_TRADE = 0.001   # 0.1% per buy or sell

df = pd.read_parquet(f"data/{TICKER}_returns.parquet")


def run(delay, cost):
    position = df["Signal"].shift(delay).fillna(0)
    trades = position.diff().abs().fillna(0)          # 1 on each buy or sell day
    ret = position * df["Return"] - trades * cost
    equity = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Sharpe": ret.mean() / ret.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Trade legs": int(trades.sum()),
    }


scenarios = {
    "1. Baseline (delay 1, no cost)": run(1, 0),
    "2. Delay 2, no cost": run(2, 0),
    "3. Delay 1, with cost": run(1, COST_PER_TRADE),
    "4. Delay 2, with cost": run(2, COST_PER_TRADE),
}
print(pd.DataFrame(scenarios).T.round(3))
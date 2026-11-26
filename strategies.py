from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Contract: signal(close) -> Series of 1 (hold) / 0 (cash), same index as close."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def signal(self, close: pd.Series) -> pd.Series: ...


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    """Benchmark: always hold."""

    @property
    def name(self) -> str:
        return "Buy & hold"

    def signal(self, close: pd.Series) -> pd.Series:
        return pd.Series(1, index=close.index)


@dataclass(frozen=True)
class SmaCrossover(Strategy):
    """Trend following: hold while the fast average is above the slow one."""
    fast: int = 20
    slow: int = 50

    def __post_init__(self):
        if self.fast >= self.slow:
            raise ValueError("fast window must be smaller than slow window")

    @property
    def name(self) -> str:
        return f"SMA {self.fast}/{self.slow}"

    def signal(self, close: pd.Series) -> pd.Series:
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        return (fast_ma > slow_ma).astype(int)   # NaN during warm-up compares False -> 0


@dataclass(frozen=True)
class RsiMeanReversion(Strategy):
    """Mean reversion: buy when RSI drops below entry_below, sell when it rises above exit_above."""
    period: int = 14
    entry_below: float = 30.0
    exit_above: float = 55.0

    @property
    def name(self) -> str:
        return f"RSI({self.period}) <{self.entry_below:g} >{self.exit_above:g}"

    def _rsi(self, close: pd.Series) -> pd.Series:
        change = close.diff()
        gain = change.clip(lower=0)
        loss = -change.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / self.period, adjust=False).mean()   # Wilder smoothing
        avg_loss = loss.ewm(alpha=1 / self.period, adjust=False).mean()
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        rsi = rsi.where(avg_loss != 0, 100.0)       # no losses at all -> RSI 100
        rsi.iloc[: self.period] = np.nan            # warm-up: not enough history yet
        return rsi

    def signal(self, close: pd.Series) -> pd.Series:
        rsi = self._rsi(close).to_numpy()
        out = np.zeros(len(rsi), dtype=int)
        holding = 0
        for i, value in enumerate(rsi):             # stateful: remember whether we are in a trade
            if holding == 0 and value < self.entry_below:
                holding = 1
            elif holding == 1 and value > self.exit_above:
                holding = 0
            out[i] = holding
        return pd.Series(out, index=close.index)
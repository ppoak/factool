import numpy as np
import pandas as pd
from .base import (
    quotes_day, 
    BaseFactor,
    zscore,
)


class LiquidityFactor(BaseFactor):

    def get_turnover_month(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        volume = quotes_day.read("volume", start=rollback, stop=date)
        shares = quotes_day.read("circulation_a", start=rollback, stop=date).dropna()
        res = np.log((volume / shares).sum().clip(lower=1e-10))
        res.name = date
        return res * 0.35

    def get_turnover_quarter(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0
        for i in range(0, 43, 21):
            res += np.exp(self.get_turnover_month(quotes_day.get_trading_days_rollback(date, i)))
        res = np.log(res / 3)
        res.name = date
        return res
    
    def get_turnover_annual(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0
        for i in range(0, 232, 21):
            res += np.exp(self.get_turnover_month(quotes_day.get_trading_days_rollback(date, i)))
        res = np.log(res / 12)
        res.name = date
        return res

    def get_compound_turnover(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0.35 * zscore(self.get_turnover_month(date).to_frame().T) + \
            0.35 * zscore(self.get_turnover_quarter(date).to_frame().T) + \
            0.3 * zscore(self.get_turnover_annual(date).to_frame().T)
        res.loc[date]
        res.name = date
        return res

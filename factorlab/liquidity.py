import numpy as np
import pandas as pd
import statsmodels.api as sm
from .factor import Factor


class LiquidityFactor(Factor):

    def get_turnover_month(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        volume = quotes_day.read("volume", start=rollback, stop=date)
        shares = quotes_day.read("circulation_a", start=rollback, stop=date)
        res = np.log((volume / shares).sum().clip(lower=1e-10))
        res.name = date
        return res * 0.35

    def get_turnover_quarter(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0
        for i in range(0, 43, 21):
            res += np.exp(self.get_turnover_month(quotes_day.get_trading_days_rollback(date, i)))
        res = np.log((res / 3).clip(lower=1e-10))
        res.name = date
        return res
    
    def get_turnover_annual(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0
        for i in range(0, 232, 21):
            res += np.exp(self.get_turnover_month(quotes_day.get_trading_days_rollback(date, i)))
        res = np.log((res / 12).clip(lower=1e-10))
        res.name = date
        return res

    def get_compound_turnover(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0.35 * zscore(self.get_turnover_month(date).to_frame().T) + \
            0.35 * zscore(self.get_turnover_quarter(date).to_frame().T) + \
            0.3 * zscore(self.get_turnover_annual(date).to_frame().T)
        res = res.loc[date]
        res.name = date
        return res

    def get_turnover_cv(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        volume = quotes_day.read("volume", start=rollback, stop=date)
        shares = quotes_day.read("circulation_a", start=rollback, stop=date)
        turnover = (volume / shares).tail(20)
        res = turnover.std()/turnover.mean()
        res.name = date
        return res

    def get_nonliquidity_cv(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        ret  = (price * _adj).pct_change(fill_method=None).tail(20).abs()
        amount = quotes_day.read("amount", start=rollback, stop=date).tail(20)
        nonliquidity = ret / amount
        res = nonliquidity.std()/ nonliquidity.mean()
        res.name = date
        return res

    def get_hli(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 1)
        high = quotes_day.read("high", start=rollback, stop=date)
        low = quotes_day.read("low", start=rollback, stop=date)
        p1 = (np.log(high/low).sum()**2) / 2
        p2 = np.log(high.max()/low.min())**2
        p3 = (np.sqrt(2*p1)-np.sqrt(p1))/(3-2*np.sqrt(2)) - np.sqrt(p2/(3-2*np.sqrt(2)))
        hl = 2*(np.exp(p3) -1)/ (1 + np.exp(p3))
        amount = quotes_day.read("amount", start=date, stop=date).squeeze()
        res = hl/amount
        res.name = date
        return res

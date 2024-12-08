import numpy as np
import pandas as pd
import statsmodels.api as sm
from .factor import FactorManager


class MarketSizeFactor(FactorManager):

    def get_log_marketcap(self, date: str | pd.Timestamp) -> pd.Series:        
        shares = quotes_day.read("circulation_a", start=date, stop=date)
        price = quotes_day.read("close", start=date, stop=date)
        adjfactor = quotes_day.read("adjfactor", start=date, stop=date)
        res = np.log(shares * price * adjfactor).squeeze()
        return res

    def get_nonlinear_size(self, date: str | pd.Timestamp) -> pd.Series:
        marketcap = self.get_log_marketcap(date)
        y = (marketcap ** 3).dropna()
        x = sm.add_constant(marketcap).dropna()
        model = sm.OLS(y, x)
        res = model.fit()
        res = res.resid

        #缩尾和标准化
        mean = res.mean()
        std = res.std()
        res = res.clip(mean - 3 * std, mean + 3 * std)
        res = (res - res.mean()) / res.std()
        res.name = date
        return res
        
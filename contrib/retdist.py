import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import FactorManager


class RetDistFactor(FactorManager):

    def get_intraday_distribution(self, date: str) -> pd.DataFrame:
        data = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = data.pct_change(fill_method=None)
        ret.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = pd.concat([ret.skew(), ret.kurt()], axis=1, 
            keys=['intraday_return_skew', 'intraday_return_kurt'])
        res.index = pd.MultiIndex.from_product([
            res.index, [date]], names=["order_book_id", "date"])
        return res

    def get_down_trend_volatility(self, date: str) -> pd.DataFrame:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        def safe_divide(x):
            negative_sum = x[x < 0].pow(2).sum()
            total_sum = x.pow(2).sum()
            if total_sum == 0:
                return np.nan
            else:
                return negative_sum / total_sum
        res = ret.apply(safe_divide)
        res.name = date
        return res
        
    def get_long_short_ratio(self, date: str) -> pd.DataFrame:
        rollback = quotes_day.get_trading_days_rollback(date, 5)
        price = quotes_min.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        vol_per_unit = abs(vol / ret).replace([np.inf, -np.inf], np.nan)
        tot_ret = (price.iloc[-1] / price.iloc[0] - 1).abs()
        res = (tot_ret * vol_per_unit.mean()) / vol.sum()
        res.name = date
        return res

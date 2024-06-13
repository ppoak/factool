import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, quotes_min, index_quotes_day, 
    BaseFactor
)


class RetDistFactor(BaseFactor):

    def get_intraday_distribution(self, date: str) -> pd.DataFrame:
        data = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = data.pct_change(fill_method=None)
        res = pd.concat([ret.skew(), ret.kurt()], axis=1, 
            keys=['intraday_return_skew', 'intraday_return_kurt'])
        res.index = pd.MultiIndex.from_product([
            res.index, [date]], names=["order_book_id", "date"])
        return res

    def get_down_trend_volatility(self, date: str) -> pd.DataFrame:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        res = ret.apply(lambda x: x[x < 0].pow(2).sum() / x.pow(2).sum())
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

    def get_coskewness(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 126)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(126)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(126).loc[:,'000985.XSHG']
        X = sm.add_constant(market_ret)
        y = stock_ret
        model = sm.OLS(y, X).fit()

        epsilon_i = model.resid
        epsilon_m = market_ret - market_ret.mean()
        res = np.mean(epsilon_i.mul(epsilon_m**2, axis=0),axis=0) / (np.sqrt((epsilon_i**2).mean()) * (epsilon_m**2).mean())
        res.index.name = 'order_book_id'
        res.name = date
        return res

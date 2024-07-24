import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, index_quotes_day, quotes_min,
    index_quotes_min,
    BaseFactor,
    zscore,
)


class LiquidityFactor(BaseFactor):

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
    
    def get_20d_resid_ill_liquidity(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        ret  = (price * _adj).pct_change(fill_method=None).tail(20)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(20).squeeze()

        X = sm.add_constant(market_ret)
        y = ret
        model = sm.OLS(y, X).fit()
        epsilon = model.resid

        amount = quotes_day.read("amount", start=rollback, stop=date).tail(20)
        res = (epsilon.abs()/amount).mean()
        res = res.replace([np.inf, -np.inf], np.nan)
        res.index.name = 'order_book_id'
        res.name = date
        return res

    def get_5minute_resid_ill_liquidity(self, date: str | pd.Timestamp) -> pd.Series:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret  = price.pct_change(periods=5, fill_method=None).dropna(how='all')
        market_ret = index_quotes_min.read('close', code='000001.XSHG', start=date, stop=date + pd.Timedelta(days=1)).pct_change(periods=5, fill_method=None).squeeze().dropna(how='all')
        amount = quotes_min.read('total_turnover',start=date, stop=date + pd.Timedelta(days=1)).rolling(5).sum().dropna(how='all')

        X = sm.add_constant(market_ret)
        y = ret
        model = sm.OLS(y, X).fit()
        epsilon = model.resid

        res = (epsilon.abs()/amount).mean()
        res = res.replace([np.inf, -np.inf], np.nan)
        res.index.name = 'order_book_id'
        res.name = date
        return res
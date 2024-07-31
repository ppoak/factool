import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, quotes_min, industry_info, 
    zscore, BaseFactor
)

    
class MomentumFactor(BaseFactor):

    def get_nonrecent_momentum(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 525)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret  = np.log(1 + (price * _adj).pct_change(fill_method=None)
            ).tail(525).sort_index().ewm(halflife=126, adjust=False).mean()
        res = stock_ret.head(504).sum()
        res.name = date
        return res
    
    def get_momentum_acceleration(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        recent_ret = (1 + (price * _adj).pct_change(fill_method=None)).tail(126).cumprod()[-1:].squeeze()
        past_ret = (1 + (price * _adj).pct_change(fill_method=None)).head(126).cumprod()[-1:].squeeze()
        res = recent_ret - past_ret
        res.index.name = 'order_book_id'
        res.name = date
        return res

    def get_mutilag_trend(self, date: str):
        base_price = quotes_day.read("close", start=date, stop=date)
        L = [3, 5, 10, 20, 50, 100, 200]

        res = pd.Series()
        for lag in L:
            rollback = quotes_day.get_trading_days_rollback(date, lag)
            price = quotes_day.read("close", start=rollback, stop=date).rolling(lag).mean().tail(1)
            ma = (price / base_price).loc[date]
            ma.name = 'lag_{}'.format(lag)
            if res.empty:
                res = ma
            else:
                res = pd.concat([res,ma],axis=1)

        future = self.get_future(start=date, stop=date, period=21)
        future.name = 'future'
        df = pd.concat([res,future],axis=1).dropna()

        X = sm.add_constant(df.drop(columns = 'future'))
        y = df['future']

        model = sm.OLS(y, X).fit()
        res = model.params[1:]
        res.name = date
        return res

    def get_mutilag_trend_pred(self, date: str):
        beta = 0
        for i in range(0, 232, 21):
            beta += self.get_mutilag_trend(quotes_day.get_trading_days_rollback(date, i))
        beta = beta/12

        base_price = quotes_day.read("close", start=date, stop=date)
        L = [3, 5, 10, 20, 50, 100, 200]
        res = pd.Series()
        for lag in L:
            rollback = quotes_day.get_trading_days_rollback(date, lag)
            price = quotes_day.read("close", start=rollback, stop=date).rolling(lag).mean().tail(1)
            ma = (price / base_price).loc[date]
            ma.name = 'lag_{}'.format(lag)
            if res.empty:
                res = ma
            else:
                res = pd.concat([res,ma],axis=1)

        res = res.dot(beta)
        res.name = date
        return res

    # def get_multiterm_ewm_llt_resid(self, date: str): 
    #     rollback = quotes_day.get_trading_days_rollback(date, 5)
    #     price = quotes_day.read("close", start=rollback, stop=date)
    #     _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
    #     price_adj = (price * _adj).unstack().to_frame(name='price_adj')
        
    #     LLT = llt.read(start=rollback, stop=date - pd.Timedelta(days=1))
    #     df = pd.merge(LLT, price_adj, left_index=True, right_index=True)
    #     df = df.div(df['price_adj'],axis=0)

    #     # future = self.get_future(start=rollback, stop=date, period=1).unstack().to_frame(name ='future')
    #     vwap = self.read("volume_weighted_price", start=rollback, stop=date)
    #     ret = (vwap * _adj).pct_change(fill_method=None).shift(-1).unstack().to_frame(name ='ret')
    #     ret = ret.replace([np.inf, -np.inf], np.nan)
    #     df = pd.merge(df, ret, left_index=True, right_index=True)
        
    #     def perform_regression(group):
    #         group = group.dropna()
    #         X = group[['LLT_3', 'LLT_5', 'LLT_20', 'LLT_60', 'LLT_125', 'LLT_250']].droplevel(self._date_level)
    #         y = group['ret'].droplevel(self._date_level)
    #         X = sm.add_constant(X)
    #         model = sm.OLS(y, X).fit()
    #         return model.resid
        
    #     res = df.groupby(level='date').apply(perform_regression)
    #     if isinstance(res.index, pd.MultiIndex):
    #         res = pd.Series(res.unstack().mean())
    #     else:
    #         res = pd.Series(res.mean())
            
    #     res.name = date
    #     return res
    
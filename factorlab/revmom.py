import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, quotes_min, industry_info, BaseFactor
)

    
class MomentumFactor(BaseFactor):

    def get_nonrecent_momentum(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 525)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret  = np.log(1 + (price * _adj).pct_change(fill_method=None)
            ).tail(525).ewm(halflife=126, adjust=False).mean()
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

    def get_CMO(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        delta = price.diff().tail(20)
        up = delta.where(delta > 0, 0)
        down = -delta.where(delta < 0, 0)
        sum_up = up.sum()
        sum_down = down.sum()
        res = 100 * (sum_up - sum_down) / (sum_up + sum_down)
        res.name = date
        return res
    
    def get_time_series_momentum(self, date: str):
        def get_ret_month(date: str):
            rollback = quotes_day.get_trading_days_rollback(date, 21)
            price = quotes_day.read("close", start=rollback, stop=date)
            _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
            res = (price * _adj).pct_change(periods=21, fill_method=None).tail(1)
            return res
        dates = [quotes_day.get_trading_days_rollback(date, i) for i in range(0, 232, 21)]
        ret = pd.concat([get_ret_month(d) for d in dates], axis=0)
        ret = ret.sort_index(ascending=True)

        ret_adj = ret - ret.ewm(com=2, adjust=False).mean()
        std_adj = np.sqrt((ret_adj.std(axis=1)**2).ewm(com=2, adjust=False).mean())

        res = np.sign(ret_adj.head(11).sum() / 12) * ret_adj.loc[date]/std_adj.loc[date]
        res.name = date
        return res
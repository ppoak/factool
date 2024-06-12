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
            ).tail(525).sort_index(ascending=False).ewm(halflife=126, adjust=False).mean()
        res = stock_ret.tail(504).sum()
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
    
    def get_trend_fund(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 5)
        volume_5d_90 = quotes_min.read("volume", start=rollback, stop=date).quantile(0.90)
        volume = quotes_min.read("volume", start=date, stop=date+pd.Timedelta(days=1))
        price = quotes_min.read("close", start=date, stop=date+pd.Timedelta(days=1))
        volume, volume_5d_90 = volume.align(volume_5d_90, axis=1, copy=False)

        trend_fund_volume = volume[volume > volume_5d_90]
        trend_fund_price = price[~np.isnan(trend_fund_volume)]

        support_price = trend_fund_price[trend_fund_price < trend_fund_price.mean()]
        support_volume = trend_fund_volume[~np.isnan(support_price)].sum()
        resistance_price = trend_fund_price[trend_fund_price > trend_fund_price.mean()]
        resistance_volume = trend_fund_volume[~np.isnan(resistance_price)].sum()

        shares = quotes_day.read("circulation_a", start=date, stop=date).loc[date]
        res = (support_volume * resistance_volume) / shares
        res.name = date
        return res

    def get_trend_fund_20d(self, date: str):
        # res = 0
        # for i in range(0, 20, 1): 
        #     res += self.get_trend_fund(quotes_day.get_trading_days_rollback(date, i))
        # res = res/20

        # 如果因子库中有数据
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        trend_fund = self.read("trend_fund", start=rollback, stop=date).tail(20).sum()/20

        # 对市值和行业进行中性化处理
        shares = quotes_day.read("circulation_a", start=date, stop=date)
        price = quotes_day.read("close", start=date, stop=date)
        adjfactor = quotes_day.read("adjfactor", start=date, stop=date)
        marketcap = np.log(shares * price * adjfactor).loc[date]
        marketcap.name = 'marketcap'

        ind = industry_info.read('first_industry_name',start=date, stop=date).loc[date]
        ind = pd.get_dummies(ind, prefix='', prefix_sep='')
        ind = ind.select_dtypes(include=[bool]).astype(int) 

        X = sm.add_constant(pd.concat([marketcap, ind], axis=1)).dropna()
        y = trend_fund.reindex(X.index)

        model = sm.OLS(y, X).fit()
        res = model.resid
        res.name = date
        return res



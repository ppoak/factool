import numpy as np
import pandas as pd
from .base import (
    quotes_day, quotes_min, BaseFactor
)

    
class MomentumFactor(BaseFactor):

    def get_nonrecent_momentum(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 525)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret  = np.log(1 + (price * _adj).pct_change(fill_method=None)
            ).tail(525).sort_index(ascending=False).ewm(halflife=126).mean()
        res = stock_ret.tail(504).sum()
        res.name = date
        return res
    
    def get_trend_fund(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 5)
        volume_5d_90 = quotes_min.read("volume", start=rollback, stop=date).quantile(0.90)
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))

        trend_fund_volume = volume[volume > volume_5d_90]
        trend_fund_price = price[~np.isnan(trend_fund_volume)]

        support_price = trend_fund_price[trend_fund_price < trend_fund_price.mean()]
        support_volume = trend_fund_volume[~np.isnan(support_price)].sum()
        resistance_price = trend_fund_price[trend_fund_price > trend_fund_price.mean()]
        resistance_volume = trend_fund_volume[~np.isnan(resistance_price)].sum()

        shares = quotes_day.read("circulation_a", start=date, stop=date).loc[date]
        res = (support_volume * resistance_volume) / shares
        return res

    def get_trend_fund_20d(self, date: str):
        res = 0
        for i in range(0, 20, 1): 
            res += self.get_trend_fund(quotes_day.get_trading_days_rollback(date, i))
        res = res/20

        # 如果因子库中有数据
        # rollback = quotes_day.get_trading_days_rollback(date, 20)
        # res = self.read("trend_fund", start=rollback, stop=date).tail(20).sum()/20
        return res



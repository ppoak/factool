import numpy as np
import pandas as pd
from .base import (
    quotes_day, BaseFactor
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
    
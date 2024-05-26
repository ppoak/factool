import numpy as np
import pandas as pd
from .base import (
    fqtd, BaseFactor
)

    
class MomentumFactor(BaseFactor):
    def get_rstr(self, date: str): # 不包含最近的21天
        rollback = fqtd.get_trading_days_rollback(date, 525)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret  = np.log(1 + (price * _adj).pct_change(fill_method=None)).tail(525).sort_index(ascending=False).ewm(halflife=126).mean()
        res = stock_ret.tail(504).sum()
        res.name = date
        return res
    
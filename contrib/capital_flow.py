import pandas as pd
from .factor import BaseFactor


class CapitalFlow(BaseFactor):

    def get_stock_connect_stableinc(self, time: pd.Timestamp) -> pd.Series:
        rollback = self.source.get_time(time, 20)[0]
        chold = self.source.get_factor("shares_holding", begin=rollback, end=time)
        shares = self.source.get_factor("circulation_a", begin=rollback, end=time)
        per = chold / shares
        res = (per.iloc[-1] - per.iloc[0]) / per.std()
        return res

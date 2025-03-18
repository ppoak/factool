import pandas as pd
from .base import FactorManager


class CapFlowFactor(FactorManager):

    def get_stock_connect_stableinc(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        chold = stock_connect.read("shares_holding", start=rollback, stop=date)
        shares = quotes_day.read("circulation_a", start=rollback, stop=date)
        per = chold / shares
        res = (per.iloc[-1] - per.iloc[0]) / per.std()
        res.name = date
        return res

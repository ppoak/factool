import numpy as np
import pandas as pd
from .base import (
    quotes_day, financial, 
    BaseFactor,
    zscore
)

class EvaluationFactor(BaseFactor):

    def get_book_to_price(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date]

        price = quotes_day.read('close', start=date, stop=date)
        _adj = quotes_day.read('adjfactor', start=date, stop=date)
        shares = quotes_day.read('circulation_a', start=date, stop=date)
        size = (price * _adj * shares).loc[date].dropna()
        res = bv / size
        res.name = date
        return res

    def get_market_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        pe = financial.read('equity_preferred_stock', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        ld = financial.read('non_current_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        price = quotes_day.read('close', start=date, stop=date)
        _adj = quotes_day.read('adjfactor', start=date, stop=date)
        shares = quotes_day.read('circulation_a', start=date, stop=date)
        me = (price * _adj * shares).loc[date].dropna()
        res = (me + ld + pe) / me

        res.name = date
        return res

    def get_debt_to_asset(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        ta = financial.read('total_assets', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].dropna()
        td = financial.read('total_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        res = td / ta

        res.name = date
        return res
    
    def get_book_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        pe = financial.read('equity_preferred_stock', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        ld = financial.read('non_current_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        be = (bv - pe).dropna()
        res = (be + ld + pe) / be

        res.name = date
        return res

    def get_barra_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0.38 * zscore(self.get_market_leverage(date).to_frame().T) + \
            0.35 * zscore(self.get_debt_to_asset(date).to_frame().T) + \
            0.27 * zscore(self.get_book_leverage(date).to_frame().T)
        res = res.loc[date]
        res.name = date
        return res

import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, financial, 
    BaseFactor,
    zscore, wscore
)

class EvaluationFactor(BaseFactor):

    def get_book_to_price(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date]

        price = quotes_day.read('close', start=date, stop=date)
        _adj = quotes_day.read('adjfactor', start=date, stop=date)
        shares = quotes_day.read('circulation_a', start=date, stop=date)
        size = (price * _adj * shares).loc[date]
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
        me = (price * _adj * shares).loc[date]
        res = (me + ld + pe) / me
        res = res.replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def get_debt_to_asset(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        ta = financial.read('total_assets', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        td = financial.read('total_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        res = td / ta
        res = res.replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res
    
    def get_book_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        pe = financial.read('equity_preferred_stock', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        ld = financial.read('non_current_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        be = bv - pe
        res = (be + ld + pe) / be
        res = res.replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def get_compound_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        res = 0.38 * zscore(self.get_market_leverage(date).to_frame().T) + \
            0.35 * zscore(self.get_debt_to_asset(date).to_frame().T) + \
            0.27 * zscore(self.get_book_leverage(date).to_frame().T)
        res = res.loc[date]
        res.name = date
        return res

    def get_growth_asset(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        A = financial.read('total_assets',start=rollback, stop=date)
        A = A.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        res =  A.apply(lambda col: (col.dropna().iloc[-1]-col.dropna().iloc[-2])/col.dropna().iloc[-2] if col.count() >= 2 else None)
        res.name = date
        return res
    
    def get_change_inventory(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        inv = financial.read('inventory',start=rollback, stop=date)
        inv = inv.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        A = financial.read('total_assets',start=rollback, stop=date)
        A = A.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))

        avg_A = A.apply(lambda col: (col.dropna().iloc[0] + col.dropna().iloc[-1])/2  if col.count() >= 2 else None)
        avg_inv = inv.apply(lambda col: (col.dropna().iloc[-1] - col.dropna().iloc[0]) if col.count() >= 2 else None)
        res = avg_inv/avg_A
        res.name = date
        return res
    
    def get_operating_std(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=756)
        operating_rev = financial.read('operating_revenue', start=rollback, stop=date)
        operating_rev_unique = operating_rev.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        res = operating_rev_unique.apply(lambda col: (col.dropna().iloc[-1] - col.dropna().iloc[-6:].mean()) / col.dropna().iloc[-6:].std() if col.count() >= 2 else None)
        res.name = date
        return res
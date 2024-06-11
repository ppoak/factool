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

    def get_growth_pe(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        rollback_past = quotes_day.get_trading_days_rollback(date, rollback=504)
        pe = financial.read('total_equity',start=rollback, stop=date).ffill().iloc[-1]
        pe_past = financial.read('total_equity',start=rollback_past, stop=rollback).ffill().iloc[-1]
        res = (pe - pe_past)/pe_past
        res.name = date
        return res

    def get_improve_revenue(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=504)
        revenue = zscore(financial.read('revenue',start=rollback, stop=date).ffill()[-1:]).squeeze()
        expense = zscore(financial.read('total_expense',start=rollback, stop=date).ffill()[-1:]).squeeze()
        revenue = revenue.loc[expense.index]

        y = revenue.dropna()
        X = sm.add_constant(expense).dropna()
        model = sm.OLS(y, X)
        res = model.fit()
        res = res.resid
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_capital_investment(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        A_paid= financial.read('cash_paid_for_asset', start=rollback, stop=date).ffill().iloc[-1]
        A_disposal = financial.read('cash_received_from_disposal_of_asset', start=rollback, stop=date).ffill().iloc[-1]
        operating_rev = financial.read('operating_revenue', start=rollback, stop=date).ffill().iloc[-1]

        res = (A_paid - A_disposal) / operating_rev
        res = res.replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res
    
    def get_capital_investment_ratio(self, date: str | pd.Timestamp) -> pd.Series:
        ce = self.get_capital_investment(date)
        res = 0
        for i in range(252, 252*4, 252): 
            res += self.get_capital_investment(quotes_day.get_trading_days_rollback(date, i)).fillna(0)
    
        res = (ce / (res/3)) -1
        res = res.replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, financial,
    industry_info, BaseFactor,
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

    def get_1quarter_growth_asset(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        A = financial.read('total_assets',start=rollback, stop=date)
        A = A.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        res =  A.apply(lambda col: (col.dropna().iloc[-1]-col.dropna().iloc[-2])/col.dropna().iloc[-2] if col.count() >= 2 else None)
        res.name = date
        return res
    
    def get_1year_asset_based_change_inventory(self, date: str | pd.Timestamp) -> pd.Series:
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
    
    def get_6quarter_operating_std(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=756)
        operating_rev = financial.read('operating_revenue', start=rollback, stop=date)
        operating_rev_unique = operating_rev.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        res = operating_rev_unique.apply(lambda col: (col.dropna().iloc[-1] - col.dropna().iloc[-6:].mean()) / col.dropna().iloc[-6:].std() if col.count() >= 2 else None)
        res.name = date
        return res

    def get_nonoperating_surplus(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        TA = financial.read('net_profit', start=rollback, stop=date).ffill().tail(1).squeeze()

        A = financial.read('total_assets',start=rollback, stop=date)
        A = A.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        last_A = A.apply(lambda col: (col.dropna().iloc[0]) if col.count() >= 2 else None)

        rev = financial.read('revenue',start=rollback, stop=date)
        rev = rev.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        delta_rev = rev.apply(lambda col: (col.dropna().iloc[-1] - col.dropna().iloc[0]) if col.count() >= 2 else None)
        
        ppe = financial.read('total_fixed_assets',start=rollback, stop=date)
        ppe = ppe.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        curr_ppe = ppe.apply(lambda col: (col.dropna().iloc[-1]) if col.count() >= 2 else None)
        
        coefficients = {}
        ind = industry_info.read('first_industry_name',start=rollback, stop=rollback).squeeze()

        common_index = last_A.index.intersection(ind.index)
        ind = ind[common_index]
        last_A = last_A.loc[common_index]
        delta_rev = delta_rev.loc[common_index]
        curr_ppe = curr_ppe.loc[common_index]
        TA = TA.loc[common_index]

        for i in ind.unique():
            ind_mask = (ind == i)
            df = pd.concat([1/last_A[ind_mask], delta_rev[ind_mask]/last_A[ind_mask], curr_ppe[ind_mask]/last_A[ind_mask], TA[ind_mask]/last_A[ind_mask]], axis=1).dropna()
            df.columns = ['Feature1', 'Feature2', 'Feature3', 'Target']

            if df.shape[0] > 0:
                X = df[['Feature1', 'Feature2', 'Feature3']]
                y = df['Target']
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                coefficients[i] = model.params[1:]

        AR = financial.read('net_accts_receivable',start=rollback, stop=date)
        AR = AR.apply(lambda col: col.dropna().drop_duplicates().reset_index(drop=True))
        delta_AR = AR.apply(lambda col: (col.dropna().iloc[-1] - col.dropna().iloc[0]) if col.count() >= 2 else None)
        delta_AR = delta_AR.loc[common_index]

        res = pd.Series(index=common_index)
        for i in ind.unique():
            ind_mask = (ind == i)
            if i in coefficients:
                df_new = pd.concat([1/last_A[ind_mask], (delta_rev[ind_mask]-delta_AR[ind_mask])/last_A[ind_mask], curr_ppe[ind_mask]/last_A[ind_mask]], axis=1)
                df_new.columns = ['Feature1', 'Feature2', 'Feature3']
                res[ind_mask] = np.sum(df_new * coefficients[i], axis=1)
                
        res.name = date
        return res
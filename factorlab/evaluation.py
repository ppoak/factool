import pandas as pd
import numpy as np
from .base import (
    quotes_day, financial, BaseFactor
)

class EvaluationFactor(BaseFactor):

    def get_book_to_price(self, date: str | pd.Timestamp) -> pd.Series:
        # 当前公司的账面价值
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date]

        # 当前公司的市值
        price = quotes_day.read('close', start=date, stop=date)
        _adj = quotes_day.read('adjfactor', start=date, stop=date)
        shares = quotes_day.read('circulation_a', start=date, stop=date)
        size = (price * _adj * shares).loc[date]
        res = bv / size
        res.name = date
        return res

    def get_mlev(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        #优先股账面价值
        pe = financial.read('equity_preferred_stock', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        #长期负债账面价值
        ld = financial.read('non_current_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        #普通股市值
        price = quotes_day.read('close', start=date, stop=date)
        _adj = quotes_day.read('adjfactor', start=date, stop=date)
        shares = quotes_day.read('circulation_a', start=date, stop=date)
        me = (price * _adj * shares).loc[date].fillna(0)
        res = (me + ld + pe)/ me
        res = self.standardize(res, date)
        return res * 0.38

    def get_dtoa(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        #总资产账面价值
        ta = financial.read('total_assets', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        #总负债账面价值
        td = financial.read('total_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        res = td / ta
        res = self.standardize(res, date)
        return res * 0.35
    
    def get_blev(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, rollback=252)
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)

        #优先股账面价值
        pe = financial.read('equity_preferred_stock', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        #长期负债账面价值
        ld = financial.read('non_current_liabilities', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)

        #普通股账面价值
        bv = financial.read('total_equity', start=rollback, stop=date).reindex(trading_days).ffill().loc[date].fillna(0)
        be = bv - pe
        res = (be + ld + pe) / be
        res = self.standardize(res, date)
        return res * 0.27

    def get_barra_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        res = self.get_mlev(date) + self.get_blev(date) + self.get_dtoa(date)
        res.name = date
        return res

    def get_stom(self, date: str | pd.Timestamp) -> pd.Series:
        #月换手率
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        turnover = quotes_day.read("turnover", start=rollback, stop=date).tail(21).sum()
        res = np.log(turnover + 1e-6)
        # res = self.standardize(res, date)
        return res * 0.35
    
    def get_stoq(self, date: str | pd.Timestamp) -> pd.Series:
        #季度换手率均值
        rollback = quotes_day.get_trading_days_rollback(date, 63)
        turnover = quotes_day.read("turnover", start=rollback, stop=date).tail(63).sum()
        res = np.log(turnover/3 + 1e-6)
        # res = self.standardize(res, date)
        return res * 0.35
    
    def get_stoa(self, date: str | pd.Timestamp) -> pd.Series:
        #年换手率均值
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        turnover = quotes_day.read("turnover", start=rollback, stop=date).tail(252).sum()
        res = np.log(turnover/12 + 1e-6)
        # res = self.standardize(res, date)
        return res * 0.3 

    def get_liquidity(self, date: pd.Timestamp) -> pd.DataFrame:
        res = self.get_stom(date) + self.get_stoq(date) + self.get_stoa(date)
        res.name = date
        return res
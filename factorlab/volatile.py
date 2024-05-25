import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    fqtd, fqtm, fidxwgt, fidxqtd, ffin, BaseFactor
)


class VolatileFactor(BaseFactor):

    def standardize(self, data: pd.Series ,date: pd.Timestamp) -> pd.Series:
        weight = fidxwgt.read('000985.XSHG',start=date, stop=date).loc[date]
        mean = np.sum(weight * data)
        std = data.std()
        res = (data - mean) / std
        return res

    def get_information_distribution_uniformity(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 20)
        price = fqtm.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.groupby(price.index.date).pct_change(fill_method=None)
        std = ret.groupby(ret.index.date).std()
        res = std.std() / std.mean()
        res.name = date
        return res

    def get_dastd(self, date: str):
        #过去252个交易日日超额收益率波动率
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj = fqtd.read("adjfactor", start=rollback, stop=date)

        #rf=0,用stock_ret代替excess_ret
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252) 
        res = np.sqrt(((stock_ret - stock_ret.mean()) ** 2).sort_index(ascending=False).ewm(halflife=42).mean().sum())
        return res * 0.74
    
    def get_cmra(self, date: str):
        #过去12个月超额收益的离差
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj = fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = np.log(1 + (price * _adj).pct_change(fill_method=None)).tail(252)

        date_intervals = pd.cut(stock_ret.index, bins=12, labels=False)
        zt = (1 + stock_ret).groupby(date_intervals).prod() #compounded
        zt_max = zt.loc[zt.sum(axis=1).idxmax(),:]
        zt_min = zt.loc[zt.sum(axis=1).idxmin(),:]
        res = np.log(zt_max) - np.log(zt_min)
        return res * 0.16
    
    def get_hsigma(self, date: str):
        #beta的残差波动率
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63).mean()
        market_ret = fidxqtd.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63).mean().loc[:,'000985.XSHG']

        X = sm.add_constant(market_ret)
        Y = stock_ret
        model = sm.OLS(Y, X).fit()
        res = model.resid.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res * 0.1

    def get_residual_volatility(self, date: str):
        hsigma = self.standardize(self.get_hsigma(date),date)
        cmra = self.standardize(self.get_cmra(date),date)
        dastd = self.standardize(self.get_dastd(date),date)
        res = dastd + cmra + hsigma
        res.name = date
        return res

    def get_beta(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63).mean()
        market_ret = fidxqtd.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63).mean().loc[:,'000985.XSHG']

        res = {}
        var = np.var(market_ret)
        for code in stock_ret.columns:
            covr = np.cov(stock_ret[code],market_ret)[0][1]
            result = covr/var
            res[code] = result

        res = pd.Series(res)
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_fright_degree(self,date: str):
        #惊恐度
        rollback = fqtd.get_trading_days_rollback(date, 20)
        market_ret = fidxqtd.read('close',start=rollback, stop=date).loc[:,'000985.XSHG'].ffill().pct_change().tail(20)  #中证全指代表市场收益率
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).ffill().pct_change().tail(20)
        deviation = (stock_ret.subtract(market_ret, axis=0)).abs()
        stand = (stock_ret.abs()).add(market_ret.abs(), axis=0)+ 0.1
        fright = deviation/stand
        
        #日内标准差
        df = fqtm.read('close',start = rollback, stop = date + pd.Timedelta(days=1))
        daily_std = df.groupby(df.index.date).std().tail(20)

        #加权调整收益率
        adj_ret = fright * daily_std * stock_ret

        fright_ret = adj_ret.mean()
        fright_vol = adj_ret.std()

        res = (fright_ret + fright_vol)/2
        res.name = date
        return res

    def get_std_4m(self,date: str):
        #近4个月日收益率序列的标准差
        rollback = fqtd.get_trading_days_rollback(date, 88)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).ffill().pct_change().tail(88)
        res = stock_ret.std()
        res.name = date
        return res
    
    def get_capm_std_3m(self,date: str):
        #近3个月内CAPM回归残差
        rollback = fqtd.get_trading_days_rollback(date, 66)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(66)
        market_ret = fidxqtd.read('close',code='000001.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(66)  #上证综指代表市场收益率
        X = sm.add_constant(market_ret)
        Y = stock_ret
        model = sm.OLS(Y, X).fit()

        # 计算残差
        res = model.resid.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_ff3_3m(self,date: str):
        #近3个月Fama-French三因子回归残差的标准差
        rollback = fqtd.get_trading_days_rollback(date, 66)
        price = fqtd.read("close", start=rollback, stop=date)
        _adj= fqtd.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).ffill().pct_change().tail(66)
        market_ret = fidxqtd.read('close',start=rollback, stop=date).loc[:,'000001.XSHG'].ffill().pct_change().tail(66)  #上证综指代表市场收益率
        
        #市值因子
        shares = fqtd.read("circulation_a", start=rollback, stop=date)
        market_cap = (shares * price * _adj).tail(66)
        sorted_market_cap = market_cap.mean().sort_values()  #将序列股票分为2组
        midpoint = len(sorted_market_cap) // 2
        S = sorted_market_cap.index[:midpoint]
        B = sorted_market_cap.index[midpoint:]

        # 规模因子
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        bv = ffin.read('total_equity', start=rollback, stop=date)
        bv = bv.reindex(trading_days).ffill().bfill().tail(66)
        bm = bv / market_cap
        sorted_bm = bm.mean().sort_values()
        p30 = sorted_bm.quantile(0.3)
        p70 = sorted_bm.quantile(0.7)
        L = sorted_bm.index[sorted_bm <= p30]
        M = sorted_bm.index[(sorted_bm > p30) & (sorted_bm <= p70)]
        H = sorted_bm.index[sorted_bm > p70]

        #将截面股票分为6组
        SL = stock_ret[S.intersection(L)].mean(axis=1) #交集
        SM = stock_ret[S.intersection(M)].mean(axis=1)
        SH = stock_ret[S.intersection(H)].mean(axis=1)

        BL = stock_ret[B.intersection(L)].mean(axis=1)
        BM = stock_ret[B.intersection(M)].mean(axis=1)
        BH = stock_ret[B.intersection(H)].mean(axis=1)

        SMB = (SL + SM + SH) / 3 - (BL + BM + BH) / 3
        HML = (SH + BH) / 2 - (SL + BL) / 2

        # 计算三因子回归残差
        X = sm.add_constant(pd.DataFrame({'Market_Return': market_ret, 'SMB': SMB, 'HML': HML}).fillna(0))
        Y = stock_ret
        model = sm.OLS(Y, X).fit()
        res = model.resid
        return res
    
    def get_ff3_std_3m(self,date: str):
        res = self.get_ff3_3m(date) 
        res = res.std()
        res.name = date
        return res

    def get_ff3_std_up_3m(self,date: str):
        res = self.get_ff3_3m(date) 
        res = res[res > 0]
        res = res.std()
        res.name = date
        return res
    
    def get_ff3_std_down_3m(self,date: str):
        res = self.get_ff3_3m(date) 
        res = res[res < 0]
        res = res.std()
        res.name = date
        return res
    
    def get_ff3_std_ud_3m(self,date: str):
        down = self.get_ff3_std_down_3m(date) 
        up = self.get_ff3_std_up_3m(date)
        res = down + up
        res.name = date
        return res
    
    def get_rise_std_4m(self,date: str):
        #近4个月日内最大涨幅波动率
        rollback = fqtd.get_trading_days_rollback(date, 88)
        df = fqtm.read("high, open", start=rollback, stop=date).tail(88)
        max = df.groupby('date')['high'].max()
        ben = df.groupby('date')['open'].first()
        rise = (max - ben) / ben
        res = rise.std()
        res.name = date
        return res
    
    def get_rise_fall_std_5m(self,date: str):
        #近5个月日内最大涨幅波动率减去日内最大跌幅波动率
        rollback = fqtd.get_trading_days_rollback(date, 110).tail(110)
        df = fqtm.read("high, low, open", start=rollback, stop=date)
        max = df.groupby('date')['high'].max()
        low = df.groupby('date')['low'].min()
        diff = (max - low) / low
        res = diff.std()
        res.name = date
        return res

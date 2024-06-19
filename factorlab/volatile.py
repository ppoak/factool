import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, quotes_min, index_weights, 
    index_quotes_day, financial, 
    BaseFactor,
    zscore,
)


class VolatileFactor(BaseFactor):

    def get_information_distribution_uniformity(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_min.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.groupby(price.index.date).pct_change(fill_method=None)
        std = ret.groupby(ret.index.date).std()
        res = std.std() / std.mean()
        res.name = date
        return res

    def get_daily_return_volatility(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)

        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252) 
        res = np.sqrt(((stock_ret - stock_ret.mean()) ** 2).sort_index(ascending=False).ewm(halflife=42, adjust=False).mean().sum())
        res.name = date
        return res
    
    def get_daily_return_diviation(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = np.log((1 + (price * _adj).pct_change(fill_method=None)).clip(lower=1e-10)).tail(252)

        zt = stock_ret.groupby(stock_ret.index.month).sum()
        res = np.log((1 + zt.max()).clip(lower=1e-10)) - np.log((1 + zt.min()).clip(lower=1e-10))
        res.name = date
        return res
    
    def get_beta_return_residual(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63, adjust=False).mean()
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(252).sort_index(ascending=False).ewm(halflife=63, adjust=False).mean().loc[:,'000985.XSHG']
        beta = self.read("market_beta", start=rollback, stop=date)
        
        res = (stock_ret - beta.mul(market_ret, axis=0)).std()
        res.name = date
        return res

    def get_residual_volatility(self, date: str) -> pd.Series:
        res = zscore(self.get_beta_return_residual(date).to_frame().T) * 0.1 + \
            zscore(self.get_daily_return_diviation(date).to_frame().T) * 0.16 + \
            zscore(self.get_daily_return_volatility(date).to_frame().T) * 0.74
        res = res.loc[date]
        res.name = date
        return res

    def get_market_beta(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252).sort_index(ascending=False)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(252).sort_index(ascending=False).loc[:,'000985.XSHG']
    
        # res = {}
        # var = np.var(market_ret, ddof=1)
        # for code in stock_ret.columns:
        #     covr = np.cov(stock_ret[code],market_ret)[0][1]
        #     result = covr/var
        #     res[code] = result
        # res = pd.Series(res)

        h = 63
        alpha = 1 - np.exp(np.log(0.5) / h)
        weights = pd.Series([alpha * (1 - alpha)**t for t in range(252)][::-1], index=stock_ret.index)
        weights /=  np.sum(weights)

        X = sm.add_constant(market_ret)
        y = stock_ret
        model = sm.WLS(y, X, weights=weights).fit()
        res = model.params[1:]
        res = res.squeeze()
        res.index = y.columns

        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_downside_market_beta(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(252)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(252).loc[:,'000985.XSHG']
    
        market_ret = market_ret[market_ret < 0]
        stock_ret = stock_ret.loc[market_ret.index]

        res = {}
        var = np.var(market_ret, ddof=1)
        for code in stock_ret.columns:
            covr = np.cov(stock_ret[code],market_ret)[0][1]
            result = covr/var
            res[code] = result
        res = pd.Series(res)

        res.index.name = 'order_book_id'
        res.name = date
        return res

    def get_fright_degree(self, date: str) -> pd.Series:
        #惊恐度
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(20).loc[:,'000985.XSHG']  #中证全指代表市场收益率
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(20)
        deviation = (stock_ret.subtract(market_ret, axis=0)).abs()
        stand = (stock_ret.abs()).add(market_ret.abs(), axis=0)+ 0.1
        fright = deviation/stand
        
        #日内标准差
        df = quotes_min.read('close',start = rollback, stop = date + pd.Timedelta(days=1))
        daily_std = df.groupby(df.index.date).std().tail(20)

        #加权调整收益率
        adj_ret = fright * daily_std * stock_ret

        fright_ret = adj_ret.mean()
        fright_vol = adj_ret.std()

        res = (fright_ret + fright_vol)/2
        res.name = date
        return res

    def get_std_4m(self, date: str) -> pd.Series:
        #近4个月日收益率序列的标准差
        rollback = quotes_day.get_trading_days_rollback(date, 84)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(84)
        res = stock_ret.std()
        res.name = date
        return res
    
    def get_capm_std_3m(self, date: str) -> pd.Series:
        #近3个月内CAPM回归残差
        rollback = quotes_day.get_trading_days_rollback(date, 63)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(63)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(63)
        X = sm.add_constant(market_ret)
        y = stock_ret
        model = sm.OLS(y, X).fit()

        # 计算残差
        res = model.resid.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_ff3_3m(self, date: str) -> pd.Series:
        #近3个月Fama-French三因子回归残差的标准差
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(63)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(63).loc[:,'000985.XSHG']
        
        #市值因子
        shares = quotes_day.read("circulation_a", start=rollback, stop=date)
        market_cap = (shares * price * _adj).tail(63)
        sorted_market_cap = market_cap.mean().sort_values()  #将序列股票分为2组
        midpoint = len(sorted_market_cap) // 2
        S = sorted_market_cap.index[:midpoint]
        B = sorted_market_cap.index[midpoint:]

        # 规模因子
        trading_days = quotes_day.get_trading_days(start=rollback, stop=date)
        bv = financial.read('total_equity', start=rollback, stop=date)
        bv = bv.reindex(trading_days).ffill().bfill().tail(63)
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
        X = sm.add_constant(pd.DataFrame({'Market_Return': market_ret, 'SMB': SMB, 'HML': HML})).dropna()
        y = stock_ret
        model = sm.OLS(y, X).fit()
        res = model.resid
        return res
    
    def get_ff3_std_3m(self, date: str) -> pd.Series:
        res = self.get_ff3_3m(date) 
        res = res.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res

    def get_ff3_std_up_3m(self, date: str) -> pd.Series:
        res = self.get_ff3_3m(date) 
        res = res[res > 0]
        res = res.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_ff3_std_down_3m(self, date: str) -> pd.Series:
        res = self.get_ff3_3m(date) 
        res = res[res < 0]
        res = res.std()
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_ff3_std_ud_3m(self, date: str) -> pd.Series:
        down = self.get_ff3_std_down_3m(date) 
        up = self.get_ff3_std_up_3m(date)
        res = down + up
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_rise_std_4m(self, date: str) -> pd.Series:
        #近4个月日内最大涨幅波动率
        rollback = quotes_day.get_trading_days_rollback(date, 84)
        high = quotes_min.read("high", start=rollback, stop=date)
        low = quotes_min.read("low", start=rollback, stop=date)
        max = high.groupby(high.index.date).max().tail(84)
        min = low.groupby(low.index.date).min().tail(84)
        rise = (max - min) / min
        res = rise.std()
        res.name = date
        return res
    
    def get_rise_fall_std_5m(self, date: str) -> pd.Series:
        #近5个月日内最大涨幅减去日内最大跌幅波动率
        rollback = quotes_day.get_trading_days_rollback(date, 105)
        high = quotes_min.read("high", start=rollback, stop=date)
        low = quotes_min.read("low", start=rollback, stop=date)
        max = high.groupby(high.index.date).max().tail(105)
        min = low.groupby(low.index.date).min().tail(105)
        rise = (max - min) / min
        down = (max - min) / max
        diff = rise - down
        res = diff.std()
        res.name = date
        return res
    
    def get_capm_coskewness(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 126)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(126)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).tail(126).loc[:,'000985.XSHG']
        X = sm.add_constant(market_ret)
        y = stock_ret
        model = sm.OLS(y, X).fit()

        epsilon_i = model.resid
        epsilon_m = market_ret - market_ret.mean()
        res = np.mean(epsilon_i.mul(epsilon_m**2, axis=0),axis=0) / (np.sqrt((epsilon_i**2).mean()) * (epsilon_m**2).mean())
        res.index.name = 'order_book_id'
        res.name = date
        return res
    
    def get_negative_coskewness(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(20)
        deviations_cubed = np.sum((stock_ret.sub(stock_ret.mean(axis=1),axis=0)) **3, axis=0)
        deviations_squared = np.sum((stock_ret.sub(stock_ret.mean(axis=1),axis=0)) **2, axis=0)
        
        n = 20
        numerator = -n * (n - 1) ** 1.5 * deviations_cubed
        denominator = (n - 1) * (n - 2) * (deviations_squared ** 1.5)
        res = numerator/denominator
        res.name = date
        return res

    def get_down_up_std_ratio(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 60)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj= quotes_day.read("adjfactor", start=rollback, stop=date)
        stock_ret = (price * _adj).pct_change(fill_method=None).tail(60)
        cum_return = np.expm1(np.log1p(stock_ret).sum()).mean()

        stock_ret_down = stock_ret[stock_ret < cum_return]
        stock_ret_up = stock_ret[stock_ret > cum_return]
        nd = stock_ret_down.notna().sum() - 1
        nu = stock_ret_up.notna().sum() - 1

        res = (nd * (((stock_ret_down - cum_return)**2).sum())) / (nu * ((stock_ret_up - cum_return)**2).sum())
        res = np.log(res.replace([np.inf, -np.inf], np.nan).dropna() + 1e-10)
        res.name = date
        return res
    
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    quotes_day, quotes_min, industry_info, 
    wscore, neutralization, BaseFactor
)


class DeraPriceFactor(BaseFactor):

    def get_volume_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res
    
    def get_time_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        res = p.mean()
        res.name = date
        return res
    
    def get_tail_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        p = p.between_time("14:30", "15:00")
        vol = vol.between_time("14:30", "15:00")
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res

    def get_head_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        p = p.between_time("9:30", "10:00")
        vol = vol.between_time("9:30", "10:00")
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res


class PriceVolumeCorr(BaseFactor):

    def get_smart_money_ratio(self, date: pd.Timestamp) -> pd.DataFrame:
        rollback = self.get_trading_days_rollback(date, 9)
        price = quotes_min.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None).abs()
        vol = quotes_min.read("volume", start=rollback, stop=date + pd.Timedelta(days=1))
        retvol = ret / (vol ** 0.25)
        rank = retvol.rank(axis=0, ascending=False)
        rank = rank.le(retvol.count() // 5, axis=1)
        retvol = vol.where(rank)
        res = ((retvol * price).sum() / retvol.sum()) / ((vol * price).sum() / vol.sum())
        res.name = date
        return res

    def get_price_volume_corr(self, date: pd.Timestamp) -> pd.DataFrame:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        res = price.corrwith(volume, axis=0).replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def get_average_relative_price_percent(self, date: pd.Timestamp) -> pd.DataFrame:
        df = quotes_min.read("open, high, low, close", start=date, stop=date + pd.Timedelta(days=1))
        twap = df.mean(axis=1).groupby(level=quotes_min._code_level).mean()
        high = df["high"].groupby(level=quotes_min._code_level).max()
        low = df["low"].groupby(level=quotes_min._code_level).min()
        arrp = (twap - low) / (high - low)
        arrp.name = date
        return arrp
    
    def get_compound_volume_first(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        dp = quotes_day.read("close", start=rollback, stop=date).diff(1).iloc[1:].tail(20)
        dv = quotes_day.read("volume", start=rollback, stop=date).diff(1).iloc[1:].head(20)
        dv.index = dp.index
        # 原始
        dV_dP_Corr = dp.corrwith(dv, axis=0)
        dV_dP_Corr.name = date
        dV_dP_Corr = wscore(dV_dP_Corr.to_frame().T, date).loc[date]
        
        dp_p = dp[dp > 0]
        dv_p = dv[dv > 0]
        dV_dP_Corr_pp = dp_p.corrwith(dv_p, axis=0)
        dV_dP_Corr_pp.name = date
        dV_dP_Corr_pp = wscore(dV_dP_Corr_pp.to_frame().T, date).loc[date]

        dp_n = dp[dp < 0]
        dv_n = dv[dv < 0]
        dV_dP_Corr_nn = dp_n.corrwith(dv_n, axis=0)
        dV_dP_Corr_nn.name = date
        dV_dP_Corr_nn = wscore(dV_dP_Corr_nn.to_frame().T, date).loc[date]

        dV_dP_Corr_np = dp_p.corrwith(dv_n, axis=0)
        dV_dP_Corr_np.name = date
        dV_dP_Corr_np = wscore(dV_dP_Corr_np.to_frame().T, date).loc[date]

        dV_dP_Corr_pn = dp_n.corrwith(dv_p, axis=0)
        dV_dP_Corr_pn.name = date
        dV_dP_Corr_pn = wscore(dV_dP_Corr_pn.to_frame().T, date).loc[date]        

        # 复合量先价行
        res = dV_dP_Corr_pp - dV_dP_Corr_pn - dV_dP_Corr_np + dV_dP_Corr_nn
        return res
    
    def get_trend_fund(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 5)
        volume_5d_90 = quotes_min.read("volume", start=rollback, stop=date).quantile(0.90)
        volume = quotes_min.read("volume", start=date, stop=date+pd.Timedelta(days=1))
        price = quotes_min.read("close", start=date, stop=date+pd.Timedelta(days=1))
        volume, volume_5d_90 = volume.align(volume_5d_90, axis=1, copy=False)

        trend_fund_volume = volume[volume > volume_5d_90]
        trend_fund_price = price[~np.isnan(trend_fund_volume)]

        support_price = trend_fund_price[trend_fund_price < trend_fund_price.mean()]
        support_volume = trend_fund_volume[~np.isnan(support_price)].sum()
        resistance_price = trend_fund_price[trend_fund_price > trend_fund_price.mean()]
        resistance_volume = trend_fund_volume[~np.isnan(resistance_price)].sum()

        shares = quotes_day.read("circulation_a", start=date, stop=date).loc[date]
        res = (support_volume * resistance_volume) / shares
        res.name = date
        return res

    def get_trend_fund_20d(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        res = self.read("trend_fund", start=rollback, stop=date).tail(20).sum()/20
        res.name = date
        # res = res.to_frame().T
        # neutralization(res,True,True)
        return res

    def get_price_spread(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(252)
        distance_matrix = 1- ret.corr(method='pearson')

        nearest_stocks = {}
        for stock in distance_matrix.columns:
            nearest_stocks[stock] = distance_matrix[stock].sort_values(ascending=True)[1:11].index.tolist()

        reference_prices = {}
        for stock in nearest_stocks.keys():
            portfolio_returns = ret[nearest_stocks[stock]].mean(axis=1)
            reference_prices[stock] = (1 + portfolio_returns).cumprod().iloc[-1]

        reference_prices = pd.Series(reference_prices)
        res = np.log(price.loc[date]) - np.log(reference_prices)
        res.name = date
        return res
    
    def get_spread_bias(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 60)
        price_spread = self.read("price_spread", start=rollback, stop=date)
        res = (price_spread.loc[date] - price_spread.mean()) / price_spread.std()
        res.name = date
        return res
    
    def get_20d_rsi(self, date: str): 
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        up = ret[ret>0].mean()
        down = ret[ret<0].mean().abs()
        res = up/(up+down)
        res.name = date
        return res
    
    def get_minute_rsi(self, date: str): 
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        up = ret[ret>0].mean()
        down = ret[ret<0].mean().abs()
        res = up/(up+down)
        res.name = date
        return res

    def get_5minute_rsi(self, date: str): 
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(periods=5, fill_method=None)
        up = ret[ret>0].mean()
        down = ret[ret<0].mean().abs()
        res = up/(up+down)
        res.name = date
        return res
    
    def get_minute_rsi_rolling_mean20d(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        res = self.read('minute_rsi', start=rollback, stop=date).tail(20).mean()
        res.name = date
        return res

    def get_5minute_rsi_rolling_mean20d(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        res = self.read('5minute_rsi', start=rollback, stop=date).tail(20).mean()
        res.name = date
        return res
    
    def get_minute_rsi_weighted_by_20d_turnover(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        rsi = self.read('minute_rsi', start=rollback, stop=date).tail(20)
        volume = quotes_day.read("volume", start=rollback, stop=date)
        shares = quotes_day.read("circulation_a", start=rollback, stop=date)
        turnover = (volume / shares).tail(20)
        weight = turnover/turnover.sum()
        res = (weight * rsi).sum()
        res.name = date
        return res
    
    def gets_20d_maxret(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        ret = price.pct_change(fill_method=None).tail(20)
        res = ret.max()
        res.index.name = 'date' 
        return res
    
    def get_20d_top10_cum_ret(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        ret = price.pct_change(fill_method=None).tail(20)
        res = ret.apply(lambda x: x.nlargest(10).sum(), axis=0)
        res.name = 'date' 
        return res

    def get_20d_max_truerange(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        prev_close = quotes_day.read("close", start=rollback, stop=date).shift(1)
        high = quotes_day.read("high", start=rollback, stop=date)
        low = quotes_day.read("low", start=rollback, stop=date)
        res = np.maximum.reduce([(high - low)/prev_close, abs(high - prev_close)/prev_close,  abs(prev_close - low)/prev_close]) # 每行最大值
        res = pd.DataFrame(res, index=high.index, columns=high.columns).tail(20).max()
        res.name = 'date' 
        return res
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .datasource import quotes_min, quotes_day
from .factor import Factor


class DeraPriceFactor(Factor):

    def get_volume_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read(
            index="datetime", columns="code", pivot="close", 
            datetime__ge=date, datetime__le=date + pd.Timedelta(days=1)
        )
        v = quotes_min.read(
            index="datetime", columns="code", pivot="volume", 
            datetime__ge=date, datetime__le=date + pd.Timedelta(days=1)
        )
        w = v / v.sum()
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


class PriceVolumeCorr(Factor):

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
    
    def get_20d_maxret(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        res = ret.max()
        res.name = date
        return res
    
    def get_20d_top10_cum_ret(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        res = ret.apply(lambda x: x.nlargest(10).sum(), axis=0)
        res.name = date
        return res

    def get_20d_max_truerange(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        prev_close = (quotes_day.read("close", start=rollback, stop=date) * _adj).shift(1)
        high = (quotes_day.read("high", start=rollback, stop=date) * _adj)
        low = (quotes_day.read("low", start=rollback, stop=date) * _adj)
        res = np.maximum.reduce([(high - low)/prev_close, abs(high - prev_close)/prev_close,  abs(prev_close - low)/prev_close]) # 每行最大值
        res = pd.DataFrame(res, index=high.index, columns=high.columns).tail(20).max()
        res.name = date
        return res

    def get_industry_co_20d_maxret(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        max_ret = ret.dropna(axis=1, how='all').idxmax()

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        common_idx = ind.index.intersection(max_ret.index)
        max_ret = max_ret.loc[common_idx]
        
        res = pd.DataFrame({
            'order_book_id': max_ret.index,
            'date': max_ret.values,
            'industry': max_ret.index.map(ind)
        })
 
        def get_industry_return(row):
            date = row['date']
            industry = row['industry']
            if date in ind_ret.index:
                return ind_ret.at[date, industry]
            else:
                return np.nan
        res['industry_return'] = res.apply(get_industry_return, axis=1)

        res = res.set_index('order_book_id')['industry_return']
        res.name = date
        return res
    
    def get_industry_co_20d_top5_cum_ret(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        max_ret = ret.apply(lambda x: x.nlargest(5).index, axis=0).T

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        common_idx = ind.index.intersection(max_ret.index)
        max_ret = max_ret.loc[common_idx]
        max_ret['industry'] = max_ret.index.map(ind)

        def get_industry_return(date, industry):
            try:
                return ind_ret.at[date, industry]
            except KeyError:
                return pd.NA 
        res = max_ret.apply(lambda row: row[:-1].apply(lambda date: get_industry_return(date, row['industry'])), axis=1).sum(axis=1)
        res.index.name = 'order_book_id' 
        res.name = date
        return res
    
    def get_industry_co_20d_max_truerange(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        prev_close = (quotes_day.read("close", start=rollback, stop=date) * _adj).shift(1)
        high = quotes_day.read("high", start=rollback, stop=date) * _adj
        low = quotes_day.read("low", start=rollback, stop=date) * _adj
        tr = np.maximum.reduce([(high - low)/prev_close, abs(high - prev_close)/prev_close,  abs(prev_close - low)/prev_close]) # 每行最大值
        tr = pd.DataFrame(tr, index=high.index, columns=high.columns).tail(20)
        max_tr = tr.dropna(axis=1, how='all').idxmax()

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        common_idx = ind.index.intersection(max_tr.index)
        max_tr = max_tr.loc[common_idx]
        
        res = pd.DataFrame({
            'order_book_id': max_tr.index,
            'date': max_tr.values,
            'industry': max_tr.index.map(ind)
        })
 
        def get_industry_return(row):
            date = row['date']
            industry = row['industry']
            if date in ind_ret.index:
                return ind_ret.at[date, industry]
            else:
                return np.nan
        res['industry_return'] = res.apply(get_industry_return, axis=1)

        res = res.set_index('order_book_id')['industry_return']
        res.name = date
        return res
    
    def get_industry_co_20d_top5_cumret_weighted(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        max_ret = ret.apply(lambda x: x.nlargest(5).index, axis=0).T

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        # ind = barra.read(start = date, stop = date)[ind_columns].idxmax(axis=1).reset_index(level='date', drop=True)
        common_idx = ind.index.intersection(max_ret.index)
        max_ret = max_ret.loc[common_idx]
        max_ret['industry'] = max_ret.index.map(ind)

        def get_industry_return(date, industry):
            try:
                return ind_ret.at[date, industry]
            except KeyError:
                return pd.NA 
        res = max_ret.apply(lambda row: row[:-1].apply(lambda date: get_industry_return(date, row['industry'])), axis=1)
        n = 5
        weights = np.array([2 ** -((i - 1) / (n - 1)) for i in range(1, n + 1)])    
        res = res.multiply(weights).sum(axis=1)
        res.name = date
        return res
    
    def get_industry_co_20d_top5_cum_pricevolume_weighted(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(20) 
        max_price_volume = (ret*volume).apply(lambda x: x.nlargest(5).index, axis=0).T

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        common_idx = ind.index.intersection(max_price_volume.index)
        max_price_volume = max_price_volume.loc[common_idx]
        max_price_volume['industry'] = max_price_volume.index.map(ind)

        def get_industry_return(date, industry):
            try:
                return ind_ret.at[date, industry]
            except KeyError:
                return pd.NA 
        res = max_price_volume.apply(lambda row: row[:-1].apply(lambda date: get_industry_return(date, row['industry'])), axis=1)
        n = 5
        weights = np.array([2 ** -((i - 1) / (n - 1)) for i in range(1, n + 1)])    
        res = res.multiply(weights).sum(axis=1)
        res.name = date
        return res

    def get_industry_co_20d_bottom15_cum_pricevolume_weighted(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(20) 
        smallest_price_volume = (ret*volume).apply(lambda x: x.nsmallest(15).index, axis=0).T

        ind = industry_info.read(start=date, stop=date)['first_industry_name'].reset_index(level='date', drop=True)
        ind_columns = ind.unique()
        ind_ret = industry_returns.read(start=rollback, stop=date).tail(20)
        ind_ret = ind_ret[ind_columns]

        common_idx = ind.index.intersection(smallest_price_volume.index)
        smallest_price_volume = smallest_price_volume.loc[common_idx]
        smallest_price_volume['industry'] = smallest_price_volume.index.map(ind)

        def get_industry_return(date, industry):
            try:
                return ind_ret.at[date, industry]
            except KeyError:
                return pd.NA 
        res = smallest_price_volume.apply(lambda row: row[:-1].apply(lambda date: get_industry_return(date, row['industry'])), axis=1)
        n = 15
        weights = np.array([2 ** -((i - 1) / (n - 1)) for i in range(1, n + 1)])    
        res = res.multiply(weights).sum(axis=1)
        res.index.name = 'order_book_id' 
        res.name = date
        return res


    def get_compound_industry_co_reverse_momemtum(self, date: str) -> pd.Series:
        vicm = self.read('industry_co_20d_top5_cum_pricevolume_weighted', start=date, stop=date).squeeze()
        vicr = self.read('industry_co_20d_bottom15_cum_pricevolume_weighted',start=date, stop=date).squeeze()
        res = vicm - vicr
        res.index.name = 'order_book_id' 
        res.name = date
        return res
    
    def get_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        rollback = quotes_day.get_trading_days_rollback(date, 1)
        price = quotes_day.read("close", code=universe, start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", code=universe, start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(1).squeeze()

        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def get_str_000906(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000906.XSHG')

        rollback = quotes_day.get_trading_days_rollback(date, 1)
        price = quotes_day.read("close", code=universe, start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", code=universe, start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(1).squeeze()

        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def get_str_weighted(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        str_20 = self.read("str", start=rollback, stop=date)
        str_rank = str_20.rank(axis=1, method='min', ascending=False)
        w =  ((0.7**str_rank)/((0.7**str_rank).sum()/21)).tail(20)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0)

        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20).fillna(0)
        
        res = pd.Series(index=ret.columns)
        for col in ret.columns:
            cov_value = w[col].cov(ret[col])
            res[col] = cov_value
        res.name = date
        return res
    
    def get_intraday_ret_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        price = quotes_min.read("close", code=universe, start=date, stop=date + pd.Timedelta(days=1))
        ret = (1 + price.pct_change(fill_method=None)).prod()
        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def get_10minute_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        price = quotes_min.read("close", code=universe, start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(periods=10, fill_method=None)
        res = (ret.sub(ret.mean(axis=1),axis=0)).abs()/ (ret.abs().add(np.abs(ret.mean(axis=1)), axis=0) + 0.01)
        res = res.mean()
        res.name = date
        return res
    
    def get_stv(self, date: str) -> pd.Series:
        volume = quotes_day.read("volume", start=date, stop=date)
        circulation_a = quotes_day.read("circulation_a", start=date, stop=date)
        turnover = (volume/circulation_a).squeeze()
        res = (turnover - turnover.mean()).abs() / (turnover.abs() + np.abs(turnover.mean()) + 0.1) 
        res.name = date
        return res

    def get_stv_weighted_v1(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        stv = self.read("stv", start=rollback, stop=date)
        rank = stv.rank(axis=1, method='min', ascending=False)
        w =  ((0.7**rank)/((0.7**rank).sum()/21)).tail(20)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0)

        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(20).fillna(0)
        
        res = pd.Series(index=ret.columns)
        for col in ret.columns:
            cov_value = w[col].cov(ret[col])
            res[col] = cov_value
        res.name = date
        return res
    
    def get_stv_weighted_v2(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        stv = self.read("stv", start=rollback, stop=date)
        rank = stv.rank(axis=1, method='min', ascending=False)
        w =  ((0.7**rank)/((0.7**rank).sum()/21)).tail(20)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0)

        stv = stv.dropna(how='all',axis=1).fillna(0)
        res = pd.Series(index=stv.columns)
        for col in stv.columns:
            cov_value = w[col].cov(stv[col])
            res[col] = cov_value
        res.name = date
        return res
    
    def get_abnretd(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).squeeze()
        res = ret.sub(market_ret, axis=0).max().abs()
        res.name = date
        return res
    
    def get_abneretm(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(periods=20, fill_method=None).iloc[-1]
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(periods=20, fill_method=None).iloc[-1]
        res = (ret- market_ret.values).abs()
        res.name = date
        return res

    def get_abnvold(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(252)
        res = (volume.tail(21) / volume.mean(axis=0)).max().abs()
        res.name = date
        return res

    def get_abnvolm(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(252)
        res = (volume.tail(21).sum() / volume.mean(axis=0)).abs()
        res.name = date
        return res
    
    def get_attn(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(20)
        n = 20
        weights = np.array([2 ** -((i - 1) / (n - 1)) for i in range(1, n + 1)])  
        res = volume.mul(weights[::-1],axis=0).sum()
        res.name = date
        return res

    def get_er(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(periods=21, fill_method=None)
        res = ret.iloc[-1] / ret.mean(axis=0)
        res.name = date
        return res

    def get_nearness_high_ly(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        high = quotes_day.read("high", start=rollback, stop=date).tail(252)
        res = high.tail(21).max()/high.max()
        res.name = date
        return res

    def get_nearness_high_historical(self, date: str) -> pd.Series:
        high = quotes_day.read( "high", stop=date)
        res = high.tail(21).max()/high.max()
        res.name = date
        return res
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .operators import wscore
from .base import (
    FactorManager, 
    quotes_day, 
    quotes_min
)


class DeraPriceFactor(FactorManager):

    def calc_volume_weighted_price(self, date: pd.Timestamp):
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        volume = quotes_min.read(name="volume", begin=date, end=date + pd.Timedelta(days=1))
        weight = volume / volume.sum()
        res = (price * weight).sum()
        res.name = date
        return res
    
    def calc_time_weighted_price(self, date: pd.Timestamp):
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        res = price.mean()
        res.name = date
        return res
    
    def calc_tail_weighted_price(self, date: pd.Timestamp):
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        volume = quotes_min.read(name="volume", begin=date, end=date + pd.Timedelta(days=1))
        price = price.between_time("14:30", "15:00")
        volume = volume.between_time("14:30", "15:00")
        weight = volume / volume.sum()
        res = (price * weight).sum()
        res.name = date
        return res

    def calc_head_weighted_price(self, date: pd.Timestamp):
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        volume = quotes_min.read(name="volume", begin=date, end=date + pd.Timedelta(days=1))
        price = price.between_time("09:30", "10:00")
        volume = volume.between_time("09:30", "10:00")
        weight = volume / volume.sum()
        res = (price * weight).sum()
        res.name = date
        return res


class PriceVolumeCorr(FactorManager):

    def calc_smart_money_ratio(self, date: pd.Timestamp) -> pd.DataFrame:
        rollback = self.get_trading_days_rollback(date, 9)
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None).abs()
        volume = quotes_min.read(name="volume", begin=date, end=date + pd.Timedelta(days=1))
        retvol = ret / (volume ** 0.25)
        rank = retvol.rank(axis=0, ascending=False)
        rank = rank.le(retvol.count() // 5, axis=1)
        retvol = volume.where(rank)
        res = ((retvol * price).sum() / retvol.sum()) / ((volume * price).sum() / volume.sum())
        res.name = date
        return res

    def calc_price_volume_corr(self, date: pd.Timestamp) -> pd.DataFrame:
        price = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        volume = quotes_min.read(name="volume", begin=date, end=date + pd.Timedelta(days=1))
        res = price.corrwith(volume, axis=0).replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def calc_average_relative_price_percent(self, date: pd.Timestamp) -> pd.DataFrame:
        high = quotes_min.read(name="high", begin=date, end=date + pd.Timedelta(days=1))
        low = quotes_min.read(name="low", begin=date, end=date + pd.Timedelta(days=1))
        close = quotes_min.read(name="close", begin=date, end=date + pd.Timedelta(days=1))
        open_ = quotes_min.read(name="open", begin=date, end=date + pd.Timedelta(days=1))
        twap = (high + low + open_ + close).mean()
        high = high.max()
        low = low.min()
        res = ((twap - low) / (high - low)).replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res
    
    def calc_compound_volume_first(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        diff_price = quotes_day.read("close", begin=rollback, end=date).diff(1).iloc[-20:]
        diff_volume = quotes_day.read("volume", begin=rollback, end=date).diff(1).iloc[-20:]
        
        corr = diff_price.corrwith(diff_volume, axis=0)
        corr.name = date
        corr = wscore(corr.to_frame().T).loc[date]
        
        diff_price_pos = diff_price[diff_price > 0]
        diff_volume_pos = diff_volume[diff_volume > 0]
        corr_pos_pos = diff_price_pos.corrwith(diff_volume_pos, axis=0)
        corr_pos_pos.name = date
        corr_pos_pos = wscore(corr_pos_pos.to_frame().T).loc[date]

        diff_price_neg = diff_price[diff_price < 0]
        diff_volume_neg = diff_volume[diff_volume < 0]
        corr_neg_neg = diff_price_neg.corrwith(diff_volume_neg, axis=0)
        corr_neg_neg.name = date
        corr_neg_neg = wscore(corr_neg_neg.to_frame().T).loc[date]

        corr_neg_pos = diff_price_pos.corrwith(diff_volume_neg, axis=0)
        corr_neg_pos.name = date
        corr_neg_pos = wscore(corr_neg_pos.to_frame().T).loc[date]

        corr_pos_neg = diff_price_neg.corrwith(diff_volume_pos, axis=0)
        corr_pos_neg.name = date
        corr_pos_neg = wscore(corr_pos_neg.to_frame().T).loc[date]        

        res = corr_pos_pos - corr_pos_neg - corr_neg_pos + corr_neg_neg
        return res
    
    def calc_trend_fund(self, date: str):
        rollback = self.get_trading_days_rollback(date, 4)
        volume_5d = quotes_min.read(
            pivot="volume", index="datetime", columns="code",
            datetime__ge=rollback, datetime__le=date + pd.Timedelta(days=1)
        )
        volume_5d_90 = volume_5d.quantile(0.90)
        volume = volume_5d.loc[date:]
        price = quotes_min.read(
            index="datetime", columns="code", pivot="close", 
            datetime__ge=date, datetime__le=date + pd.Timedelta(days=1)
        )

        trend_fund_volume = volume[volume > volume_5d_90]
        trend_fund_price = price[~np.isnan(trend_fund_volume)]

        support_price = trend_fund_price[trend_fund_price < trend_fund_price.mean()]
        support_volume = trend_fund_volume[~np.isnan(support_price)].sum()
        resistance_price = trend_fund_price[trend_fund_price > trend_fund_price.mean()]
        resistance_volume = trend_fund_volume[~np.isnan(resistance_price)].sum()

        shares = quotes_day.read(
            index="date", pivot="circulation_a", columns="code", 
            date__ge=date, date__le=date
        ).loc[date]
        res = (support_volume * resistance_volume) / shares
        res.name = date
        return res

    def calc_price_spread(self, date: str):
        rollback = self.get_trading_days_rollback(date, 252)
        price = quotes_day.read(
            index="date", pivot="close", columns="code", 
            date__ge=rollback, date__le=date
        )
        _adj = quotes_day.read(
            index="date", pivot="adjfactor", columns="code", 
            date__ge=rollback, date__le=date
        )
        ret = (price * _adj).pct_change(fill_method=None).tail(252)
        distance_matrix = 1 - ret.corr(method='pearson')

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
    
    def calc_rsi_20d(self, date: str): 
        rollback = self.get_trading_days_rollback(date, 20)
        price = quotes_day.read(
            index="date", pivot="close", columns="code", 
            date__ge=rollback, date__le=date
        )
        _adj = quotes_day.read(
            index="date", pivot="adjfactor", columns="code", 
            date__ge=rollback, date__le=date
        )
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        up = ret[ret > 0].mean()
        down = ret[ret < 0].mean().abs()
        res = up / (up + down)
        res.name = date
        return res
    
    def calc_minute_rsi(self, date: str): 
        price = quotes_min.read(
            index="datetime", columns="code", pivot="close", 
            datetime__ge=date, datetime__le=date + pd.Timedelta(days=1)
        )
        ret = price.pct_change(fill_method=None)
        up = ret[ret > 0].mean()
        down = ret[ret < 0].mean().abs()
        res = up / (up + down)
        res.name = date
        return res

    def calc_rsi_5min(self, date: str): 
        price = quotes_min.read(
            index="datetime", columns="code", pivot="close", 
            datetime__ge=date, datetime__le=date + pd.Timedelta(days=1)
        )
        ret = price.pct_change(periods=5, fill_method=None)
        up = ret[ret > 0].mean()
        down = ret[ret < 0].mean().abs()
        res = up / (up + down)
        res.name = date
        return res

    def calc_maxret_20d(self, date: str):
        rollback = self.get_trading_days_rollback(date, 20)
        price = quotes_day.read(
            index="date", pivot="close", columns="code", 
            date__ge=rollback, date__le=date
        )
        _adj = quotes_day.read(
            index="date", pivot="adjfactor", columns="code", 
            date__ge=rollback, date__le=date
        )
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        res = ret.max()
        res.name = date
        return res
    
    def calc_20d_top10_cum_ret(self, date: str):
        rollback = self.get_trading_days_rollback(date, 20)
        price = quotes_day.read(
            index="date", pivot="close", columns="code", 
            date__ge=rollback, date__le=date
        )
        _adj = quotes_day.read(
            index="date", pivot="adjfactor", columns="code", 
            date__ge=rollback, date__le=date
        )
        ret = (price * _adj).pct_change(fill_method=None).tail(20)
        res = ret.apply(lambda x: x.nlargest(10).sum(), axis=0)
        res.name = date
        return res

    def calc_20d_max_truerange(self, date: str):
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        prev_close = (quotes_day.read("close", start=rollback, stop=date) * _adj).shift(1)
        high = (quotes_day.read("high", start=rollback, stop=date) * _adj)
        low = (quotes_day.read("low", start=rollback, stop=date) * _adj)
        res = np.maximum.reduce([(high - low)/prev_close, abs(high - prev_close)/prev_close,  abs(prev_close - low)/prev_close]) # 每行最大值
        res = pd.DataFrame(res, index=high.index, columns=high.columns).tail(20).max()
        res.name = date
        return res

    def calc_industry_co_20d_maxret(self, date: str) -> pd.Series:
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
 
        def calc_industry_return(row):
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
    
    def calc_industry_co_20d_top5_cum_ret(self, date: str) -> pd.Series:
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

        def calc_industry_return(date, industry):
            try:
                return ind_ret.at[date, industry]
            except KeyError:
                return pd.NA 
        res = max_ret.apply(lambda row: row[:-1].apply(lambda date: get_industry_return(date, row['industry'])), axis=1).sum(axis=1)
        res.index.name = 'order_book_id' 
        res.name = date
        return res
    
    def calc_industry_co_20d_max_truerange(self, date: str) -> pd.Series:
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
 
        def calc_industry_return(row):
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
    
    def calc_industry_co_20d_top5_cumret_weighted(self, date: str) -> pd.Series:
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

        def calc_industry_return(date, industry):
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
    
    def calc_industry_co_20d_top5_cum_pricevolume_weighted(self, date: str) -> pd.Series:
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

        def calc_industry_return(date, industry):
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

    def calc_industry_co_20d_bottom15_cum_pricevolume_weighted(self, date: str) -> pd.Series:
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

        def calc_industry_return(date, industry):
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


    def calc_compound_industry_co_reverse_momemtum(self, date: str) -> pd.Series:
        vicm = self.read('industry_co_20d_top5_cum_pricevolume_weighted', start=date, stop=date).squeeze()
        vicr = self.read('industry_co_20d_bottom15_cum_pricevolume_weighted',start=date, stop=date).squeeze()
        res = vicm - vicr
        res.index.name = 'order_book_id' 
        res.name = date
        return res
    
    def calc_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        rollback = quotes_day.get_trading_days_rollback(date, 1)
        price = quotes_day.read("close", code=universe, start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", code=universe, start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(1).squeeze()

        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def calc_str_000906(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000906.XSHG')

        rollback = quotes_day.get_trading_days_rollback(date, 1)
        price = quotes_day.read("close", code=universe, start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", code=universe, start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None).tail(1).squeeze()

        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def calc_str_weighted(self, date: str) -> pd.Series:
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
    
    def calc_intraday_ret_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        price = quotes_min.read("close", code=universe, start=date, stop=date + pd.Timedelta(days=1))
        ret = (1 + price.pct_change(fill_method=None)).prod()
        res = (ret - ret.mean()).abs() / (ret.abs() + np.abs(ret.mean()) + 0.1) 
        res.name = date
        return res
    
    def calc_10minute_str_000985(self, date: str) -> pd.Series:
        universe = self.setup_universe(date, benchmark='000985.XSHG')

        price = quotes_min.read("close", code=universe, start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(periods=10, fill_method=None)
        res = (ret.sub(ret.mean(axis=1),axis=0)).abs()/ (ret.abs().add(np.abs(ret.mean(axis=1)), axis=0) + 0.01)
        res = res.mean()
        res.name = date
        return res
    
    def calc_stv(self, date: str) -> pd.Series:
        volume = quotes_day.read("volume", start=date, stop=date)
        circulation_a = quotes_day.read("circulation_a", start=date, stop=date)
        turnover = (volume/circulation_a).squeeze()
        res = (turnover - turnover.mean()).abs() / (turnover.abs() + np.abs(turnover.mean()) + 0.1) 
        res.name = date
        return res

    def calc_stv_weighted_v1(self, date: str) -> pd.Series:
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
    
    def calc_stv_weighted_v2(self, date: str) -> pd.Series:
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
    
    def calc_abnretd(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(fill_method=None)
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(fill_method=None).squeeze()
        res = ret.sub(market_ret, axis=0).max().abs()
        res.name = date
        return res
    
    def calc_abneretm(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(periods=20, fill_method=None).iloc[-1]
        market_ret = index_quotes_day.read('close', code='000985.XSHG', start=rollback, stop=date).pct_change(periods=20, fill_method=None).iloc[-1]
        res = (ret- market_ret.values).abs()
        res.name = date
        return res

    def calc_abnvold(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(252)
        res = (volume.tail(21) / volume.mean(axis=0)).max().abs()
        res.name = date
        return res

    def calc_abnvolm(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(252)
        res = (volume.tail(21).sum() / volume.mean(axis=0)).abs()
        res.name = date
        return res
    
    def calc_attn(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        volume = quotes_day.read("volume", start=rollback, stop=date).tail(20)
        n = 20
        weights = np.array([2 ** -((i - 1) / (n - 1)) for i in range(1, n + 1)])  
        res = volume.mul(weights[::-1],axis=0).sum()
        res.name = date
        return res

    def calc_er(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        price = quotes_day.read("close", start=rollback, stop=date)
        _adj = quotes_day.read("adjfactor", start=rollback, stop=date)
        ret = (price * _adj).pct_change(periods=21, fill_method=None)
        res = ret.iloc[-1] / ret.mean(axis=0)
        res.name = date
        return res

    def calc_nearness_high_ly(self, date: str) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 252)
        high = quotes_day.read("high", start=rollback, stop=date).tail(252)
        res = high.tail(21).max()/high.max()
        res.name = date
        return res

    def calc_nearness_high_historical(self, date: str) -> pd.Series:
        high = quotes_day.read( "high", stop=date)
        res = high.tail(21).max()/high.max()
        res.name = date
        return res
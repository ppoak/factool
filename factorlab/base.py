import quool
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

def zscore(df: pd.DataFrame):
    if isinstance(df.index, pd.MultiIndex):
        return df.groupby(level='date').transform(lambda x: (x - x.mean()) / x.std())
    else:
        return df.sub(df.mean(axis=1), axis=0
        ).div(df.std(axis=1), axis=0)
    
def minmax(df: pd.DataFrame):
    return df.sub(df.min(axis=1), axis=0).div(
        df.max(axis=1) - df.min(axis=1), axis=0)

def madoutlier( 
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    def apply_mad(df: pd.DataFrame) -> pd.DataFrame:
        median = df.median(axis=1)
        ad = df.sub(median, axis=0)
        mad = ad.abs().median(axis=1)
        thresh_down = median - dev * mad
        thresh_up = median + dev * mad
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
            other=np.nan, axis=0).where(~df.isna())
    
    if isinstance(df.index, pd.MultiIndex):
        return df.apply(lambda x: apply_mad(x.unstack('order_book_id')).unstack())
    else:
        return apply_mad(df)
        
def stdoutlier( 
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    thresh_down = mean - dev * std
    thresh_up = mean + dev * std
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def iqroutlier( 
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
): 
    thresh_up = df.quantile(1 - dev / 2, axis=1)
    thresh_down = df.quantile(dev / 2, axis=1)
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def fillna(
    df: pd.DataFrame,
    val: int | str = 0,
):
    return df.fillna(val)

def log(df: pd.DataFrame):
    return np.log((df + 1e-6).sub(df.min(axis=1), axis=0))

def sqrt(df: pd.DataFrame):
    return np.sqrt(df.sub(df.min(axis=1), axis=0))

def tsmean(df: pd.DataFrame, n: int = 20):
    return df.rolling(n).mean()

def neutralization( 
    df: pd.DataFrame, 
    industry: bool = False, 
    market: bool = False
): 
    trading_days = df.index
    start = trading_days[0]
    stop = trading_days[-1]

    if industry:
        ind = industry_info.read('first_industry_name', start=start, stop=stop).stack()
        ind = pd.get_dummies(ind, prefix='', prefix_sep='').astype(int)
        
    if market:
        marketcap = barra.read("log_marketcap",  start=start, stop=stop)

    def _neutralization(group, ind=None, marketcap=None, date=None):
        commom_index = group.index.intersection(ind.index).intersection(marketcap.index)
        group = group.loc[commom_index].fillna(0)
        X = [np.ones(len(group))] 
        if ind is not None:
            X.append(ind.loc[commom_index].fillna(0).values)
        if marketcap is not None:
            X.append(marketcap.loc[commom_index].fillna(0).values)
        
        X = np.column_stack(X)
        y = group.values
        
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X.dot(beta)
        return pd.Series(res, index=commom_index, name=date)
    
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_neutralization)(df.loc[date], 
                                 ind.loc[date] if ind is not None else None,
                                 marketcap.loc[date] if marketcap is not None else None, 
                                 date) 
        for date in trading_days
    )
    return pd.concat(results, axis=1).T.sort_index().loc[start:stop]

class BaseFactor(quool.Factor):
    # 如果因子需要截面计算，需要实现这个方法避免bias
    def setup_universe(
            self, 
            date: pd.Timestamp, 
            universe: str = '000985.XSHG', #'000906.XSHG
    )-> list:
        universe = filter.read(universe,start=date, stop=date).loc[date]
        return universe[universe==False].index.tolist()
    
    def filter_factor(
        self, 
        factor: pd.DataFrame,
        universe: str = '000985.XSHG',
    ):
        nonrealizable = filter.read(universe,start=factor.index[0], stop=factor.index[-1])
        return super().filter_factor(factor, nonrealizable=nonrealizable)
        
    def get_future(
        self, 
        ptype: str = "volume_weighted_price",
        period: int = 1, 
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        universe: str = '000985.XSHG',
    ):
        if stop is not None:
            stop = self.get_trading_days_rollback(stop, -period - 1)
        price = prices.read(ptype, start=start, stop=stop)
        adjfactor = quotes_day.read("adjfactor", start=start, stop=stop)
        price = price * adjfactor
        nonrealizable = filter.read(universe, start=start, stop=stop)
        return super().get_future(price, period, nonrealizable)

    def industry_inforcoef(
        self, 
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        factor: pd.DataFrame = None,
        future: pd.DataFrame = None,
        method: str = 'pearson',
    ):
        ind = industry_info.read('first_industry_name', start=start, stop=stop)
        return super().industry_inforcoef(ind, factor, future, method)
    
    def style_exposure(
        self, 
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None, 
        top_group: pd.DataFrame = None, 
        universe = '000985.XSHG'
    ):      
        barra = barra_rq.read(['size', 'non_linear_size', 'momentum', 'liquidity', 'book_to_price',
        'leverage', 'growth', 'earnings_yield', 'beta', 'residual_volatility'], start=start, stop=stop)
        weight = index_weights.read(universe,start=start, stop=stop)
        weight = (weight/weight).div(weight.count(axis=1),axis=0)
        return super().style_exposure(barra, weight, top_group)
    
    def get(self, name: str, start: str = None, stop: str = None, n_jobs: int = -1):
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = quotes_day.get_trading_days(start, stop)
        return super().get(name, trading_days, n_jobs)
    

quotes_day = quool.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
quotes_min = quool.Factor("./data/quotes-min", code_level="order_book_id", date_level="datetime")
stock_connect = quool.Factor("./data/stock-connect", code_level="order_book_id", date_level="date")
financial = quool.Factor("./data/financial", code_level="order_book_id", date_level="date")
industry_info = quool.Factor("./data/industry-info", code_level="order_book_id", date_level="date")
index_quotes_day = quool.Factor("./data/index-quotes-day", code_level="order_book_id", date_level="date")
index_quotes_min = quool.Factor("./data/index-quotes-min", code_level="order_book_id", date_level="datetime")
index_weights = quool.Factor("./data/index-weights", code_level="order_book_id", date_level="date")
barra = quool.Factor("./data/barra", code_level="order_book_id", date_level="date")
filter = quool.Factor("./data/filter-mask", code_level="order_book_id", date_level="date")
prices = quool.Factor("./data/prices", code_level="code", date_level="date")
industry_returns = quool.Factor("./data/industry-returns", code_level="industry", date_level="date")
barra_rq = quool.Factor("./data/barra_rq", code_level="order_book_id", date_level="date")

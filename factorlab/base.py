import quool
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

def wscore(df: pd.DataFrame, date: pd.Timestamp):
    weight = index_weights.read('000985.XSHG',start=date, stop=date)
    return (df.sub(np.sum(weight * df, axis=1),axis=0)
            ).div(df.std(axis=1), axis=0)

def zscore(df: pd.DataFrame):
    return df.sub(df.mean(axis=1), axis=0
        ).div(df.std(axis=1), axis=0)

def minmax(df: pd.DataFrame):
    return df.sub(df.min(axis=1), axis=0).div(
        df.max(axis=1) - df.min(axis=1), axis=0)

def madoutlier( # 高度鲁棒：对极端值和异常值非常敏感，适合大量极端值或严重偏态。
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
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

def stdoutlier( # 适用于正态数据
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

def iqroutlier( # 适合数据的分布大致对称或略微偏态。
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

def log(df: pd.DataFrame): # 适用于分布简单的右偏数据，适合处理正值
    return np.log((df + 1e-6).sub(df.min(axis=1), axis=0))

def sqrt(df: pd.DataFrame): # 减少数据的范围
    return np.sqrt(df.sub(df.min(axis=1), axis=0))

def box_cox(df: pd.DataFrame): # 正态化, 适合处理正值且偏态不是很严重的数据。
    def safe_boxcox(row):
        non_nan_values = row.dropna()
        if non_nan_values.empty:
            return pd.Series([np.nan] * len(row), index=row.index)
        else:
            transformed_values = boxcox((non_nan_values - non_nan_values.min() + 1e-6))[0]
            res = pd.Series([np.nan] * len(row), index=row.index)
            res.loc[non_nan_values.index] = transformed_values
            return res

    df_transformed = df.apply(safe_boxcox, axis=1, result_type='expand')
    return df_transformed

def yeo_johnson(df: pd.DataFrame): # 正态化, 适合偏态较严重，可以含零值和负值的数据
    def transform_row(row):
        row_reshaped = row.values.reshape(-1, 1)
        transformed_row = pt.fit_transform(row_reshaped)
        return transformed_row.flatten()

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df_transformed = df.apply(transform_row, axis=1, result_type='expand')
    df_transformed.columns = df.columns
    return df_transformed

def tsmean(df: pd.DataFrame, n: int = 20):
    return df.rolling(n).mean()


class BaseFactor(quool.Factor):

    def get_future(
        self, 
        ptype: str = "volume_weighted_price",
        period: int = 1, 
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        skip_nonperiod_day: bool = False,
    ):
        if stop is not None:
            stop = self.get_trading_days_rollback(stop, -period - 1)
        price = self.read(ptype, start=start, stop=stop)
        adjfactor = quotes_day.read("adjfactor", start=start, stop=stop)
        price = price * adjfactor
        ret = price / price.shift(1) - 1
        st = quotes_day.read("st", start=start, stop=stop)
        suspended = quotes_day.read("suspended", start=start, stop=stop)
        nonrealizable = st | suspended | (ret >= 0.1)

        return super().get_future(price, period, skip_nonperiod_day, nonrealizable)

    def get(self, name: str, start: str = None, stop: str = None, n_jobs: int = -1):
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = quotes_day.get_trading_days(start, stop)
        return super().get(name, trading_days, n_jobs)


quotes_day = quool.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
quotes_min = quool.Factor("./data/quotes-min", code_level="order_book_id", date_level="datetime")
stock_connect = quool.Factor("./data/stock-connect", code_level="order_book_id", date_level="date")
financial = quool.Factor("./data/financial", code_level="order_book_id", date_level="date")
index_weights = quool.Factor("./data/index-weights", code_level="order_book_id", date_level="date")
index_quotes_day = quool.Factor("./data/index-quotes-day", code_level="order_book_id", date_level="date")
industry_info = quool.Factor("./data/industry-info", code_level="order_book_id", date_level="date")

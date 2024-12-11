import numpy as np
import pandas as pd
from .base import index_weights


def add(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
    if fillna:
        return dfa.add(dfb, fill_value=fillna)
    return dfa + dfb

def sub(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
    if fillna:
        return dfa.sub(dfb, fill_value=fillna)
    return dfa - dfb

def mul(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
    if fillna:
        return dfa.mul(dfb, fill_value=fillna)
    return dfa * dfb

def div(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
    if fillna:
        return dfa.div(dfb, fill_value=fillna)
    return dfa / dfb

def mask(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
    return dfa.mask(dfb, other=np.nan)

def where(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
    return dfa.where(dfb, other=dfc)

def shift(df: pd.DataFrame, n: int):
    return df.shift(n)

def rsum(df: pd.DataFrame, n: int, axis: int = 0):
    if n < 0:
        return df.expanding(min_periods=-n, axis=axis).sum()
    elif n > 0 and n < 1:
        return df.ewm(alpha=n, axis=axis).sum()
    return df.rolling(min_periods=n, axis=axis).sum()

def corr(dfa: pd.DataFrame, dfb: pd.DataFrame, axis: int = 0):
    return dfa.corrwith(dfb, axis=axis)

def rank(df: pd.DataFrame, ascending: bool = False, axis: int = 0):
    return df.rank(axis=axis, ascending=ascending)

def group(df: pd.DataFrame, n: int, axis: int = 0):
    return df.apply(lambda x: pd.qcut(x, q=n, labels=False), axis=1) + 1

def zscore(df: pd.DataFrame):
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def wscore(df: pd.DataFrame):
    weight = index_weights.read(
        pivot="weight", index="date", columns="code",
        date__in=df.index, index_code='000985.XSHG'
    )
    return (df.sub(np.sum(weight * df, axis=1),axis=0)
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

def weightify(df: pd.DataFrame):
    return df.div(df.sum(axis=1), axis=0)

def diff(df: pd.DataFrame, n: int = 1, axis: int = 0, nofirst: bool = False):
    if nofirst:
        df = df.copy()
        first = df.iloc[0].copy()
        df = df.diff(n, axis=axis)
        df.iloc[0] = first
        return df
    return df.diff(n, axis=axis)

def absolute(df: pd.DataFrame):
    return df.abs()

def sum(df: pd.DataFrame, axis: int = 0):
    return df.sum(axis=axis)

def cumsum(df: pd.DataFrame, axis: int = 0):
    return df.cumsum(axis=axis)

def cumprod(df: pd.DataFrame, axis: int = 0):
    return df.cumprod(axis=axis)

def log(df: pd.DataFrame):
    return np.log((df + 1e-6).sub(df.min(axis=1), axis=0))

def sqrt(df: pd.DataFrame):
    return np.sqrt(df.sub(df.min(axis=1), axis=0))

def mean(df: pd.DataFrame, axis: int = 0):
    return df.mean(axis=axis)

def tsmean(df: pd.DataFrame, n: int = 20):
    return df.rolling(n).mean()

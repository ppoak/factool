import numpy as np
import pandas as pd


class Operator:

    @staticmethod
    def add(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.add(dfb, fill_value=fillna)
        return dfa + dfb

    @staticmethod
    def sub(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.sub(dfb, fill_value=fillna)
        return dfa - dfb

    @staticmethod
    def mul(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.mul(dfb, fill_value=fillna)
        return dfa * dfb

    @staticmethod
    def div(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.div(dfb, fill_value=fillna)
        return dfa / dfb

    @staticmethod
    def mask(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
        return dfa.mask(dfb, other=np.nan)

    @staticmethod
    def where(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
        return dfa.where(dfb, other=dfc)

    @staticmethod
    def shift(df: pd.DataFrame, n: int):
        return df.shift(n)

    @staticmethod
    def rsum(df: pd.DataFrame, n: int, axis: int = 0):
        if n < 0:
            return df.expanding(min_periods=-n, axis=axis).sum()
        elif n > 0 and n < 1:
            return df.ewm(alpha=n, axis=axis).sum()
        return df.rolling(min_periods=n, axis=axis).sum()

    @staticmethod
    def rmean(df: pd.DataFrame, n: int, axis: int = 0):
        if n < 0:
            return df.expanding(min_periods=-n, axis=axis).mean()
        elif n > 0 and n < 1:
            return df.ewm(alpha=n, axis=axis).mean()
        return df.rolling(min_periods=n, axis=axis).mean()

    @staticmethod
    def corr(dfa: pd.DataFrame, dfb: pd.DataFrame, axis: int = 0):
        return dfa.corrwith(dfb, axis=axis)

    @staticmethod
    def rank(df: pd.DataFrame, ascending: bool = False, axis: int = 0):
        return df.rank(axis=axis, ascending=ascending)

    @staticmethod
    def group(df: pd.DataFrame, n: int, axis: int = 0):
        return df.apply(lambda x: pd.qcut(x, q=n, labels=False), axis=1) + 1

    @staticmethod
    def zscore(df: pd.DataFrame):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    @staticmethod
    def wscore(df: pd.DataFrame, weight: pd.DataFrame):
        return (df.sub(np.sum(weight * df, axis=1), axis=0)).div(df.std(axis=1), axis=0)

    @staticmethod
    def minmax(df: pd.DataFrame):
        return df.sub(df.min(axis=1), axis=0).div(
            df.max(axis=1) - df.min(axis=1), axis=0
        )

    @staticmethod
    def madoutlier(df: pd.DataFrame, dev: int, drop: bool = False):
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
                other=np.nan,
                axis=0,
            ).where(~df.isna())

        if isinstance(df.index, pd.MultiIndex):
            return df.apply(lambda x: apply_mad(x.unstack("order_book_id")).unstack())
        else:
            return apply_mad(df)

    @staticmethod
    def stdoutlier(df: pd.DataFrame, dev: int, drop: bool = False):
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        thresh_down = mean - dev * std
        thresh_up = mean + dev * std
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0), other=np.nan, axis=0
        ).where(~df.isna())

    @staticmethod
    def iqroutlier(df: pd.DataFrame, dev: int, drop: bool = False):
        thresh_up = df.quantile(1 - dev / 2, axis=1)
        thresh_down = df.quantile(dev / 2, axis=1)
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0), other=np.nan, axis=0
        ).where(~df.isna())

    @staticmethod
    def fillna(
        df: pd.DataFrame,
        val: int | str = 0,
    ):
        return df.fillna(val)

    @staticmethod
    def weightify(df: pd.DataFrame):
        return df.div(df.sum(axis=1), axis=0)

    @staticmethod
    def diff(df: pd.DataFrame, n: int = 1, axis: int = 0, nofirst: bool = False):
        if nofirst:
            df = df.copy()
            first = df.iloc[0].copy()
            df = df.diff(n, axis=axis)
            df.iloc[0] = first
            return df
        return df.diff(n, axis=axis)

    @staticmethod
    def absolute(df: pd.DataFrame):
        return df.abs()

    @staticmethod
    def sum(df: pd.DataFrame, axis: int = 0):
        return df.sum(axis=axis)

    @staticmethod
    def cumsum(df: pd.DataFrame, axis: int = 0):
        return df.cumsum(axis=axis)

    @staticmethod
    def cumprod(df: pd.DataFrame, axis: int = 0):
        return df.cumprod(axis=axis)

    @staticmethod
    def log(df: pd.DataFrame):
        return np.log((df + 1e-6).sub(df.min(axis=1), axis=0))

    @staticmethod
    def sqrt(df: pd.DataFrame):
        return np.sqrt(df.sub(df.min(axis=1), axis=0))

    @staticmethod
    def mean(df: pd.DataFrame, axis: int = 0):
        return df.mean(axis=axis)

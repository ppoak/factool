import quool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


def zscore(df: pd.DataFrame):
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

def log(df: pd.DataFrame, dropinf: bool = True):
    if dropinf:
        return np.log(df).replace([np.inf, -np.inf], np.nan)
    return np.log(df)

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

        ret = price / price.shift(1) - 1
        st = quotes_day.read("st", start=start, stop=stop)
        suspended = quotes_day.read("suspended", start=start, stop=stop)
        nontradable = st | suspended

        price = price.where((~nontradable) | (ret >= 0.1), other=np.nan)
        future = price.shift(-1 - period) / price.shift(-1) - 1
        future = future.dropna(axis=0, how='all')

        if skip_nonperiod_day:
            return future.iloc[::period].squeeze()
        return future.squeeze()
    
    def _prepare_factor(
        self, 
        factor: str | pd.DataFrame | pd.Series,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        processor: list = None
    ):
        if isinstance(factor, pd.Series) and factor.index.nlevels == 2:
            factor = factor.unstack(self._code_level)

        elif isinstance(factor, pd.DataFrame):
            factor = factor.loc[start:stop]

        elif isinstance(factor, str):
            factor = self.read(factor, start=start, stop=stop)
        
        else:
            ValueError("Invalid factor type")
        
        processor = processor or []
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            factor = proc(factor, **kwargs)
        
        if isinstance(start, (list, pd.DatetimeIndex)):
            factor = factor.loc[start]
        else:
            factor = factor.loc[start:stop]

        return factor
    
    def perform_crosssection(
        self, 
        factor: str | pd.DataFrame, 
        *,
        date: str | pd.Timestamp | pd.DataFrame | pd.Series,
        processor: list = None,
        period: int = 1,
        ptype: str = "volume_weighted_price",
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, date, date)
        factor = self._prepare_factor(factor, future.name, future.name, processor)
        data = pd.concat([factor.squeeze(), future], axis=1, keys=["Factor", future.name])

        if image is not None:
            pd.plotting.scatter_matrix(data, figsize=(20, 20), hist_kwds={'bins': 100})
            
            plt.tight_layout()
            if isinstance(image, (str, Path)):
                plt.savefig(image)
            else:
                plt.show()
                
        if result is not None:
            data.to_excel(result)
        
        return data.corr()

    def perform_inforcoef(
        self,
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "volume_weighted_price",
        processor: list = None,
        rolling: int = 20, 
        method: str = 'pearson', 
        skip_nonperiod_day: bool = False,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)
        
        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, future.index, None, processor)
        else:
            factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor)

        inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if image is not None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax, label='infor-coef', alpha=0.7, title=f'{factor} Information Coef')
            inforcoef.rolling(rolling).mean().plot(linestyle='--', ax=ax, label='trend')
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True, ax=ax, label='cumm-infor-coef')
            pd.Series(np.zeros(inforcoef.shape[0]), index=inforcoef.index).plot(color='grey', ax=ax, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()
        
        if result is not None:
            inforcoef.to_excel(result)
        return inforcoef
    
    def perform_grouping(
        self, 
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        processor: list = None,
        ptype: str = "volume_weighted_price",
        ngroup: int = 5, 
        commission: float = 0.002, 
        skip_nonperiod_day: bool = True,
        n_jobs: int = 1,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)
        
        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, start=future.index, processor=processor)
        else:
            factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor)
        
        # ngroup test
        try:
            groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
        except:
            for date in factor.index:
                try:
                    pd.qcut(factor.loc[date], q=ngroup, labels=False)
                except:
                    raise ValueError(f"on date {date}, grouping failed")
        
        def _grouping(x):
            group = groups.where(groups == x)
            weight = (group / group).fillna(0)
            weight = weight.div(weight.sum(axis=1), axis=0)
            _period = period if skip_nonperiod_day else 1
            delta = weight.diff(periods=_period).fillna(0)
            turnover = delta.abs().sum(axis=1) / 2 / _period
            ret = (future * weight).sum(axis=1).shift(1) / _period
            ret -= commission * turnover
            ret = ret.fillna(0)
            val = (ret + 1).cumprod()
            return {
                'evaluation': quool.evaluate(val, turnover=turnover, image=False),
                'value': val, 'turnover': turnover,
            }
            
        ngroup_result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grouping)(i) for i in range(1, ngroup + 1))
        ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_returns = ngroup_value.pct_change().fillna(0)
        longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
        longshort_value = (longshort_returns + 1).cumprod()
        longshort_evaluation = quool.evaluate(longshort_value, image=False)
        
        # naming
        longshort_evaluation.name = "longshort"
        longshort_value.name = "longshort value"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            longshort_value.plot(ax=ax, linestyle='--')
            ngroup_value.plot(ax=ax, alpha=0.8)
            ngroup_turnover.plot(ax=ax, secondary_y=True, alpha=0.2)
            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()            
        
        if result is not None:
            with pd.ExcelWriter(result) as writer:
                ngroup_evaluation.to_excel(writer, sheet_name="ngroup_evaluation")
                longshort_evaluation.to_excel(writer, sheet_name="longshort_evaluation")
                ngroup_value.to_excel(writer, sheet_name="ngroup_value")
                ngroup_turnover.to_excel(writer, sheet_name="ngroup_turnover")
                longshort_value.to_excel(writer, sheet_name="longshort_value")
        
        return pd.concat([ngroup_evaluation, longshort_evaluation], axis=1)
                
    def perform_topk(
        self, 
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "volume_weighted_price",
        processor: list = None,
        topk: int = 100, 
        commission: float = 0.002, 
        skip_nonperiod_day: bool = True,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)

        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, start=future.index, processor=processor)
        else:
            factor = self._prepare_factor(factor, start, stop, processor)
            
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        _period = period if skip_nonperiod_day else 1
        turnover = topks.diff(period=_period).fillna(0).abs().sum(axis=1) / 2 / _period
        ret = (topks * future).sum(axis=1).shift(1).fillna(0) - turnover * commission
        ret = ret.fillna(0) / _period
        val = (1 + ret).cumprod()
        eva = quool.evaluate(val, turnover=turnover, image=False)

        val.name = "value"
        turnover.name = "turnover"
        eva.name = "evaluation"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            val.plot(ax=ax, title="Top K")
            turnover.plot(ax=ax, secondary_y=True, alpha=0.5)
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()

        if result is not None:
            pd.concat([eva, val, turnover], axis=1).to_excel(result)

        return eva

    def get(self, name: str, start: str = None, stop: str = None, n_jobs: int = -1):
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = quotes_day.get_trading_days(start, stop)
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index().loc[start:stop]
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]


quotes_day = quool.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
quotes_min = quool.Factor("./data/quotes-min", code_level="order_book_id", date_level="datetime")
stock_connect = quool.Factor("./data/stock-connect", code_level="order_book_id", date_level="date")
financial = quool.Factor("./data/financial", code_level="order_book_id", date_level="date")
index_weights = quool.Factor("./data/index-weights", code_level="order_book_id", date_level="date")
index_quotes_day = quool.Factor("./data/index-quotes-day", code_level="order_book_id", date_level="date")

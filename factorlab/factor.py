import numpy as np
import pandas as pd
import dataforge as forge
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


class Factor(forge.ParquetManager):

    def __init__(self, path: str, partition: str = None):
        super().__init__(path=path, index=["date", "code", "name"], partition=partition)
        if self.partitions:
            partition_sizes = np.array([partition.stat().st_size for partition in self.partitions])
            min_partition = self.partitions[np.argmin(partition_sizes)]
            columns = pd.read_parquet(min_partition).columns
            if not pd.Index(["date", "code", "name"]).isin(columns).all():
                raise ValueError("Factor must have columns: date, code, name")

    def read(
        self, 
        processor: list = None,
        **kwargs
    ) -> pd.Series | pd.DataFrame:
        processor = processor or []
        if not isinstance(processor, list):
            processor = [processor]
        
        df = super().read(**kwargs)
        
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            df = proc(df, **kwargs)
        return df.dropna(axis=0, how='all')

    def get_trading_days(
        self,
        index_code: str = "000001.XSHG",
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
    ):
        params = {
            "code": index_code,
            f"date__ge": pd.to_datetime(begin or "2000-01-01"),
            f"date__le": pd.to_datetime(end or "now"),
            "index": "date",
            "sort_index": True
        }
        return super().read(**params).index
    
    def get_trading_days_rollback(
        self, 
        date: str | pd.Timestamp = None, 
        rollback: int = 1
    ):
        date = pd.to_datetime(date or 'now')
        if rollback >= 0:
            trading_days = self.get_trading_days(begin=None, end=date)
            rollback = trading_days[trading_days <= date][-rollback - 1]
        else:
            trading_days = self.get_trading_days(begin=date, end=None)
            rollback = trading_days[min(len(trading_days) - 1, -rollback)]
        return rollback
    
    def get_returns(
        self, 
        ptype: str,
        period: int = 1, 
        lag: int = 1,
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        skip_nonperiod_day: bool = False,
        nonrealizable: pd.DataFrame = True,
    ):
        if end is not None:
            end = self.get_trading_days_rollback(end, -period - 1)
        price = self.read(
            index="date", columns="code", pivot="value", 
            name=ptype, date__ge=begin, date__le=end
        )
        price = price.where(~nonrealizable, other=np.nan)
        if period > 0:
            future = price / price.shift(period) - 1
        else:
            future = price.shift(-lag - period) / price.shift(-lag) - 1
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
        if isinstance(factor, pd.DataFrame):
            factor = factor.loc[start:stop]
        elif isinstance(factor, str):
            factor = self.read(
                index="date", columns="code", pivot="value",
                name=factor, date__ge=start, date__le=stop
            )
        else:
            ValueError("Invalid factor type")
        
        processor = processor or []
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            factor = proc(factor, **kwargs)
        
        return factor
    
    def perform_crosssection(
        self, 
        factor: str | pd.DataFrame, 
        *,
        date: str | pd.Timestamp | pd.DataFrame | pd.Series = None,
        processor: list = None,
        period: int = -1,
        ptype: str = "volume_weighted_price",
        image: str | bool = True, 
        result: str = None
    ):
        date = self.get_trading_days_rollback(date, 0)
        future = self.get_returns(ptype, period, date, date)
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
        period: int = -1,
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
        future = self.get_returns(ptype, period, start, stop)
        
        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, future.index, None, processor)
        else:
            factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor)

        inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if image is not None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax, label='infor-coef', alpha=0.7, title='Information Coef')
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
        future = self.get_returns(ptype, period, start, stop)
        
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
                'evaluation': forge.Evaluator._evaluate(val),
                'turnover': turnover,
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
        longshort_evaluation = forge.Evaluator._evaluate(longshort_value)
        
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
        future = self.get_returns(ptype, period, start, stop)

        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, start=future.index, processor=processor)
        else:
            factor = self._prepare_factor(factor, start, stop, processor)
            
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        _period = period if skip_nonperiod_day else 1
        turnover = topks.diff(periods=_period).fillna(0).abs().sum(axis=1) / 2 / _period
        ret = (topks * future).sum(axis=1).shift(1).fillna(0) - turnover * commission
        ret = ret.fillna(0) / _period
        val = (1 + ret).cumprod()
        eva = forge.Evaluator._evaluate(val)

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

    def upsert(self, data, partition=None, njobs=-1):
        if not pd.Index(self.index).isin(data.columns).all():
            raise ValueError("Malformed data, please check your input")
        return super().upsert(data, partition, njobs)

    def get(self, name: str, trading_days: pd.DatetimeIndex, n_jobs: int = -1):
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index()

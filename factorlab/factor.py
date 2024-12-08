import numpy as np
import pandas as pd
import dataforge as forge
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


class FactorManager(forge.ParquetManager):

    def __init__(
        self, 
        path: str | Path, 
        name_col: str = "name",
        val_col: str = "value",
        date_col: str = "date",
        code_col: str = "code",
        partition: str = None,
    ):
        super().__init__(path=path, index=[date_col, code_col, name_col], partition=partition)
        if bool(name_col) != bool(val_col):
            raise ValueError("Either both name_col and val_col are provided or neither is provided")
        self.date_col = date_col
        self.code_col = code_col
        self.name_col = name_col
        self.val_col = val_col
    
    def read(
        self, 
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
        processor: list = None,
        **kwargs
    ) -> pd.DataFrame:
        processor = processor or []
        if not isinstance(processor, list):
            processor = [processor]
        
        if self.name_col:
            kwargs.update({"pivot": self.val_col, self.name_col: name, "index": self.date_col, "columns": self.code_col})
        else:
            kwargs.update({"pivot": name, "index": self.date_col, "columns": self.code_col})
        if begin is not None:
            kwargs.update({f"{self.date_col}__ge": begin})
        if end is not None:
            kwargs.update({f"{self.date_col}__le": end})
        df = super().read(**kwargs)
        
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            df = proc(df, **kwargs)
        return df.dropna(axis=0, how='all')

    def get_trading_days(
        self,
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        persistence_code: str = "000001.XSHG",
    ):
        params = {
            "index": self.date_col,
            self.code_col: persistence_code,
        }
        if begin is not None:
            params[f"{self.date_col}__ge"] = pd.to_datetime(begin)
        if end is not None:
            params[f"{self.date_col}__le"] = pd.to_datetime(end)
        return super().read(**params).index.unique().sort_values()

    def get_trading_days_rollback(
        self, 
        date: str | pd.Timestamp = None, 
        rollback: int = 1
    ):
        date = pd.to_datetime(date or 'now')
        if rollback >= 0:
            trading_days = self.get_trading_days(begin=None, end=date)
            rollback_days = trading_days[trading_days <= date]
            rollback = rollback_days[max(-len(rollback_days), -rollback - 1)]
        else:
            trading_days = self.get_trading_days(begin=date, end=None)
            rollback = trading_days[min(len(trading_days) - 1, -rollback)]
        return rollback
    
    @staticmethod
    def _get_returns(
        price: pd.DataFrame,
        period: int = 1, 
        lag: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
    ):
        if weight is not None:
            price = price.where(weight > 0)
        if feasible is not None:
            price = price.where(feasible)

        if period > 0:
            returns = price / price.shift(period) - 1
        else:
            returns = price.shift(-lag + period) / price.shift(-lag) - 1
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all')

        return returns.iloc[::abs(period)]
    
    def get_returns(
        self,
        name: str,
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        period: int = 1,
        lag: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        skip_nonperiod_day: bool = False,
    ) -> pd.DataFrame:
        price = self.read(name=name, begin=begin, end=end)
        return self._get_returns(
            price=price, 
            period=period, 
            lag=lag, 
            weight=weight, 
            feasible=feasible, 
        )
    
    @staticmethod
    def _perform_crosssection(
        factor: pd.Series,
        returns: pd.Series,
        date: str | int | pd.Timestamp = -1
    ):
        if isinstance(date, (str, pd.Timestamp)):
            result = pd.concat([factor.loc[date], returns.loc[date]], axis=1)
        else:
            result = pd.concat([factor.iloc[date], returns.iloc[date]], axis=1)
        return result
    
    @staticmethod
    def _perform_inforcoef(
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        method: str = "spearman",
    ):
        return factor.corrwith(returns, method=method, axis=1)
    
    @staticmethod
    def _perform_grouping(
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        ngroup: int = 5,
        commission: float = 0.0005,
        n_jobs: int = 1,
    ):
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
            delta = weight.diff().fillna(0)
            turnover = delta.abs().sum(axis=1) / 2
            ret = (returns * weight).sum(axis=1).shift(1)
            ret -= commission * turnover
            ret = ret.fillna(0)
            val = (ret + 1).cumprod()
            return {
                'value': val,
                'turnover': turnover,
                'evaluation': forge.Evaluator._evaluate(val),
            }
            
        ngroup_result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grouping)(i) for i in range(1, ngroup + 1)
        )

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
        longshort_evaluation.name = "longshort"
        longshort_value.name = "longshort value"

        return {
            "value": pd.concat([ngroup_value, longshort_value], axis=1),
            "turnover": ngroup_turnover,
            "evaluation": pd.concat([ngroup_evaluation, longshort_evaluation], axis=1),
        }
    
    @staticmethod
    def _perform_topk(
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        topk: int = 5,
        commission: float = 0.0005,
    ):
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        turnover = topks.diff().fillna(0).abs().sum(axis=1) / 2
        ret = (topks * returns).sum(axis=1).shift(1).fillna(0) - turnover * commission
        ret = ret.fillna(0)
        val = (1 + ret).cumprod()
        eva = forge.Evaluator._evaluate(val)

        val.name = "value"
        turnover.name = "turnover"
        eva.name = "evaluation"
        
        return {
            "value": val,
            "turnover": turnover,
            "evaluation": eva,
        }
    
    @staticmethod
    def _align_factor_returns(
        factor: pd.DataFrame,
        returns: pd.DataFrame,
    ):
        factor = factor.loc[returns.index.intersection(factor.index)]
        returns = returns.loc[factor.index]
        return factor, returns

    def performance(
        self,
        name: str,
        ptype: str,
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        processor: list = None,
        period: int = -1,
        lag: int = 1,
        weight: pd.DataFrame = None,
        feasible: pd.DataFrame = None,
        skip_nonperiod_day: bool = False,
        crosssection_date: int | str | pd.Timestamp = -1,
        method: str = "spearman",
        ngroup: int = 5,
        topk: int = 30,
        commission: float = 0.0005,
        n_jobs: int = 1,
    ):
        factor = self.read(name=name, begin=begin, end=end, processor=processor)
        returns = self.get_returns(
            name=ptype, 
            begin=begin, 
            end=end, 
            period=period, 
            lag=lag, 
            weight=weight, 
            feasible=feasible, 
            skip_nonperiod_day=skip_nonperiod_day
        )
        factor, returns = self._align_factor_returns(factor=factor, returns=returns)
        crosssection_result = self._perform_crosssection(factor=factor, returns=returns, date=crosssection_date)
        inforcoef_result = self._perform_inforcoef(factor=factor, returns=returns, method=method)
        grouping_result = self._perform_grouping(factor=factor, returns=returns, ngroup=ngroup, commission=commission, n_jobs=n_jobs)
        topk_result = self._perform_topk(factor=factor, returns=returns, topk=topk, commission=commission)

        return {
            "crosssection": crosssection_result,
            "inforcoef": inforcoef_result,
            "grouping": grouping_result,
            "topk": topk_result,
        }

    def upsert(self, data, partition=None, njobs=-1):
        if not pd.Index(self.index).isin(data.columns).all():
            raise ValueError("Malformed data, please check your input")
        return super().upsert(data, partition, njobs)

    def calc(self, name: str, trading_days: pd.DatetimeIndex, n_jobs: int = -1):
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "calc_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index()


quotes_day = FactorManager(r"D:/Documents/DataBase/quotes_day", name_col=None, val_col=None)
quotes_min = FactorManager(r"D:/Documents/DataBase/quotes_min", name_col=None, val_col=None)
index_weights = FactorManager(r"D:/Documents/DataBase/index_weights", name_col=None, val_col=None)

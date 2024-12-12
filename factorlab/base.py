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

    def read(
        self, 
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
        **kwargs
    ) -> pd.DataFrame:
        if self.name_col:
            kwargs.update({"pivot": self.val_col, self.name_col: name, "index": self.date_col, "columns": self.code_col})
        else:
            kwargs.update({"pivot": name, "index": self.date_col, "columns": self.code_col})
        if begin is not None:
            kwargs.update({f"{self.date_col}__ge": begin})
        if end is not None:
            kwargs.update({f"{self.date_col}__le": end})
        df = super().read(**kwargs)
        
        return df.sort_index().dropna(axis=0, how='all')

    def calc(self, name: str, trading_days: pd.DatetimeIndex, njobs: int = -1):
        result = Parallel(n_jobs=njobs, backend='loky')(
            delayed(getattr(self, "calc_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index()


quotes_day = FactorManager(r"D:/Documents/DataBase/quotes_day", name_col=None, val_col=None)
quotes_min = FactorManager(r"D:/Documents/DataBase/quotes_min", name_col=None, val_col=None, date_col="datetime")
index_weights = FactorManager(r"D:/Documents/DataBase/index_weights", name_col="index_code", val_col="weight")

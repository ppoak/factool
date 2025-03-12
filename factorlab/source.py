import duckdb
import pandas as pd
from pathlib import Path
from xtquant import xtdata
from tqdm.auto import tqdm
from quool import ParquetManager
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


class FactorSource(ABC):

    @abstractmethod
    def get_times(self, begin: str, end: str):
        raise NotImplementedError

    @abstractmethod
    def get_time(self, time: str, n: int):
        raise NotImplementedError

    @abstractmethod
    def get_factor(self, name: str, begin: str, end: str):
        raise NotImplementedError

    def calc(self, name: str, trading_days: pd.DatetimeIndex, njobs: int = -1):
        result = Parallel(n_jobs=njobs, backend="loky")(
            delayed(getattr(self, "calc_" + name))(date)
            for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index()


class XtFactorSource(FactorSource):

    def __init__(self, path: str, sector: str = "沪深A股", period: str = "1d"):
        xtdata.data_dir = path
        self._stock_list = xtdata.get_stock_list_in_sector(sector)
        self._period = period

    def get_times(self, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        data = xtdata.get_market_data_ex(
            ["time"], stock_list=["000001.SZ"], start_time=begin, end_time=end
        )
        return pd.to_datetime(data["000001.SZ"].index)

    def get_time(self, time: str, n: int):
        if n > 0:
            return self.get_times(None, time)[-n:]
        return self.get_times(time, None)[:n]

    def get_factor(self, name: str, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        factor = xtdata.get_market_data_ex(
            [name], stock_list=self._stock_list, start_time=begin, end_time=end
        )
        factor = pd.concat(
            [f[name] for f in factor.values()], axis=1, keys=factor.keys()
        )
        factor.index = pd.to_datetime(factor.index)
        return factor


class ParquetFactorSource(FactorSource, ParquetManager):

    def __init__(
        self,
        path: str | Path,
        name_col: str = "name",
        val_col: str = "value",
        date_col: str = "date",
        code_col: str = "code",
        partition: str = None,
    ):
        ParquetManager.__init__(
            self, path=path, index=[date_col, code_col, name_col], partition=partition
        )
        if bool(name_col) != bool(val_col):
            raise ValueError(
                "Either both name_col and val_col are provided or neither is provided"
            )
        self.date_col = date_col
        self.code_col = code_col
        self.name_col = name_col
        self.val_col = val_col

    def get_times(
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
        return ParquetManager.read(self, **params).index.unique().sort_values()

    def get_time(self, time: str | pd.Timestamp = None, n: int = 1):
        time = pd.to_datetime(time or "now")
        if n >= 0:
            trading_days = self.get_times(begin=None, end=time)
            rollback_days = trading_days[trading_days <= time]
            n = rollback_days[max(-len(rollback_days), -n - 1)]
        else:
            trading_days = self.get_times(begin=time, end=None)
            n = trading_days[min(len(trading_days) - 1, -n)]
        return n

    def get_factor(
        self,
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
        **kwargs,
    ) -> pd.DataFrame:
        if self.name_col:
            kwargs.update(
                {
                    "pivot": self.val_col,
                    self.name_col: name,
                    "index": self.date_col,
                    "columns": self.code_col,
                }
            )
        else:
            kwargs.update(
                {"pivot": name, "index": self.date_col, "columns": self.code_col}
            )
        if begin is not None:
            kwargs.update({f"{self.date_col}__ge": begin})
        if end is not None:
            kwargs.update({f"{self.date_col}__le": end})
        df = ParquetManager.read(self, **kwargs)

        return df.sort_index().dropna(axis=0, how="all")


class DuckDBFactorSource(FactorSource):

    def __init__(self, path: str, name: str):
        self._path = path
        self._name = name

    def get_times(self, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        with duckdb.connect(self._path) as con:
            code = con.execute(f"SELECT code FROM {self._name} LIMIT 1").fetchone()
            times = con.execute(
                f"SELECT DISTINCT time FROM {self._name} WHERE code = ? AND time >= ? AND time <= ? ORDER BY time",
                (code, begin, end),
            ).fetchall()
        return pd.to_datetime(times)

    def get_time(self, time: str, n: int):
        if n > 0:
            return self.get_times(None, time)[-n:]
        return self.get_times(time, None)[:n]

    def get_factor(self, name: str, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        with duckdb.connect(self._path) as con:
            factor = con.execute(
                f"SELECT * FROM {self._name} WHERE time >= ? AND time <= ? AND name = ?",
                (begin, end, name),
            ).fetch_df()
            factor = factor.pivot(index="time", columns="code", values="value")
        return factor

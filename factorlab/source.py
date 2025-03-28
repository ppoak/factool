import duckdb
import pandas as pd
from pathlib import Path
from xtquant import xtdata
from functools import partial
from quool import ParquetManager
from abc import ABC, abstractmethod
from .operators import zscore, madoutlier


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

    @abstractmethod
    def save(self, name: str, df: pd.DataFrame):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return self.__str__()


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
            [name],
            stock_list=self._stock_list,
            period=self._period,
            start_time=begin,
            end_time=end,
            dividend_type="back",
        )
        factor = pd.concat(
            [f[name] for f in factor.values()], axis=1, keys=factor.keys()
        )
        factor.index = pd.to_datetime(factor.index)
        return factor

    def save(self, name: str, df: pd.DataFrame):
        raise ValueError("XtFactorSource does not support save operation")


class ParquetFactorSource(FactorSource):

    def __init__(
        self,
        path: str | Path,
        time_col: str = "time",
        code_col: str = "code",
        partition_col: str = None,
    ):
        self.path = Path(path)
        self.manager = ParquetManager(
            self.path,
            index_col=[time_col, code_col],
            partition_col=partition_col or "month",
        )
        self.time_col = time_col
        self.code_col = code_col

    def get_times(
        self,
        begin: str | pd.Timestamp = None,
        end: str | pd.Timestamp = None,
        code: str = "000001.XSHE",
    ):
        params = {
            "index": self.time_col,
            self.code_col: code,
        }
        if begin is not None:
            params[f"{self.time_col}__ge"] = pd.to_datetime(begin)
        if end is not None:
            params[f"{self.time_col}__le"] = pd.to_datetime(end)
        return self.manager.read(**params).index.unique().sort_values()

    def get_time(self, time: str | pd.Timestamp = None, n: int = 1):
        time = pd.to_datetime(time or "now")
        if n >= 0:
            trading_days = self.get_times(begin=None, end=time)
            rollback_days = trading_days[trading_days <= time]
            n = rollback_days[max(-len(rollback_days), -n - 1):]
        else:
            trading_days = self.get_times(begin=time, end=None)
            n = trading_days[:min(len(trading_days) - 1, -n)]
        return n

    def get_factor(
        self,
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
        **kwargs,
    ) -> pd.DataFrame:
        kwargs.update(
            {
                "pivot": name,
                "index": self.time_col,
                "columns": self.code_col,
            }
        )
        if begin is not None:
            kwargs.update({f"{self.time_col}__ge": begin})
        if end is not None:
            kwargs.update({f"{self.time_col}__le": end})
        df = self.manager.read(**kwargs)

        return df.sort_index().dropna(axis=0, how="all")

    def save(
        self,
        df: pd.DataFrame,
        partitioner: str,
        processors: list[callable] = None,
    ):
        processors = processors or [zscore, partial(madoutlier, dev=5)]
        if df.index.nlevels == 1:
            for processor in processors:
                processed = processor(df)
            df = pd.concat(
                [processed.stack(), df.stack().to_frame("_processed")], axis=1
            ).reset_index(names=["time", "code"])
            self.manager.upsert(factor, partitioner=partitioner)

        elif df.index.nlevels == 2:
            factors = [df[col].unstack() for col in df.columns]
            for processor in processors:
                factors = [processor(factor) for factor in factors]
            factor = pd.concat(
                [factor.stack() for factor in factors],
                keys=df.columns + "_processed",
                axis=1,
            )
            factor = pd.concat([factor, df], axis=1).reset_index(names=["time", "code"])
            self.manager.update_insert(factor, partitioner=partitioner)


class DuckDBFactorSource(FactorSource):

    def __init__(self, path: str):
        self._path = path

    def get_times(self, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        with duckdb.connect(self._path) as con:
            metadata = con.execute(
                f"SELECT id, class FROM metadata LIMIT 1"
            ).fetchone()[0]
            code = con.execute(
                f"SELECT code FROM {metadata[1]} WHERE id = ? LIMIT 1", (metadata[0],)
            ).fetchone()[0]
            times = con.execute(
                f"SELECT DISTINCT time FROM {metadata[1]} WHERE code = ? AND id = ? AND time >= ? AND time <= ? ORDER BY time",
                (code, metadata[0], begin, end),
            ).fetchall()
        return pd.to_datetime(times)

    def get_time(self, time: str, n: int):
        if n > 0:
            return self.get_times(None, time)[-n:]
        return self.get_times(time, None)[:n]

    def get_factor(
        self, name: str, begin: str = None, end: str = None, raw: bool = True
    ):
        begin = pd.to_datetime(begin or "1990-01-01")
        end = pd.to_datetime(end or "now")
        value = "raw" if raw else "processed"
        with duckdb.connect(self._path) as con:
            _id, _class = con.execute(
                f"SELECT id, class FROM metadata WHERE name = ?", (name,)
            ).fetchone()
            data = con.execute(
                f"SELECT code, time, {value} FROM {_class} WHERE time >= ? AND time <= ? AND id = ?",
                (begin, end, _id),
            ).fetch_df()
            data = data.pivot(index="time", columns="code", values=value)
        return data

    def save(self, name: str, df: pd.DataFrame, processors: list[callable] = None):
        raise NotImplementedError(
            "Save method for DuckDBFactorSource is not implemented"
        )

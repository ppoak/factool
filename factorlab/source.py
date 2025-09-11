import duckdb
import pandas as pd
from pathlib import Path
from functools import partial
from abc import ABC, abstractmethod
from .operators import zscore, madoutlier
from quool import ParquetManager, DuckParquet


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


class DuckParquetSource(FactorSource):

    def __init__(
        self,
        dataset_path: str,
        time_col: str = "time",
        code_col: str = "code",
        name: str = None,
        db_path: str = None,
        threads: int = 4,
    ) -> None:
        self.dp = DuckParquet(dataset_path, name, db_path, threads)
        self.time_col = time_col
        self.code_col = code_col

    def get_times(self, begin: str = None, end: str = None, time_col: str = "time"):
        times = self.dp.select(
            columns=f"{time_col} AS time",
            where="time >= ? AND time <= ?",
            params=[
                pd.to_datetime(begin or "1990-01-01"),
                pd.to_datetime(end or "now"),
            ],
            distinct=True,
            order_by="time",
        ).squeeze()
        return pd.to_datetime(times)

    def get_time(self, time: str, n: int):
        if n > 0:
            return self.get_times(None, time)[-n:]
        return self.get_times(time, None)[:-n]

    def get_all_factors(self) -> pd.DataFrame:
        schema = self.dp.get_schema()
        schema = schema[~schema["column_name"].isin([self.time_col, self.code_col])]
        return schema

    def get_factor(
        self,
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
    ) -> pd.DataFrame:
        begin = pd.to_datetime(begin or "2000-01-01")
        end = pd.to_datetime(end or "now")
        return self.dp.dpivot(
            index=self.time_col,
            columns=self.code_col,
            values=name,
            where=f"{self.time_col} >= '{begin}' AND {self.time_col} <= '{end}'",
            order_by=self.time_col,
        ).set_index(self.time_col)

    def save(
        self,
        df: pd.DataFrame,
        name: str = "factor",
        processors: list[callable] = None,
    ):
        processors = processors or [zscore, partial(madoutlier, dev=5)]
        names = "__".join(
            [
                (
                    processor.__name__
                    if not isinstance(processor, partial)
                    else processor.func.__name__
                    + "_"
                    + "_".join([f"{k}{v}" for k, v in processor.keywords.items()])
                )
                for processor in processors
            ]
        )
        if df.index.nlevels == 1:
            for processor in processors:
                processed = processor(df)
            factor = pd.concat(
                [processed.stack(), df.stack()],
                keys=[name, f"{name}__{names}"],
                axis=1,
            ).reset_index(names=[self.time_col, self.code_col])

        elif df.index.nlevels == 2:
            factors = [df[col].unstack() for col in df.columns]
            for processor in processors:
                factors = [processor(factor) for factor in factors]
            factor = pd.concat(
                [df]
                + [
                    factor.stack().to_frame(df.columns[i] + f"__{names}")
                    for i, factor in enumerate(factors)
                ],
                axis=1,
            ).reset_index(names=[self.time_col, self.code_col])

        factor["date"] = factor[self.time_col].dt.strftime("%Y-%m-%d")
        self.dp.upsert_from_df(
            factor, keys=[self.time_col, self.code_col], partition_by=["date"]
        )


class ParquetFactorSource(FactorSource):

    def __init__(
        self,
        path: str | Path,
        time_col: str = "time",
        code_col: str = "code",
        grouper: str = None,
    ):
        print(
            "Warning: ParquetFactorSource will soon be depreciated in higher version, please use DuckParquetSource instead."
        )
        self.path = Path(path)
        self.manager = ParquetManager(
            self.path,
            unikey=[time_col, code_col],
            grouper=grouper,
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
            n = rollback_days[max(-len(rollback_days), -n - 1) :]
        else:
            trading_days = self.get_times(begin=time, end=None)
            n = trading_days[: min(len(trading_days) - 1, -n)]
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
        processors: list[callable] = None,
    ):
        processors = processors or [zscore, partial(madoutlier, dev=5)]
        if df.index.nlevels == 1:
            for processor in processors:
                processed = processor(df)
            df = pd.concat(
                [processed.stack(), df.stack().to_frame("_processed")], axis=1
            ).reset_index(names=["time", "code"])
            self.manager.update(factor)

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
            self.manager.update(factor)

    def __str__(self):
        return super().__str__() + "\n" + str(self.manager)


class DuckDBFactorSource(FactorSource):

    def __init__(self, path: str):
        self.path = path
        self.manager = DuckDBManager(path)

    def get_times(self, table: str, begin: str = None, end: str = None):
        begin = pd.to_datetime(begin or "1990-01-01").strftime(r"%Y%m%d")
        end = pd.to_datetime(end or "now").strftime(r"%Y%m%d")
        times = self.manager.select(
            table,
            columns=["time"],
            ands=["time >= ?", "time <= ?"],
            params=(begin, end),
            distinct=True,
        ).squeeze()
        return pd.to_datetime(times)

    def get_time(self, table: str, time: str, n: int):
        if n > 0:
            return self.get_times(table, None, time)[-n:]
        return self.get_times(table, time, None)[:n]

    def get_factor(
        self,
        table: str,
        name: str,
        begin: str = None,
        end: str = None,
    ):
        begin = pd.to_datetime(begin or "1990-01-01")
        end = pd.to_datetime(end or "now")
        data = self.manager.pivot(
            table,
            index="time",
            columns="code",
            value=name,
            ands=["time >= ?", "time <= ?"],
            params=(begin, end),
        )
        return data

    def save(self, name: str, df: pd.DataFrame, processors: list[callable] = None):
        processors = processors or [zscore, partial(madoutlier, dev=5)]
        if df.index.nlevels == 1:
            for processor in processors:
                processed = processor(df)
            df = pd.concat(
                [processed.stack(), df.stack().to_frame("_processed")], axis=1
            ).reset_index(names=["time", "code"])
            self.manager.upsert(factor, name)

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
            self.manager.upsert(factor, name)

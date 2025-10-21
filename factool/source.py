import pandas as pd
from functools import partial
from abc import ABC, abstractmethod
from .operators import zscore, madoutlier
from parquool import DuckParquet


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
            return self.get_times(None, time).iloc[-n - 1]
        return self.get_times(time, None).iloc[-n]

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

    def __str__(self):
        return super().__str__() + f"(\n{self.get_all_factors()}\n)"

    def __repr__(self):
        return super().__repr__()

from functools import partial
from typing import Union

import pandas as pd

from parquool import DuckParquet

from .oprator import Operator


class DuckParquetSource:
    """
    High-level data source wrapper around parquool.DuckParquet for storing and retrieving
    time-series factor data keyed by a time column and a code column.

    This class provides utilities to:
    - Query the distinct available timestamps in the dataset.
    - Compute offset timestamps relative to a reference time.
    - Inspect factor columns available in the dataset.
    - Pivot a factor column into a time-by-code matrix for a given date range.
    - Save original and processed factor values back to the dataset via upsert, partitioned by date.

    Args:
        dataset_path (str): Path to a Parquet dataset directory or file to be managed
            by DuckParquet.
        time_col (str): Name of the time column used throughout queries and pivots.
            Defaults to "date".
        code_col (str): Name of the entity identifier column. Defaults to "code".
        name (Optional[str]): Optional logical dataset name used by DuckDB under the hood.
            If provided, it may be used for registration or naming contexts in the backend.
        db_path (Optional[str]): Optional DuckDB database path used by parquool for metadata
            and operations. If None, an in-memory database may be used.
        threads (int): Number of worker threads for DuckDB operations. Defaults to 4.

    Notes:
        This class does not load data eagerly on initialization; it only configures
        the backend and column names.
    """

    def __init__(
        self,
        dataset_path: str,
        time_col: str = "date",
        code_col: str = "code",
        name: str = None,
        db_path: str = None,
        threads: int = 4,
    ) -> None:
        """Initialize a DuckParquetSource for factor data backed by a Parquet dataset.

        Args:
            dataset_path (str): Path to the Parquet dataset to manage.
            time_col (str): Name of the time column (e.g., "date", "datetime").
                Defaults to "date".
            code_col (str): Name of the code/identifier column. Defaults to "code".
            name (Optional[str]): Optional logical name for the dataset in DuckDB.
            db_path (Optional[str]): Optional path to a DuckDB database used by the backend.
                If None, an in-memory database may be used.
            threads (int): Number of threads used by DuckDB operations. Defaults to 4.

        Returns:
            None

        Notes:
            - No data is read at construction time.
            - The provided column names are used in later queries and pivots.
        """
        self.dp = DuckParquet(dataset_path, name, db_path, threads)
        self.time_col = time_col
        self.code_col = code_col

    def get_times(self, begin: str = None, end: str = None, time_col: str = "date"):
        """
        Return distinct, sorted timestamps in the dataset between begin and end (inclusive).

        Args:
            begin (Optional[str]): Lower bound (inclusive) for the time range. If None,
                defaults to "1990-01-01". Accepts any string parsable by pandas.to_datetime.
            end (Optional[str]): Upper bound (inclusive) for the time range. If None,
                defaults to "now". Accepts any string parsable by pandas.to_datetime.
            time_col (str): Column name to treat as the time field for this query.
                Defaults to "date". This overrides the instance's time_col for this call.

        Returns:
            pandas.Series: A Series of pandas.Timestamp values sorted ascending.
                May be empty if no values fall within the specified range.

        Raises:
            KeyError: If the specified time_col does not exist in the dataset.
            ValueError: If begin is after end or if date parsing fails.
            Exception: Propagated errors from the storage backend.

        Examples:
            >>> source.get_times("2024-01-01", "2024-03-31")
            0   2024-01-02
            1   2024-01-03
            ...
        """

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
        """
        Return a timestamp offset from a given reference time by n available steps.

        The offset is computed using the set of distinct times in the dataset:
        - n > 0: returns the n-th previous available time before the reference time.
        For example, n=1 returns the previous available time.
        - n = 0: returns the first available time on or after the reference time
        (which is the reference time itself if it exists).
        - n < 0: returns the (-n)-th next available time on or after the reference time.
        For example, n=-1 returns the next available time.

        Args:
            time (Union[pandas.Timestamp, str]): Reference time. If a string is provided,
                it is parsed via pandas.to_datetime.
            n (int): Offset step count. Positive for previous times, zero for the
                reference-or-next time, negative for next times.

        Returns:
            pandas.Timestamp: The offset timestamp.

        Raises:
            IndexError: If the requested offset is out of range (e.g., not enough
                preceding or following timestamps).
            ValueError: If the input time cannot be parsed.
            Exception: Propagated errors from the storage backend.

        Examples:
            >>> source.get_time("2024-06-01", 1)   # previous available time
            Timestamp('2024-05-31 00:00:00')
            >>> source.get_time("2024-06-01", 0)   # on or after time
            Timestamp('2024-06-01 00:00:00')
            >>> source.get_time("2024-06-01", -2)  # second next available time
            Timestamp('2024-06-03 00:00:00')
        """

        if n > 0:
            return self.get_times(None, time).iloc[-n - 1]
        return self.get_times(time, None).iloc[-n]

    def get_all_factors(self) -> pd.DataFrame:
        """
        Return schema information for all factor columns excluding the time and code columns.

        The returned schema is derived from the underlying dataset and typically
        includes metadata per column such as the column name and types.

        Returns:
            pandas.DataFrame: A DataFrame containing schema rows for factor columns only.
                The exact columns depend on the backend's schema representation, but will
                exclude self.time_col and self.code_col.

        Raises:
            Exception: Propagated errors from the storage backend.

        Examples:
            >>> source.get_all_factors()
            column_name  column_type  ...
            0       close  DOUBLE        ...
            1        beta  DOUBLE        ...
        """

        schema = self.dp.get_schema()
        schema = schema[~schema["column_name"].isin([self.time_col, self.code_col])]
        return schema

    def get_factor(
        self,
        name: str,
        where: str = None,
        begin: Union[pd.Timestamp, str] = None,
        end: Union[pd.Timestamp, str] = None,
    ) -> pd.DataFrame:
        """
        Load a factor column and pivot it into a time-by-code matrix for a date range.

        The result is a wide DataFrame where:
        - The index is the time column (self.time_col), sorted ascending.
        - The columns are distinct codes (self.code_col).
        - The cell values are the factor values for (time, code) pairs.

        Args:
            name (str): Name of the factor column to retrieve.
            where (str): The sql condition filter for filtering certain range
            begin (Union[pandas.Timestamp, str, None]): Inclusive lower bound for the time range.
                Defaults to "2000-01-01" if None.
            end (Union[pandas.Timestamp, str, None]): Inclusive upper bound for the time range.
                Defaults to "now" if None.

        Returns:
            pandas.DataFrame: A pivoted DataFrame with index = self.time_col and columns = self.code_col.
                Missing pairs will appear as NaN.

        Raises:
            KeyError: If the specified factor column does not exist.
            ValueError: If begin is after end or date parsing fails.
            Exception: Propagated errors from the storage backend.

        Examples:
            >>> df = source.get_factor("close", begin="2023-01-01", end="2023-12-31")
            >>> df.index.name
            'date'
            >>> df.columns[:5]
            Index(['000001.SZ', '000002.SZ', ...], dtype='object')
        """

        begin = pd.to_datetime(begin or "2000-01-01")
        end = pd.to_datetime(end or "now")
        data = self.dp.dpivot(
            index=self.time_col,
            columns=self.code_col,
            values=name,
            where=f"{self.time_col} >= '{begin}' AND {self.time_col} <= '{end}'"
            + (f"AND {where}" if where else ""),
            order_by=self.time_col,
        ).set_index(self.time_col)
        data.attrs["name"] = name
        return data

    def save(
        self,
        df: pd.DataFrame,
        name: str = "factor",
        processors: list[callable] = None,
    ):
        """
        Upsert factor values into the dataset, optionally applying post-processing transforms.

        Supported input formats:
        - Wide format (index has one level): df is a time-by-code matrix where the index
        is self.time_col and columns are distinct codes. In this case, 'name' is used
        as the base factor name. The original and processed versions are both written.
        - Long format (index has two levels): df has a MultiIndex (self.time_col, self.code_col)
        and one or more columns representing factor names. Each column is unstacked to wide
        format, processed, and written alongside the original.

        Processors:
        - If provided, each processor must be a callable that accepts and returns a DataFrame
        in wide format (index = time, columns = codes), preserving the shape.
        - By default, the following processors are applied:
            Operator.zscore
            partial(Operator.madoutlier, dev=5)
        The processed output columns are suffixed with an identifier based on processor
        names and parameters (e.g., "__zscore__madoutlier_dev5").

        Behavior:
        - Adds a partition column 'date' derived from the time column formatted as "YYYY-MM-DD".
        - Performs an upsert using keys [self.time_col, self.code_col]; existing rows for the same
        keys are updated, and new rows are inserted.
        - Data is partitioned by the 'date' column to optimize storage and retrieval.

        Args:
            df (pandas.DataFrame): Input factor data in either wide or long format as described above.
                The time index must be datetime-like to allow date partitioning.
            name (str): Base name used when df is provided in wide format. Defaults to "factor".
            processors (Optional[list[callable]]): List of processing functions applied to the wide
                matrices prior to writing. If None, defaults to a z-score normalization and a MAD-based
                outlier reduction.

        Returns:
            None

        Raises:
            ValueError: If the DataFrame index structure is unsupported.
            TypeError: If any processor is not callable or returns a non-DataFrame.
            Exception: Propagated errors from the storage backend during upsert.

        Examples:
            Wide format:
                >>> df = source.get_factor("close", "2024-01-01", "2024-01-31")
                >>> source.save(df, name="close", processors=[my_norm, partial(my_clip, k=3)])

            Long (MultiIndex) format:
                >>> df_long = (
                ...     pd.DataFrame({"close": close_vals, "beta": beta_vals})
                ...       .set_index(["date", "code"])
                ... )
                >>> source.save(df_long, processors=[Operator.zscore])
        """

        processors = processors or [
            Operator.zscore,
            partial(Operator.madoutlier, dev=5),
        ]
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

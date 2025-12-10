from functools import partial
from typing import Union, Literal

import pandas as pd

from parquool import DuckPQ

from .oprator import Operator


class DuckPQSource(DuckPQ):
    """
    High-level data source wrapper around parquool.DuckPQ for storing and retrieving
    time-series factor data keyed by a time column and a code column.

    This class provides utilities to:
    - Query the distinct available timestamps in the dataset.
    - Compute offset timestamps relative to a reference time.
    - Inspect factor columns available in the dataset.
    - Pivot a factor column into a time-by-code matrix for a given date range.
    - Save original and processed factor values back to the dataset via upsert, partitioned by date.

    Args:
        root_path (str): Path to a Parquet dataset directory or file to be managed
            by DuckPQ.
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
        root_path: str,
        time_col: str = "date",
        code_col: str = "code",
        threads: int = 4,
    ) -> None:
        """Initialize a DuckPQSource for factor data backed by a Parquet dataset.

        Args:
            root_path (str): Path to the Parquet dataset to manage.
            time_col (str): Name of the time column (e.g., "date", "datetime").
                Defaults to "date".
            code_col (str): Name of the code/identifier column. Defaults to "code".
            name (Optional[str]): Optional logical name for the dataset in DuckDB.
            threads (int): Number of threads used by DuckDB operations. Defaults to 4.

        Returns:
            None

        Notes:
            - No data is read at construction time.
            - The provided column names are used in later queries and pivots.
        """
        super().__init__(root_path=root_path, threads=threads)
        self.time_col = time_col
        self.code_col = code_col

    def get_factor(
        self,
        table: str,
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
            table (str): Name of the factor table.
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
        data = self.select(
            table=table,
            columns=[self.code_col, self.time_col, name],
            where=f"{self.time_col} >= '{begin}' AND {self.time_col} <= '{end}'"
            + (f"AND {where}" if where else ""),
            order_by=self.time_col,
        )
        data = data.pivot(index=self.time_col, columns=self.code_col, values=name)
        data.attrs["name"] = name
        return data

    def save(
        self,
        table_name: str,
        df: pd.DataFrame,
        processors: list[callable] = None,
    ):
        """
        Upsert factor values into the dataset, optionally applying post-processing transforms.

        Supported input formats:
            Long format (index has two levels): df has a MultiIndex (self.time_col, self.code_col)
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
            table_name (str): The table name of the factor.
            df (pandas.DataFrame): Input factor data in long format as described above.
                The time index must be datetime-like to allow date partitioning.
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

        if df.index.nlevels == 2 and isinstance(df, pd.DataFrame):
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
        else:
            raise ValueError(
                "Invalid data form, input factor data must be two level (date+code) indexed DataFrame"
            )

        factor["date"] = factor[self.time_col].dt.strftime("%Y-%m-%d")
        self.upsert(
            table_name,
            factor,
            keys=[self.time_col, self.code_col],
            partition_by=["date"],
        )

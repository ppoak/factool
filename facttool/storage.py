import re
from functools import partial
from typing import Union, Iterable, Tuple, Dict, List, Optional

import pandas as pd

from parquool import DuckPQ


def parse_factor_path(path: str, sep: str = "/") -> Tuple[str, str, Optional[str]]:
    if not isinstance(path, str):
        raise TypeError(f"factor path must be str, got {type(path)}: {path}")
    s = path.strip()
    if not s:
        raise ValueError("empty factor path")
    parts = s.split(sep)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"invalid factor path: {path!r}, expected 'table{sep}factor' or 'table{sep}factor AS alias'"
        )
    table = parts[0].strip()
    rhs = parts[1].strip()
    split_as = re.split(r"\s+AS\s+", rhs, maxsplit=1, flags=re.IGNORECASE)
    if len(split_as) == 2:
        factor = split_as[0].strip()
        alias = split_as[1].strip()
        if not factor or not alias:
            raise ValueError(
                f"invalid factor path: {path!r}, expected 'table{sep}factor' or 'table{sep}factor AS alias'"
            )
        return table, factor, alias
    else:
        return table, rhs, None


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

    def load(
        self,
        factor_paths: Union[str, Iterable[str]],
        begin: str = None,
        end: str = None,
        *,
        sep: str = "/",
        join: str = "full",
        pad_begin: int = 0,
        pad_end: int = 0,
        base_table: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads one or many factors and joins them in DuckDB.
        Args:
            factor_paths: Iterable of factor paths like "table/factor" or "table/factor AS alias".
            begin: Begin date (inclusive), passed into SQL as a string.
            end: End date (inclusive), passed into SQL as a string.
            sep: Path separator used in `factor_paths`.
            join: Join type between tables: "inner", "left", "right", or "full".
            pad_begin: Relative to begin, pad more data (0 for no padding).
            pad_end: Relative to end, pad more data (0 for no padding).
            base_table: The anchor table name used as the left side of the join chain.
                If None, the first table in `factor_paths` is used.
        Returns:
            A pandas DataFrame indexed by ["date", "code"] with factor columns (aliases applied if provided).
        Raises:
            ValueError: If inputs are invalid.
            TypeError: If a factor path is not a string.
        """
        factor_paths = (
            list(factor_paths) if not isinstance(factor_paths, str) else [factor_paths]
        )
        if not factor_paths:
            raise ValueError("factor_paths is empty")
        if not isinstance(pad_begin, int):
            raise TypeError("lookback must be int")
        if pad_begin < 0:
            raise ValueError("lookback must be >= 0")
        if not isinstance(pad_end, int):
            raise TypeError("lookforward must be int")
        if pad_end < 0:
            raise ValueError("lookforward must be >= 0")
        join = join.lower()
        join_map = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "full": "FULL OUTER JOIN",
        }
        if join not in join_map:
            raise ValueError(f"invalid join={join!r}, choose from {list(join_map)}")

        # by_table: {table: [(column, alias_name), ...]}
        by_table: Dict[str, List[Tuple[str, str]]] = {}
        for p in factor_paths:
            t, c, alias = parse_factor_path(p, sep=sep)
            alias_name = alias if alias is not None else c
            lst = by_table.setdefault(t, [])
            if all(existing_alias != alias_name for _, existing_alias in lst):
                lst.append((c, alias_name))

        tables = list(by_table.keys())
        if base_table is None:
            base_table = tables[0]
        if base_table not in by_table:
            raise ValueError(
                f"base_table {base_table!r} is not present in factor_paths tables: {tables}"
            )

        # Register all tables up front.
        for t in tables:
            self.register(t)

        # ---- compute lookback/lookforward bounds (based on base_table calendar) ----
        begin_for_sql = begin or self.get_ealiest_date(base_table)
        end_for_sql = end or self.get_latest_date(base_table)
        if pad_begin > 0:
            sql_begin_lb = f"""
            WITH cal AS (
                SELECT DISTINCT CAST(date AS DATE) AS d
                FROM {base_table}
            ),
            anchor AS (
                SELECT d
                FROM cal
                WHERE d <= CAST('{begin}' AS DATE)
                ORDER BY d DESC
                LIMIT 1
            )
            SELECT CAST(d AS VARCHAR) AS begin_lb
            FROM cal
            WHERE d <= (SELECT d FROM anchor)
            ORDER BY d DESC
            OFFSET {pad_begin - 1}
            LIMIT 1
            """.strip()
            tmp = self.query(sql_begin_lb)
            if tmp.empty or tmp.iloc[0, 0] is None:
                sql_min = f"""
                SELECT CAST(MIN(CAST(date AS DATE)) AS VARCHAR) AS begin_lb
                FROM {base_table}
                """.strip()
                tmp2 = self.query(sql_min)
                begin_for_sql = tmp2.iloc[0, 0]
            else:
                begin_for_sql = tmp.iloc[0, 0]
        if pad_end > 0:
            sql_end_lf = f"""
            WITH cal AS (
                SELECT DISTINCT CAST(date AS DATE) AS d
                FROM {base_table}
            ),
            anchor AS (
                SELECT d
                FROM cal
                WHERE d >= CAST('{end}' AS DATE)
                ORDER BY d ASC
                LIMIT 1
            )
            SELECT CAST(d AS VARCHAR) AS end_lf
            FROM cal
            WHERE d >= (SELECT d FROM anchor)
            ORDER BY d ASC
            OFFSET {pad_end}
            LIMIT 1
            """.strip()
            tmp = self.query(sql_end_lf)
            if tmp.empty or tmp.iloc[0, 0] is None:
                sql_max = f"""
                SELECT CAST(MAX(CAST(date AS DATE)) AS VARCHAR) AS end_lf
                FROM {base_table}
                """.strip()
                tmp2 = self.query(sql_max)
                end_for_sql = tmp2.iloc[0, 0]
            else:
                end_for_sql = tmp.iloc[0, 0]

        def subquery(table: str) -> str:
            cols = ", ".join(sorted({col for col, _ in by_table[table]}))
            return f"""
                SELECT
                    CAST(date AS TIMESTAMP) AS date,
                    code,
                    {cols}
                FROM {table}
                WHERE date >= '{begin_for_sql}' AND date <= '{end_for_sql}'
            """.strip()

        # Build a single SQL statement joining per-table subqueries.
        base_alias = "b"
        sql_from = f"FROM ({subquery(base_table)}) AS {base_alias}\n"
        key_date = f"{base_alias}.date"
        key_code = f"{base_alias}.code"
        i = 0
        for t in tables:
            if t == base_table:
                continue
            i += 1
            a = f"t{i}"
            sql_from += (
                f"{join_map[join]} ({subquery(t)}) AS {a}\n"
                f"ON {a}.date = {key_date} AND {a}.code = {key_code}\n"
            )
            if join == "full":
                key_date = f"COALESCE({key_date}, {a}.date)"
                key_code = f"COALESCE({key_code}, {a}.code)"

        select_cols: List[str] = [f"{key_date} AS date", f"{key_code} AS code"]
        # base_table column (as alias)
        for col, alias_name in by_table[base_table]:
            select_cols.append(f"{base_alias}.{col} AS {alias_name}")
        # other columns from table (as alias)
        i = 0
        for t in tables:
            if t == base_table:
                continue
            i += 1
            a = f"t{i}"
            for col, alias_name in by_table[t]:
                select_cols.append(f"{a}.{col} AS {alias_name}")

        sql = "SELECT\n    " + ",\n    ".join(select_cols) + "\n" + sql_from
        df = self.query(sql)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "code"]).sort_index()
        return df

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

        processors = processors or []
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

    def get_latest_date(self, table: str):
        self.register(table)
        if self.tables[table].empty:
            return
        date = self.query(f"SELECT MAX({self.time_col}) FROM {table}").squeeze()
        return date

    def get_ealiest_date(self, table: str):
        self.register(table)
        if self.tables[table].empty:
            return
        date = self.query(f"SELECT MIN({self.time_col}) FROM {table}").squeeze()
        return date

    def get_date_gap(
        self,
        target_table: str,
        base_table: str,
        default: str = "1900-01-01",
    ):
        target_latest = self.get_latest_date(target_table) or default
        base_latest = self.get_latest_date(base_table) or default
        return target_latest, base_latest

    def get_date_range(
        self,
        date: str,
        n: int = 1,
        table: str = "quotes_day",
    ):
        self.register(table)
        rollback = (
            self.query(
                f"""
            SELECT DISTINCT {self.time_col} AS datetime
            FROM {table}
            WHERE {self.time_col} {'<' if n > 0 else '>'} '{date}'::DATE
            ORDER BY datetime {'DESC' if n > 0 else 'ASC'}
            LIMIT {abs(n)}
            """
            )
            .squeeze()
            .sort_values()
            .reset_index(drop=True)
        )
        return rollback

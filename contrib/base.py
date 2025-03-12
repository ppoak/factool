import duckdb
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from factorlab import operators
from factorlab import XtFactorSource


class XtFactorDuckDB(XtFactorSource):

    def __init__(
        self, path: str, duckdb: str, sector: str = "沪深A股", period: str = "1d"
    ):
        super().__init__(path, sector=sector, period=period)
        self.duckdb = Path(duckdb)
        self.duckdb.parent.mkdir(parents=True, exist_ok=True)
        self._name = None
        self._data = None

    def calc(self, name: str, begin: str, end: str, n_jobs: int = -1):
        trading_days = self.get_times(begin, end)
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(getattr(self, "calc_" + name))(date)
            for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            self._data = pd.concat(result, axis=1, keys=trading_days).T.sort_index()
            self._name = name
            return self
        elif isinstance(result[0], pd.DataFrame):
            self._data = pd.concat(result, axis=0).sort_index()
            self._name = name
            return self

    def _pandas_dtype_to_duckdb_type(self, dtype: type) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            return "DOUBLE"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP"
        elif pd.api.types.is_string_dtype(dtype):
            return "VARCHAR"
        elif pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        else:
            return "VARCHAR"

    def upsert(self, df: pd.DataFrame):
        conn = duckdb.connect(self.duckdb)
        table_name = self.__class__.__name__.lower()

        # make sure the dataframe has the required columns
        required_columns = {"code", "time"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Required columns 'code' and 'time' missing")

        # check existence of table
        table_exists = (
            conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            > 0
        )

        if not table_exists:
            create_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join([f'{col} {self._pandas_dtype_to_duckdb_type(dtype)}' for col, dtype in zip(df.columns, df.dtypes)])},
                PRIMARY KEY (code, time)
            )
            """
            create_index_sql = (
                f"CREATE INDEX idx_{table_name} ON {table_name} (code, time)"
            )
            conn.execute(create_sql)
            conn.execute(create_index_sql)

        temp_table = f"temp_{table_name}"
        conn.register(temp_table, df)

        upsert_sql = f"""
        INSERT OR REPLACE INTO {table_name} 
        SELECT * FROM {temp_table}
        """

        conn.execute(upsert_sql)
        conn.commit()
        conn.unregister(temp_table)

    def save(self, processors: list[callable] = None):
        if not hasattr(self, "_data"):
            raise ValueError("Factor not calculated yet")

        if not processors:
            processors = [operators.zscore, partial(operators.madoutlier, dev=5)]

        if self._data.index.nlevels == 1:
            if processors:
                factor = self._data
                for processor in processors:
                    factor = processor(factor)
            factor = pd.concat(
                [factor.stack(), self._data.stack()], axis=1
            ).reset_index()
            factor.columns = ["time", "code", "processed", "raw"]
            factor["name"] = self._name
            factor = factor[["time", "code", "name", "processed", "raw"]]
            self.upsert(factor)

        elif self._data.index.nlevels == 2:
            factors = [self._data[col].unstack() for col in self._data.columns]
            if processors:
                for processor in processors:
                    factors = [processor(factor) for factor in factors]
            for factor, col in zip(factors, self._data.columns):
                fact = pd.concat(
                    [factor.stack(), self._data[col]], axis=1
                ).reset_index()
                fact.columns = ["time", "code", "processed", "raw"]
                fact["name"] = col
                fact = fact[["time", "code", "name", "processed", "raw"]]
                self.upsert(fact)

    def __str__(self):
        return f"{self.__class__.__name__}(duckdb={self.duckdb}, name={self._name}, factor={self._data})"

    def __repr__(self):
        return self.__str__()

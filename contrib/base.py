import duckdb
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
from joblib import Parallel, delayed
from pathlib import Path
from factorlab import operators
from factorlab import XtFactorSource


class DuckDBFactor(XtFactorSource):

    def __init__(
        self,
        qmt_path: str,
        duckdb_path: str,
        sector: str = "沪深A股",
        period: str = "1d",
    ):
        super().__init__(qmt_path, sector=sector, period=period)
        self.duckdb_path = Path(duckdb_path)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.table_name = self.__class__.__name__.lower()
        with duckdb.connect(self.duckdb_path) as con:
            # Create metadata table if not exists
            con.execute(
                "CREATE TABLE IF NOT EXISTS metadata "
                "(id BIGINT PRIMARY KEY, name VARCHAR NOT NULL UNIQUE, class VARCHAR NOT NULL)"
            )
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} "
                "(time TIMESTAMP NOT NULL, code VARCHAR NOT NULL, "
                "id BIGINT REFERENCES metadata(id), processed DOUBLE PRECISION, "
                "raw DOUBLE PRECISION, PRIMARY KEY (code, time, id), "
                "FOREIGN KEY (id) REFERENCES metadata(id))"
            )
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_metadata ON metadata (id)")
            con.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name} ON {self.table_name} (code, time, id)"
            )
            con.commit()
        self._data = None

    def _get_factor_id(self, name: str):
        with duckdb.connect(self.duckdb_path) as con:
            factor_id = con.execute(
                f"SELECT id FROM metadata WHERE name = '{name}'"
            ).fetchone()
            if factor_id is None:
                factor_id = con.execute("SELECT MAX(id) FROM metadata").fetchone()[0] or 0
                con.execute(
                    f"INSERT INTO metadata (name, id, class) "
                    f"VALUES (?, ?, ?)", (name, factor_id + 1, self.table_name)
                )
                con.commit()
                return factor_id + 1
        return factor_id[0]

    def calc(self, name: str, begin: str, end: str, n_jobs: int = -1):
        trading_days = self.get_times(begin, end)
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(getattr(self, "calc_" + name))(date)
            for date in tqdm(list(trading_days))
        )
        self._name = name
        if isinstance(result[0], pd.Series):
            self._data = pd.concat(result, axis=1, keys=trading_days).T.sort_index()
            return self
        elif isinstance(result[0], pd.DataFrame):
            self._data = pd.concat(result, axis=0, keys=trading_days).sort_index()
            return self

    def upsert(self, df: pd.DataFrame):
        conn = duckdb.connect(self.duckdb_path)

        # make sure the dataframe has the required columns
        required_columns = {"code", "time"}
        if not required_columns.issubset(df.columns):
            raise ValueError("Required columns 'code' and 'time' missing")

        temp_table = f"temp_{self.table_name}"
        conn.register(temp_table, df)

        upsert_sql = f"""
        INSERT OR REPLACE INTO {self.table_name} 
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
            factor["id"] = self._get_factor_id(self._name)
            factor = factor[["time", "code", "id", "processed", "raw"]]
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
                fact["id"] = self._get_factor_id(col)
                fact = fact[["time", "code", "id", "processed", "raw"]]
                self.upsert(fact)

    def __str__(self):
        return (
            f"{self.__class__.__name__}(duckdb={self.duckdb_path}; factor={self._data})"
        )

    def __repr__(self):
        return self.__str__()

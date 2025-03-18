import duckdb
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from joblib import Parallel, delayed
from factorlab import operators, FactorSource


class BaseFactor:

    def __init__(
        self,
        source: FactorSource,
    ):
        self.source = source

    def calc(self, name: str, begin: str, end: str, n_jobs: int = -1):
        trading_days = self.source.get_times(begin, end)
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(getattr(self, "calc_" + name))(date)
            for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1, keys=trading_days).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0, keys=trading_days).sort_index()

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self.source})"
        )

    def __repr__(self):
        return self.__str__()

import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from factorlab import FactorSource


class BaseFactor:

    def __init__(
        self,
        sources: list[FactorSource] | FactorSource,
    ):
        if not isinstance(sources, list):
            sources = [sources]
        self.sources = sources
        self.source = sources[0]

    def calc(self, name: str, times: list, n_jobs: int = -1):
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(getattr(self, "calc_" + name))(date)
            for date in tqdm(list(times))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1, keys=times).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0, keys=times).sort_index()

    def __str__(self):
        return f"{self.__class__.__name__}({self.sources})"

    def __repr__(self):
        return self.__str__()

import os
import factool
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from joblib import Parallel, delayed


def calc(name: str, times: list, n_jobs: int = -1):
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(getattr(factool.contrib, "calc_" + name))(date)
        for date in tqdm(list(times))
    )
    if isinstance(result[0], pd.Series):
        return pd.concat(result, axis=1, keys=times).T.sort_index()
    elif isinstance(result[0], pd.DataFrame):
        return pd.concat(result, axis=0, keys=times).sort_index()


load_dotenv()
factor_name = "market_size"
source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
times = source.get_times("2015-01-02", "now")
factor_data = calc(factor_name, times, 14)

factor_db = factool.DuckParquetSource(
    Path(os.getenv("FACTORLAB_BASE_PATH")) / factor_name
)
factor_db.save(factor_data)

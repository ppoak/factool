import os
import sys
import factool
import importlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from joblib import Parallel, delayed


def calc(
    factor_def_path: str, begin: str, end: str, n_jobs: int = -1, save_path: str = None
):
    factor_name = Path(factor_def_path).stem
    sys.path.insert(0, str(Path(factor_def_path).parent))
    module = importlib.import_module(factor_name)
    factor_func = getattr(module, f"calc_{factor_name}")

    source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    times = source.get_times(begin, end, time_col="date")
    result = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(factor_func)(date) for date in tqdm(list(times))
    )
    if isinstance(result[0], pd.Series):
        factor_data = pd.concat(result, axis=1, keys=times).T.sort_index()
    elif isinstance(result[0], pd.DataFrame):
        factor_data = pd.concat(result, axis=0, keys=times).sort_index()

    if save_path:
        factool.DuckParquetSource(Path(save_path) / factor_func.__name__[5:]).save(
            factor_data
        )

    return factor_data


if __name__ == "__main__":
    load_dotenv()
    factor_file_path = "contrib/market_size.py"

    print(
        calc(
            factor_file_path,
            "2025-07-01",
            "now",
            n_jobs=14,
            save_path=os.getenv("FACTOR_DATA_PATH"),
        )
    )

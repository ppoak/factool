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
    factor_def_path: str,
    begin: str,
    end: str,
    n_jobs: int = -1,
    save_path: str = None,
    chunck: int = None,
    return_df: bool = True,
):
    """Calculate factor data

    Args:
        factor_def_path (str): factor definition file path,
        begin (str): calculation begin time,
        end (str): calculation end time,
        n_jobs (int): cpu used for calculation,
        save_path (str): where to put the factor data file, if not set, no saving in disk
        chunck (int): save every `chunck` timepoints calculated,
        return_df (bool): whether return dataframe after calculation,
            if data is extremly large, we don't recommend return_df

    Return:
        (None or DataFrame) the calculation results (`return_df = True`) and saved factor data (`save_path is not None`)
    """
    factor_name = Path(factor_def_path).stem
    sys.path.insert(0, str(Path(factor_def_path).parent))
    module = importlib.import_module(factor_name)
    factor_func = getattr(module, f"calc_{factor_name}")

    source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    times = list(source.get_times(begin, end, time_col="date"))
    if not times:
        return pd.DataFrame()

    if chunck is None or chunck <= 0:
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(factor_func)(date) for date in tqdm(times)
        )
        if isinstance(result[0], pd.Series):
            factor_data = pd.concat(result, axis=1, keys=times).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            factor_data = pd.concat(result, axis=0, keys=times).sort_index()

        if save_path:
            factool.DuckParquetSource(Path(save_path)).save(
                factor_data, name=factor_name
            )
        return factor_data

    save_ds = factool.DuckParquetSource(Path(save_path)) if save_path else None
    chunks = []
    total = len(times)
    for start in tqdm(range(0, total, chunck)):
        sub_times = times[start : start + chunck]
        result = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(factor_func)(date) for date in sub_times
        )
        if isinstance(result[0], pd.Series):
            factor_chunk = pd.concat(result, axis=1, keys=sub_times).T.sort_index()
        elif isinstance(result[0], pd.DataFrame):
            factor_chunk = pd.concat(result, axis=0, keys=sub_times).sort_index()
        else:
            raise TypeError("factor_func must return pandas Series or DataFrame")

        if save_ds is not None:
            save_ds.save(factor_chunk, name=factor_name)

        if return_df:
            chunks.append(factor_chunk)

    if return_df:
        return pd.concat(chunks, axis=0).sort_index()
    return None


if __name__ == "__main__":
    load_dotenv()
    from parquool import notify_task

    factor_name = "barra_sizes"
    factor_file_path = f"generated/{factor_name}.py"
    notifier = notify_task()

    print(
        notifier(calc)(
            factor_file_path,
            "2015-01-01",
            "now",
            n_jobs=-1,
            save_path=Path(os.getenv("FACTOR_DATA_PATH")) / factor_name,
            chunck=21,
            return_df=False,
        )
    )

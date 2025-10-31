import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Union
from factool import DuckParquetSource


def calc_barra_sizes(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    shares = source.get_factor("circulation_a", begin=time, end=time).squeeze()
    price = source.get_factor("close", begin=time, end=time).squeeze()
    log_market_size = np.log(shares * price)
    log_market_size.name = "log_market_size"
    valid = log_market_size.dropna()
    if valid.shape[0] >= 2:
        X = sm.add_constant(valid)
        y = valid**3
        model = sm.OLS(y, X).fit()
        nonlinear = model.resid.reindex(log_market_size.index)
    else:
        nonlinear = pd.Series(index=log_market_size.index, dtype=float)
    nonlinear.name = "nonlinear_market_size"
    return pd.concat([log_market_size, nonlinear], axis=1)
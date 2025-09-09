import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ..source import DuckParquetSource


def calc_market_sizes(time: str | pd.Timestamp) -> pd.Series:
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    shares = source.get_factor("circulation_a", begin=time, end=time)
    price = source.get_factor("close_post", begin=time, end=time)
    log = np.log(shares * price).squeeze()
    model = sm.OLS((log**3).dropna(), sm.add_constant(log).dropna()).fit()
    nonlinear = model.resid
    return pd.concat(
        [log, nonlinear], axis=1, keys=["log_market_size", "nonlinear_market_size"]
    )

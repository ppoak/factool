import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Union
from factool import DuckParquetSource


def calc_naive_market_size(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    """
    Calculate log_market_size and nonlinear_market_size for a single time slice.

    Returns a DataFrame with two columns:
    - log_market_size: log(circulation_a * close_post)
    - nonlinear_market_size: residuals from OLS of x^3 on (const, x)
    """
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    shares = source.get_factor("circulation_a", begin=time, end=time)
    price = source.get_factor("close_post", begin=time, end=time)

    # Element-wise market cap
    market_cap = (shares * price).squeeze()
    # Compute log market size, handle non-positive by converting to NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ms = np.log(market_cap)
    log_ms = log_ms.replace([np.inf, -np.inf], np.nan)

    # Prepare nonlinear factor: regress x^3 on const and x, use residuals
    x = log_ms.copy()
    x_valid = x.dropna()
    if x_valid.empty:
        nonlinear = pd.Series(index=x.index, dtype=float)
    else:
        y = x_valid ** 3
        X = sm.add_constant(x_valid)
        model = sm.OLS(y, X).fit()
        resid = model.resid
        nonlinear = resid.reindex(x.index)
    result = pd.concat(
        [log_ms, nonlinear], axis=1, keys=["log_market_size", "nonlinear_market_size"]
    )
    return result
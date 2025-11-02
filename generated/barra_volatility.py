import os
import warnings
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_barra_volatility(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    T = 252
    M = 21
    months = 12

    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    begin = source.get_time(time, T)

    close_post = source.get_factor("close_post", begin=begin, end=time)
    st = source.get_factor("st", begin=begin, end=time)
    suspended = source.get_factor("suspended", begin=begin, end=time)

    mask = ~(st | suspended)
    close_post = close_post.where(mask)

    # Daily returns from adjusted close
    returns = close_post.pct_change(fill_method=None)

    # Excessive return volatility (EWM of squared cross-sectional de-meaned returns)
    xsec_mean = returns.mean(axis=1)
    excessive_return = returns.sub(xsec_mean, axis=0)
    ewm_var_series = excessive_return.pow(2).ewm(halflife=42, adjust=False).mean().iloc[-1]
    ewm_var_series.name = "excessive_return_volatility"

    # Yearly excessive deviation (range of 12 monthly log-return sums, each month = 21 trading days)
    log_ret = np.log1p(returns)
    monthly_sums = []
    for i in range(months):
        block = log_ret.iloc[i * M : (i + 1) * M]
        monthly_sums.append(block.sum(axis=0, skipna=True))
    monthly_sums_df = pd.DataFrame(monthly_sums)
    yearly_deviation = monthly_sums_df.max(axis=0) - monthly_sums_df.min(axis=0)
    yearly_deviation.name = "yearly_excessive_deviation"

    # Residual volatility: std of residuals from market return * beta
    close = source.get_factor("close", begin=begin, end=time)
    circulation_a = source.get_factor("circulation_a", begin=begin, end=time)
    mcap = (close * circulation_a).where(mask)

    weights = mcap.shift(1).reindex(returns.index)
    market_return = (returns * weights).sum(axis=1) / weights.sum(axis=1)

    residual_series: pd.Series
    beta = DuckParquetSource("data/barra_beta").get_factor("barra_beta", begin=begin, end=time).reindex(returns.index)
    residuals = returns - beta.mul(market_return, axis=0)
    residual_series = residuals.std(axis=0, ddof=1)
    residual_series.name = "residual_volatility"

    result = pd.concat(
        [ewm_var_series, yearly_deviation, residual_series],
        axis=1,
    )
    return result


if __name__ == "__main__":
    print(calc_barra_volatility("2025-01-02"))
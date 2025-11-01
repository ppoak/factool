import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Union
from factool import DuckParquetSource


def calc_barra_beta(time: Union[str, pd.Timestamp]) -> pd.Series:
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    begin_time = source.get_time(time, 252)

    close_post = source.get_factor("close_post", begin=begin_time, end=time)
    st = source.get_factor("st", begin=begin_time, end=time)
    suspended = source.get_factor("suspended", begin=begin_time, end=time)
    close = source.get_factor("close", begin=begin_time, end=time)
    circ_a = source.get_factor("circulation_a", begin=begin_time, end=time)

    # Valid trading mask: exclude ST and suspended
    st_mask = (~st.fillna(False))
    suspended_mask = (~suspended.fillna(False))
    valid_mask = st_mask & suspended_mask

    # Returns using adjusted close; require both t and t-1 to be valid
    returns = close_post.pct_change(fill_method=None)
    valid_returns_mask = valid_mask & valid_mask.shift(1)
    returns = returns.where(valid_returns_mask)

    # Market cap weights
    market_cap = close * circ_a
    market_cap_valid = market_cap.where(valid_mask, 0.0)

    # Market return: value-weighted by valid market cap and non-missing returns
    cap_for_returns = market_cap_valid.where(returns.notna(), 0.0)
    total_cap = cap_for_returns.sum(axis=1)
    weighted_sum = (returns.fillna(0.0) * cap_for_returns).sum(axis=1)
    market_return = weighted_sum / total_cap.replace(0.0, np.nan)

    # Exponential weights with half-life 63 trading days
    idx = market_return.index
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float, name="barra_beta")
    ages = np.arange(n)
    ewm_weights = 0.5 ** ((n - 1 - ages) / 63.0)
    ewm_w_series = pd.Series(ewm_weights, index=idx)

    # Estimate beta for each stock via WLS: Ri = alpha + beta * RM
    betas = {}
    last_day = pd.to_datetime(time)
    last_mask = valid_mask.loc[last_day] if last_day in valid_mask.index else pd.Series(False, index=returns.columns)
    last_available = close_post.loc[last_day] if last_day in close_post.index else pd.Series(np.nan, index=returns.columns)

    for code in returns.columns:
        y = returns[code]
        # Only compute for stocks available and valid at the last day
        if not last_mask.get(code, False) or pd.isna(last_available.get(code, np.nan)):
            betas[code] = np.nan
            continue

        common = y.notna() & market_return.notna()
        if common.sum() < 30:
            betas[code] = np.nan
            continue

        y_vec = y[common].values
        x_vec = market_return[common].values
        w_vec = ewm_w_series[common].values

        try:
            model = sm.WLS(y_vec, sm.add_constant(x_vec), weights=w_vec).fit()
            betas[code] = float(model.params[1])
        except Exception:
            betas[code] = np.nan

    beta_series = pd.Series(betas, dtype=float)
    beta_series.name = "barra_beta"
    return beta_series

if __name__ == "__main__":
    print(calc_barra_beta("2025-01-02"))
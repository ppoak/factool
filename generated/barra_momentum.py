import os
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_barra_momentum(time: Union[str, pd.Timestamp]) -> pd.Series:
    T = 504
    L = 21
    halflife = 126

    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    begin = source.get_time(time, T + L)

    close = source.get_factor("close_post", begin=begin, end=time)
    st = source.get_factor("st", begin=begin, end=time)
    suspended = source.get_factor("suspended", begin=begin, end=time)

    valid_days = (~st) & (~suspended)
    close = close.where(valid_days, np.nan)

    log_close = np.log(close)
    log_ret = log_close.diff().dropna(axis=0, how='all')

    if log_ret.shape[0] < (T + L):
        # Not enough data to compute the factor; return all NaNs with appropriate index
        return pd.Series(index=log_ret.columns, dtype=float, name="nonrecent_momentum")

    # Reverse so index 0 is the most recent day
    log_ret_rev = log_ret.iloc[::-1]

    weights = np.power(0.5, np.arange(T + L) / halflife)
    w_recent = weights[:L]
    w_nonrecent = weights[L:L + T]

    recent_num = log_ret_rev.iloc[:L].mul(w_recent[:, None], axis=0).sum(axis=0)
    recent_den = w_recent.sum()
    recent_avg = recent_num / recent_den

    nonrecent_num = log_ret_rev.iloc[L:L + T].mul(w_nonrecent[:, None], axis=0).sum(axis=0)
    nonrecent_den = w_nonrecent.sum()
    nonrecent_avg = nonrecent_num / nonrecent_den

    factor = nonrecent_avg - recent_avg

    factor.name = "nonrecent_momentum"
    return factor

if __name__ == "__main__":
    print(calc_barra_momentum("2025-01-02"))
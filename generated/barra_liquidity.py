import os
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_barra_liquidity(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")

    # Determine window starts for 21, 63, 252 trading days (inclusive of 'time')
    begin_252 = source.get_time(time, 252)

    # Fetch data windows
    vol = source.get_factor("volume", begin=begin_252, end=time)
    cap = source.get_factor("circulation_a", begin=begin_252, end=time)
    vol_21 = vol.iloc[-21:]
    cap_21 = cap.iloc[-21:]
    vol_63 = vol.iloc[-63:]
    cap_63 = cap.iloc[-63:]
    vol_252 = vol.iloc[-252:]
    cap_252 = cap.iloc[-252:]

    # Sum over the window (rows are dates, columns are codes)
    sum_vol_21 = vol_21.sum(axis=0)
    sum_cap_21 = cap_21.sum(axis=0)

    sum_vol_63 = vol_63.sum(axis=0)
    sum_cap_63 = cap_63.sum(axis=0)

    sum_vol_252 = vol_252.sum(axis=0)
    sum_cap_252 = cap_252.sum(axis=0)

    # Compute ratios and take log, handling invalid values
    def safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        ratio = num / den
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        ratio[ratio <= 0] = np.nan
        return np.log(ratio)

    monthly_turnover = safe_log_ratio(sum_vol_21, sum_cap_21)
    quartly_turnover = safe_log_ratio(sum_vol_63, sum_cap_63)
    annually_turnover = safe_log_ratio(sum_vol_252, sum_cap_252)

    # Mask using st and suspended at time t
    st: pd.Series = source.get_factor("st", begin=time, end=time).squeeze()
    suspended: pd.Series = source.get_factor("suspended", begin=time, end=time).squeeze()
    mask = (~st) & (~suspended)

    monthly_turnover = monthly_turnover.where(mask, np.nan)
    quartly_turnover = quartly_turnover.where(mask, np.nan)
    annually_turnover = annually_turnover.where(mask, np.nan)

    return pd.concat(
        [monthly_turnover, quartly_turnover, annually_turnover],
        axis=1,
        keys=["monthly_turnover", "quartly_turnover", "annually_turnover"],
    )


if __name__ == "__main__":
    print(calc_barra_liquidity("2025-01-02"))
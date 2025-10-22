import os
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_capital_gain_overhang(time: Union[str, pd.Timestamp]) -> pd.Series:
    T = 260  # lookback weeks
    source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")

    if isinstance(time, str):
        time = pd.Timestamp(time)
    else:
        time = pd.to_datetime(time)

    # fetch a window of daily data to cover T weeks plus buffer
    days_back = int(T * 7 + 30)
    begin = (time - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = time.strftime("%Y-%m-%d")

    close_post = source.get_factor("close_post", begin=begin, end=end)
    volume_post = source.get_factor("volume_post", begin=begin, end=end)
    circ = source.get_factor("circulation_a", begin=begin, end=end)
    suspended = source.get_factor("suspended", begin=begin, end=end)

    # Align columns: keep intersection
    common_cols = sorted(
        set(close_post.columns)
        & set(volume_post.columns)
        & set(circ.columns)
        & set(suspended.columns)
    )
    if len(common_cols) == 0:
        return pd.Series(dtype=float, name="capital_gain_overhang")

    close_post = close_post[common_cols]
    volume_post = volume_post[common_cols]
    circ = circ[common_cols]
    suspended = suspended[common_cols]

    # Ensure datetime index
    close_post.index = pd.to_datetime(close_post.index)
    volume_post.index = pd.to_datetime(volume_post.index)
    circ.index = pd.to_datetime(circ.index)
    suspended.index = pd.to_datetime(suspended.index)

    # compute week periods (W-FRI) based on the close_post index (covers trading calendar)
    week_period = close_post.index.to_period("W-FRI")

    # Weekly price: last available close in the week (last trading day of that week)
    P_week = close_post.groupby(week_period, sort=True).last()
    # convert period index to timestamp at period end
    P_week.index = P_week.index.to_timestamp(how="end")

    # Preprocess circulation: forward fill then fill remaining with 0
    circ_ffill = circ.ffill().fillna(0.0)

    # For suspended days, set volume to 0
    suspended_bool = suspended.replace(np.nan, False).astype(bool)
    volume_masked = volume_post.where(~suspended_bool, 0.0)

    # Weekly sums for volume and circulation
    sum_vol = volume_masked.groupby(week_period, sort=True).sum()
    sum_vol.index = sum_vol.index.to_timestamp(how="end")
    sum_circ = circ_ffill.groupby(week_period, sort=True).sum()
    sum_circ.index = sum_circ.index.to_timestamp(how="end")

    # Weekly turnover V_week = sum_vol / sum_circ
    with np.errstate(divide="ignore", invalid="ignore"):
        V_week = sum_vol / sum_circ

    # Winsorize V_week to [0,1] and forward fill along time, then fill remaining NaN with 0
    V_week = V_week.clip(lower=0.0, upper=1.0).ffill().fillna(0.0)

    # Determine the target previous week timestamp (t-1)
    target_period = time.to_period("W-FRI")
    prev_period = target_period - 1
    prev_period_ts = prev_period.to_timestamp(how="end")

    week_index = P_week.index.sort_values()
    # find global prev_week_ts as the largest week index <= prev_period_ts
    eligible = week_index[week_index <= prev_period_ts]
    if len(eligible) == 0:
        # no weekly data before target -> return all NaN
        result = pd.Series(data=np.nan, index=common_cols, name="capital_gain_overhang")
        return result

    prev_week_ts = eligible.max()
    pos = int(week_index.get_indexer([prev_week_ts])[0])
    start_pos = max(0, pos - (T - 1))
    window_index = week_index[start_pos : pos + 1]

    # Slice weekly matrices for the window
    P_window = P_week.reindex(window_index)
    V_window = V_week.reindex(window_index)

    # Get P_prev (price at t-1) as a Series aligned to columns
    P_prev = P_week.loc[prev_week_ts]
    # Ensure arrays: rows = weeks in window, cols = codes
    V_arr = V_window.to_numpy(dtype=float)
    P_arr = P_window.to_numpy(dtype=float)

    # If window contains only one week, handle accordingly
    n_rows, n_cols = V_arr.shape
    if n_rows == 0:
        result = pd.Series(data=np.nan, index=common_cols, name="capital_gain_overhang")
        return result

    # compute a = 1 - V
    a = 1.0 - V_arr
    a = np.clip(a, 0.0, 1.0)

    # compute sprod for each position: sprod[i] = prod_{j=i+1..end} a[j]; last = 1
    if n_rows == 1:
        sprod = np.ones_like(a)
    else:
        a_rev = a[::-1, :]
        cumprod_rev = np.cumprod(a_rev, axis=0)
        b = cumprod_rev[::-1, :]
        sprod = np.empty_like(b)
        sprod[:-1, :] = b[1:, :]
        sprod[-1, :] = 1.0

    weights = V_arr * sprod
    # k per column
    k = np.nansum(weights, axis=0)

    # compute numerator = sum(weights * P_window)
    num = np.nansum(weights * P_arr, axis=0)

    # prepare RP vector: if k==0 -> RP = P_prev else RP = num / k
    with np.errstate(divide="ignore", invalid="ignore"):
        RP_vec = np.where(k == 0, P_prev.to_numpy(dtype=float), num / k)

    # compute CGO = (P_prev - RP) / P_prev
    P_prev_arr = P_prev.to_numpy(dtype=float)
    # conditions: if P_prev is NaN or zero -> result NaN
    invalid = np.isnan(P_prev_arr) | (P_prev_arr == 0)
    cgo_arr = (P_prev_arr - RP_vec) / P_prev_arr
    cgo_arr[invalid] = np.nan

    result = pd.Series(data=cgo_arr, index=common_cols, name="capital_gain_overhang")
    result = result.sort_index()
    return result


if __name__ == "__main__":
    print(calc_capital_gain_overhang("2025-01-02"))

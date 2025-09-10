import os
import pandas as pd
from typing import Union
from ..source import DuckParquetSource


def calc_weighted_price(time: Union[pd.Timestamp, str]) -> pd.DataFrame:
    source = DuckParquetSource(os.getenv("QUOTESMIN_PATH"), time_col="date")
    price = source.get_factor(
        name="close_post", begin=time, end=time
    )
    volume = source.get_factor(
        name="volume", begin=time, end=time
    )
    volume_weighted = (price * volume / volume.sum()).sum()
    time_weighted = price.mean()
    tail_weighted = (
        price.between_time("14:30", "15:00")
        * volume.between_time("14:30", "15:00")
        / volume.between_time("14:30", "15:00").sum()
    ).sum()
    head_weighted = (
        price.between_time("09:30", "10:00")
        * volume.between_time("09:30", "10:00")
        / volume.between_time("09:30", "10:00").sum()
    ).sum()
    return pd.concat(
        [volume_weighted, time_weighted, tail_weighted, head_weighted],
        axis=1,
        keys=["volume_weighted", "time_weighted", "tail_weighted", "head_weighted"],
    )

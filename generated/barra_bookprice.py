import os
from typing import Union
import numpy as np
import pandas as pd
from factool import DuckParquetSource


def calc_barra_bookprice(time: Union[str, pd.Timestamp]) -> pd.Series:
    quotes_source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    fin_source = DuckParquetSource(os.getenv("FINANCIALREPORT_PATH"), time_col="date")

    close = quotes_source.get_factor("close", begin=time, end=time)
    shares = quotes_source.get_factor("circulation_a", begin=time, end=time)
    st = quotes_source.get_factor("st", begin=time, end=time)
    suspended = quotes_source.get_factor("suspended", begin=time, end=time)

    mask = (~st) & (~suspended)

    mkt_cap = (close * shares).where(mask)
    mkt_cap = mkt_cap.replace(0, np.nan)

    total_equity = fin_source.get_financial("total_equity", reptype="mrq", begin=time, end=time)
    total_equity = total_equity.sort_index()
    total_equity = total_equity.reindex(total_equity.index.union(close.index)).ffill()
    total_equity_t = total_equity.loc[close.index]

    ratio_df = total_equity_t.div(mkt_cap)
    factor = ratio_df.iloc[0]
    factor = factor.where(mask.iloc[0].astype(bool))
    factor.replace([np.inf, -np.inf], np.nan, inplace=True)
    factor.name = "barra_btop"
    return factor


if __name__ == "__main__":
    print(calc_barra_bookprice("2025-01-02"))
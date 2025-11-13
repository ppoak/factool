import os
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_barra_earning(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    quotes = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="time")
    fin = DuckParquetSource(os.getenv("FINANCIALREPORT_PATH"), time_col="time")

    close = quotes.get_factor("close", begin=time, end=time).iloc[-1]
    circulation = quotes.get_factor("circulation_a", begin=time, end=time).iloc[-1]
    st = quotes.get_factor("st", begin=time, end=time).iloc[-1]
    suspended = quotes.get_factor("suspended", begin=time, end=time).iloc[-1]

    mask = (~st) & (~suspended)
    mktcap = (close * circulation).where(mask).replace(0, np.nan)

    operating_revenue = fin.get_financial(
        "operating_revenue", reptype="ttm", begin=time, end=time
    ).iloc[-1]
    net_inc_cash = fin.get_financial(
        "net_inc_cash_and_equivalents", reptype="ttm", begin=time, end=time
    ).iloc[-1]

    earningprice = (operating_revenue / mktcap).replace([np.inf, -np.inf], np.nan)
    cashflowprice = (net_inc_cash / mktcap).replace([np.inf, -np.inf], np.nan)

    df = pd.concat([earningprice, cashflowprice], axis=1)
    df.columns = ["barra_earningprice", "barra_cashflowprice"]
    return df


if __name__ == "__main__":
    import time

    begin = time.time()
    df = calc_barra_earning("2025-01-02")
    end = time.time()
    print(df)
    coverage = df.count() / df.shape[0]
    print(coverage)
    print(f"Time cost: {end - begin:.2f} s")

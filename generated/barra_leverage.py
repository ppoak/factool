import os
import numpy as np
import pandas as pd
from typing import Union
from factool import DuckParquetSource


def calc_barra_leverage(time: Union[str, pd.Timestamp]) -> pd.DataFrame:
    quotes = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
    fin = DuckParquetSource(os.getenv("FINANCIALREPORT_PATH"), time_col="date")

    close = quotes.get_factor("close_post", begin=time, end=time).squeeze()
    shares = quotes.get_factor("circulation_a", begin=time, end=time).squeeze()
    st = quotes.get_factor("st", begin=time, end=time).squeeze().fillna(False).astype(bool)
    suspended = quotes.get_factor("suspended", begin=time, end=time).squeeze().fillna(False).astype(bool)

    me = (close * shares)
    idx = me.index

    preference_shares = fin.get_financial("preference_shares", reptype="ttm", begin=time, end=time).iloc[-1].reindex(idx).fillna(0)
    non_current_liabilities = fin.get_financial("non_current_liabilities", reptype="ttm", begin=time, end=time).iloc[-1].reindex(idx)
    total_equity = fin.get_financial("total_equity", reptype="ttm", begin=time, end=time).iloc[-1].reindex(idx)
    total_assets = fin.get_financial("total_assets", reptype="ttm", begin=time, end=time).iloc[-1].reindex(idx)
    total_liabilities = fin.get_financial("total_liabilities", reptype="ttm", begin=time, end=time).iloc[-1].reindex(idx)

    mask = (~st.reindex(idx)) & (~suspended.reindex(idx))

    me_safe = me.replace(0, np.nan)
    total_equity_safe = total_equity.replace(0, np.nan)
    total_assets_safe = total_assets.replace(0, np.nan)

    market_leverage = ((me_safe + preference_shares + non_current_liabilities) / me_safe).where(mask)
    book_leverage = ((total_equity_safe + preference_shares + non_current_liabilities) / total_equity_safe).where(mask)
    debt_to_asset = (total_liabilities / total_assets_safe).where(mask)

    df = pd.concat([market_leverage, book_leverage, debt_to_asset], axis=1)
    df.columns = ["market_leverage", "book_leverage", "debt_to_asset"]
    return df


if __name__ == "__main__":
    df = calc_barra_leverage("2025-01-02")
    print(df)
    coverage = df.count() / df.shape[0]
    print(coverage)
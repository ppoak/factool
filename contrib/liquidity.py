import numpy as np
import pandas as pd
import statsmodels.api as sm
from .factor import BaseFactor


class Liquidity(BaseFactor):

    def calc_turnover_rate(self, time: str | pd.Timestamp) -> pd.Series:
        rollback = self.get_time(time, 252)[0]
        volume = self.source.get_factor("volume", begin=rollback, end=time)
        shares = self.source.get_factor("circulation_a", begin=rollback, end=time)
        month = np.log((volume.iloc[-21:] / shares.iloc[-21:]).sum().clip(lower=1e-5))
        quarter = np.log((volume.iloc[-63:] / shares.iloc[-63:]).sum().clip(lower=1e-5))
        annual = np.log((volume / shares).sum().clip(lower=1e-5))
        return pd.concat(
            [month, quarter, annual],
            axis=1,
            keys=[
                "mothly_turnover_rate",
                "quarterly_turnover_rate",
                "annual_turnover_rate",
            ],
        )

    def calc_nonliquidity_cv(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = self.source.get_time(date, 20)[0]
        price = self.source.get_factor("close_post", begin=rollback, end=date)
        amount = self.source.get_factor("amount", begin=rollback, end=date)
        ret = price.pct_change(fill_method=None).abs()
        return ret / amount

    def calc_high_low_index(self, date: str) -> pd.Series:
        rollback = self.source.get_time(date, 1)[0]
        high = self.source.get_factor("high", begin=rollback, end=date)
        low = self.source.get_factor("low", begin=rollback, end=date)
        amount = self.source.get_factor("amount", begin=rollback, end=date)
        p1 = (np.log(high / low).sum() ** 2) / 2
        p2 = np.log(high.max() / low.min()) ** 2
        p3 = (np.sqrt(2 * p1) - np.sqrt(p1)) / (3 - 2 * np.sqrt(2)) - np.sqrt(
            p2 / (3 - 2 * np.sqrt(2))
        )
        hl = 2 * (np.exp(p3) - 1) / (1 + np.exp(p3))
        return hl / amount

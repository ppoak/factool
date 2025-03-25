import numpy as np
import pandas as pd
from .factor import BaseFactor


class ReturnDistribution(BaseFactor):

    def calc_intraday_distribution(self, time: str | pd.Timestamp) -> pd.DataFrame:
        price = self.source.get_factor(
            "close", begin=time, end=time + pd.offsets.Hour(16)
        )
        ret = price.pct_change(fill_method=None)
        ret.replace([np.inf, -np.inf], np.nan, inplace=True)
        res = pd.concat(
            [ret.skew(), ret.kurt()],
            axis=1,
            keys=["intraday_return_skew", "intraday_return_kurt"],
        )
        return res

    def calc_down_trend_volatility(self, time: str | pd.Timestamp) -> pd.DataFrame:
        price = self.source.get_factor(
            "close", begin=time, end=time + pd.offsets.Hour(16)
        )
        ret = price.pct_change(fill_method=None)
        res = ret[ret < 0].pow(2).sum() / ret.pow(2).sum()
        return res

    def calc_long_short_ratio(self, time: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(time, 5)[0]
        price = self.source.get_factor(
            "close", begin=rollback, end=time + pd.offsets.Hour(16)
        )
        vol = self.source.get_factor(
            "volume", begin=rollback, end=time + pd.offsets.Hour(16)
        )
        ret = price.pct_change(fill_method=None)
        vol_per_unit = abs(vol / ret).replace([np.inf, -np.inf], np.nan)
        tot_ret = (price.iloc[-1] / price.iloc[0] - 1).abs()
        res = (tot_ret * vol_per_unit.mean()) / vol.sum()
        return res

import pandas as pd
from .factor import BaseFactor


class MomentumReverse(BaseFactor):

    def calc_naive_return_momentum(self, time: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(time, 252)[0]
        price = self.source.get_factor("close", rollback, time)
        return pd.concat(
            [
                price.iloc[-1] / price.iloc[-5] - 1,
                price.iloc[-1] / price.iloc[-21] - 1,
                price.iloc[-1] / price.iloc[-63] - 1,
                price.iloc[-1] / price.iloc[-252] - 1,
            ],
            axis=1,
            keys=[
                "naive_weekly_return",
                "naive_monthly_return",
                "naive_quarterly_return",
                "naive_yearly_return",
            ],
        )

    def calc_nonrecent_return_momentum(self, time: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(time, 252)[0]
        price = self.source.get_factor("close", rollback, time).iloc[:-5]
        return pd.concat(
            [
                price.iloc[-1] / price.iloc[-5] - 1,
                price.iloc[-1] / price.iloc[-21] - 1,
                price.iloc[-1] / price.iloc[-63] - 1,
                price.iloc[-1] / price.iloc[0] - 1,
            ],
            axis=1,
            keys=[
                "nonrecent_weekly_return",
                "nonrecent_monthly_return",
                "nonrecent_quarterly_return",
                "nonrecent_yearly_return",
            ],
        )

    def calc_decomposed_momentum(self, time: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(time, 1)[0]
        close_m = self.sources[1].get_factor(
            "close_post", time, time + pd.offsets.Hour(16)
        )
        close_d = self.source.get_factor("close_post", rollback, time)
        return_d = (close_d / close_d.shift(1) - 1).iloc[-1]
        return_m = close_m / close_m.shift(1) - 1
        return_m_mean = return_m.mean()
        return_m_std = return_m.std()
        return_m_skew = return_m.skew()
        return_m_kurt = return_m.kurt()
        return_m_abnormal = return_m.where(
            return_m - return_m_mean > 2 * return_m_std
        ).mean()
        return_m_normal = return_m.where(
            return_m - return_m_mean <= 2 * return_m_std
        ).mean()
        return pd.concat(
            [
                return_d,
                return_m_mean,
                return_m_std,
                return_m_skew,
                return_m_kurt,
                return_m_abnormal,
                return_m_normal,
            ],
            axis=1,
            keys=[
                "interday_return",
                "intraday_return",
                "intraday_return_std",
                "intraday_return_skew",
                "intraday_return_kurt",
                "intraday_return_abnormal",
                "intraday_return_normal",
            ],
        )

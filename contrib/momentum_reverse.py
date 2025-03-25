import pandas as pd
from .factor import BaseFactor


class MomentumReverse(BaseFactor):

    def calc_naive_return_momentum(self, date: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(date, 252)[0]
        price = self.source.get_factor("close", rollback, date)
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

    def calc_nonrecent_return_momentum(self, date: str | pd.Timestamp) -> pd.DataFrame:
        rollback = self.source.get_time(date, 252)[0]
        price = self.source.get_factor("close", rollback, date).iloc[:-5]
        return pd.concat(
            [
                price.iloc[-1] / price.iloc[-5] - 1,
                price.iloc[-1] / price.iloc[-21] - 1,
                price.iloc[-1] / price.iloc[-63] - 1,
                price.iloc[-1] / price.iloc[-252] - 1,
            ],
            axis=1,
            keys=[
                "nonrecent_weekly_return",
                "nonrecent_monthly_return",
                "nonrecent_quarterly_return",
                "nonrecent_yearly_return",
            ],
        )

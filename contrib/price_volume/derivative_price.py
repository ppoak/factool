import pandas as pd
from contrib.base import BaseFactor


class DerivativePrice(BaseFactor):

    def calc_weighted_price(self, time: pd.Timestamp):
        price = self.get_factor(
            name="close", begin=time, end=time + pd.offsets.Hour(n=16)
        )
        volume = self.get_factor(
            name="volume", begin=time, end=time + pd.offsets.Hour(n=16)
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

import pandas as pd
from contrib.base import XtFactorDuckDB


class DerivativePrice(XtFactorDuckDB):

    def __init__(self, qmt_path: str, duckdb_path: str, sector: str = "沪深A股"):
        super().__init__(qmt_path, duckdb_path, sector, "1m")

    def calc_volume_weighted_price(self, date: pd.Timestamp):
        price = self.get_factor(name="close", begin=date, end=date)
        volume = self.get_factor(name="volume", begin=date, end=date)
        weight = volume / volume.sum()
        res = (price * weight).sum()
        return res

    def calc_time_weighted_price(self, date: pd.Timestamp):
        price = self.get_factor(name="close", begin=date, end=date)
        res = price.mean()
        return res

    def calc_tail_weighted_price(self, date: pd.Timestamp):
        price = self.get_factor(name="close", begin=date, end=date)
        volume = self.get_factor(name="volume", begin=date, end=date)
        price = price.between_time("14:30", "15:00")
        volume = volume.between_time("14:30", "15:00")
        weight = volume / volume.sum()
        res = (price * weight).sum()
        return res

    def calc_head_weighted_price(self, date: pd.Timestamp):
        price = self.get_factor(name="close", begin=date, end=date)
        volume = self.get_factor(name="volume", begin=date, end=date)
        price = price.between_time("09:30", "10:00")
        volume = volume.between_time("09:30", "10:00")
        weight = volume / volume.sum()
        res = (price * weight).sum()
        return res

import numpy as np
import pandas as pd
from .base import FactorManager


class VolDistFactor(FactorManager):

    def get_tail_volume_percent(self, date: pd.Timestamp):
        data = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        tail_vol = data.between_time("14:31", "14:57").sum()
        day_vol = data.sum()
        res = tail_vol / day_vol
        res.name = date
        return res

    def get_volume_peak_count(self, date: pd.Timestamp):
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        mean = volume.mean()
        std = volume.std()
        peaks = volume[volume > mean + std]
        filt = peaks.notna() & peaks.shift().notna()
        peaks = peaks.where(~filt)
        res = peaks.count()
        res.name = date
        return res

    def get_foggy_amount_ratio(self, date: pd.Timestamp) -> pd.Series:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        amount = price * vol
        ret = price.pct_change(fill_method=None)
        std = ret.rolling(5).std()
        blur = std.rolling(5).std()
        foggy = blur[blur > blur.mean()]
        foggyamt = amount.where(foggy.notna())
        res = foggyamt.sum() / amount.sum()
        res.name = date
        return res

    def get_corr_volume_portion(self, date: pd.Timestamp) -> pd.Series:
        close = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        volp = vol / vol.sum()
        mean = close.rolling(10).mean()
        std = close.rolling(10).std()
        upper = mean + std
        lower = mean - std
        pos = pd.DataFrame(np.zeros(close.shape), index=close.index, columns=close.columns)
        pos[close > upper] = 1
        pos[close <= lower] = -1
        negpos_vol = volp.where(pos < 0).sum(axis=1)
        zeropos_vol = volp.where(pos == 0).sum(axis=1)
        pospos_vol = volp.where(pos > 0).sum(axis=1)
        corr_vol = pd.DataFrame(np.full(close.shape, np.nan), index=close.index, columns=close.columns)
        corr_vol = corr_vol.mask(pos > 0, pospos_vol, axis=0)
        corr_vol = corr_vol.mask(pos < 0, negpos_vol, axis=0)
        corr_vol = corr_vol.mask(pos == 0, zeropos_vol, axis=0)
        res = volp.corrwith(corr_vol)
        res.name = date
        return res
    
    def get_volume_ratio_open30(self, date: pd.Timestamp) -> pd.Series:
        df = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        morning_session = df.between_time('09:30:00', '10:00:00').sum()
        afternoon_session = df.between_time('13:00:00', '13:30:00').sum()
        res = (morning_session/afternoon_session)
        res.name = date
        return res

    def get_volume_ratio_open30_20d(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        res = self.read('volume_ratio_open30',start=rollback, stop=date).tail(20)
        res = res.sort_index().ewm(alpha = 2/(20+1), adjust=False).mean().sum()/20
        res.name = date
        return res
    
    def get_volume_ratio_open5(self, date: pd.Timestamp) -> pd.Series:
        df = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        morning_session = df.between_time('09:30:00', '09:35:00').sum()
        afternoon_session = df.between_time('13:00:00', '13:05:00').sum()
        total = df.sum()
        res = (morning_session + afternoon_session)/total
        res.name = date
        return res
    
    def get_volume_ratio_top20_price(self, date: pd.Timestamp) -> pd.Series:
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        total = volume.sum()
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        price = price[price > price.quantile(0.8)]
        mask = price.notna()
        volume = volume[mask]
        res = volume.sum()/total 
        res.name = date
        return res

    def get_volume_ratio_bottom20_price(self, date: pd.Timestamp) -> pd.Series:
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        total = volume.sum()
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        price = price[price < price.quantile(0.2)]
        mask = price.notna()
        volume = volume[mask]
        res = volume.sum()/total 
        res.name = date
        return res

    def get_std_weighted_volume(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        volume = quotes_day.read("volume", start=rollback, stop=date)
        std = self.read('daily_std', start=rollback, stop=date)
        res = (volume * std.div(std.sum(axis=0),axis=1)).sum()
        res = res/res.sum()
        res.name = date
        return res
    
    def get_std_price_corr(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        std = self.read('daily_std', start=rollback, stop=date)
        price = self.read('volume_weighted_price', start=rollback, stop=date)
        res = std.corrwith(price)
        res.name = date
        return res
    
    def get_std_top20_ret(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 20)
        std = self.read('daily_std', start=rollback, stop=date)
        std = std[std>std.quantile(0.8)]
        mask = std.notna()
        ret = quotes_day.read("close", start=rollback, stop=date).pct_change(fill_method=None)
        res = ret[mask].sum()
        res.name = date
        return res
import numpy as np
import pandas as pd
from .base import (
    quotes_day, quotes_min, wscore, BaseFactor
)


class DeraPriceFactor(BaseFactor):

    def get_volume_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res
    
    def get_time_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        res = p.mean()
        res.name = date
        return res
    
    def get_tail_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        p = p.between_time("14:30", "15:00")
        vol = vol.between_time("14:30", "15:00")
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res

    def get_head_weighted_price(self, date: pd.Timestamp):
        p = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        p = p.between_time("9:30", "10:00")
        vol = vol.between_time("9:30", "10:00")
        w = vol / vol.sum()
        res = (p * w).sum()
        res.name = date
        return res


class PriceVolumeCorr(BaseFactor):

    def get_smart_money_ratio(self, date: pd.Timestamp) -> pd.DataFrame:
        rollback = self.get_trading_days_rollback(date, 9)
        price = quotes_min.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None).abs()
        vol = quotes_min.read("volume", start=rollback, stop=date + pd.Timedelta(days=1))
        retvol = ret / (vol ** 0.25)
        rank = retvol.rank(axis=0, ascending=False)
        rank = rank.le(retvol.count() // 5, axis=1)
        retvol = vol.where(rank)
        res = ((retvol * price).sum() / retvol.sum()) / ((vol * price).sum() / vol.sum())
        res.name = date
        return res

    def get_price_volume_corr(self, date: pd.Timestamp) -> pd.DataFrame:
        price = quotes_min.read("close", start=date, stop=date + pd.Timedelta(days=1))
        volume = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        res = price.corrwith(volume, axis=0).replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def get_average_relative_price_percent(self, date: pd.Timestamp) -> pd.DataFrame:
        df = quotes_min.read("open, high, low, close", start=date, stop=date + pd.Timedelta(days=1))
        twap = df.mean(axis=1).groupby(level=quotes_min._code_level).mean()
        high = df["high"].groupby(level=quotes_min._code_level).max()
        low = df["low"].groupby(level=quotes_min._code_level).min()
        arrp = (twap - low) / (high - low)
        arrp.name = date
        return arrp
    
    def get_volume_ratio_open30(self, date: pd.Timestamp) -> pd.Series:
        df = quotes_min.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        morning_session = df.between_time('09:30:00', '10:00:00').sum()
        afternoon_session = df.between_time('13:00:00', '13:30:00').sum()
        res = (morning_session/afternoon_session)
        res.name = date
        return res

    def get_volume_ratio_open30_20d(self, date: pd.Timestamp) -> pd.Series:
        res = pd.DataFrame()
        for i in range(0, 20, 1): 
            if res.empty:
                res = self.get_volume_ratio_open30(quotes_day.get_trading_days_rollback(date, i)).to_frame().T
            else:
                res = pd.concat([res, self.get_volume_ratio_open30(quotes_day.get_trading_days_rollback(date, i)).to_frame().T])
        res = res.ewm(alpha = 2/21, adjust=False).mean().sum()/20
        res.name = date
        return res
    
    def get_compound_price_volume_corr(self, date: pd.Timestamp) -> pd.Series:
        rollback = quotes_day.get_trading_days_rollback(date, 21)
        dp = quotes_day.read("close", start=rollback, stop=date).diff(1).iloc[1:].tail(20)
        dv = quotes_day.read("volume", start=rollback, stop=date).diff(1).iloc[1:].head(20)
        dv.index = dp.index
        # 原始
        dV_dP_Corr = dp.corrwith(dv, axis=0)
        dV_dP_Corr.name = date
        dV_dP_Corr = wscore(dV_dP_Corr.to_frame().T, date).loc[date]
        
        dp_p = dp[dp > 0]
        dv_p = dv[dv > 0]
        dV_dP_Corr_pp = dp_p.corrwith(dv_p, axis=0)
        dV_dP_Corr_pp.name = date
        dV_dP_Corr_pp = wscore(dV_dP_Corr_pp.to_frame().T, date).loc[date]

        dp_n = dp[dp < 0]
        dv_n = dv[dv < 0]
        dV_dP_Corr_nn = dp_n.corrwith(dv_n, axis=0)
        dV_dP_Corr_nn.name = date
        dV_dP_Corr_nn = wscore(dV_dP_Corr_nn.to_frame().T, date).loc[date]

        dV_dP_Corr_np = dp_p.corrwith(dv_n, axis=0)
        dV_dP_Corr_np.name = date
        dV_dP_Corr_np = wscore(dV_dP_Corr_np.to_frame().T, date).loc[date]

        dV_dP_Corr_pn = dp_n.corrwith(dv_p, axis=0)
        dV_dP_Corr_pn.name = date
        dV_dP_Corr_pn = wscore(dV_dP_Corr_pn.to_frame().T, date).loc[date]        

        # 复合量先价行
        res = dV_dP_Corr_pp - dV_dP_Corr_pn - dV_dP_Corr_np + dV_dP_Corr_nn
        return res
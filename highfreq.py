import quool
import numpy as np
import pandas as pd
import factor as ft
from tqdm import tqdm
from joblib import Parallel, delayed


QTM_URI = '/home/data/quotes-min'
QTD_URI = '/home/data/quotes-day'

MIN = 240
WEEK = 5
MONTH = 21
YEAR = 252
DATE_LEVEL = 'datetime'
CODE_LEVEL = 'orderbook_id'


def get_tail_volume_percent(start: str, stop: str) -> pd.DataFrame:
    def _get(date: pd.Timestamp):
        data = ft.get_data(QTM_URI, "volume", start=date, stop=date + pd.Timedelta(days=1))
        tail_vol = data.between_time("14:30", "14:57").sum()
        day_vol = data.sum()
        res = tail_vol / day_vol
        res.name = date
        return res
        
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(
        delayed(_get)(date) for date in tqdm(list(trading_days))
    ), axis=1).T.loc[start:stop]

def get_intraday_distribution(start: str, stop: str) -> pd.DataFrame:
    def _get(date: pd.Timestamp):
        data = ft.get_data(QTM_URI, "close", start=date, stop=date + pd.Timedelta(days=1))
        ret = data.pct_change(fill_method=None)
        res = pd.concat([ret.skew(), ret.kurt()], axis=1, 
            keys=['intraday_return_skew', 'intraday_return_kurt'])
        res.index = pd.MultiIndex.from_product([
             res.index, [date]], names=["order_book_id", "date"])
        return res
    
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(
        delayed(_get)(date) for date in tqdm(list(trading_days))
    ), axis=0).sort_index().loc(axis=0)[:, start:stop]

def get_long_short_ratio(start: str, stop: str):
    def _get(_start, _stop):
        _price = ft.get_data(QTM_URI, "close", start=_start, stop=_stop + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_start, stop=_stop + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        _vol_per_unit = abs(_vol / _return).replace([np.inf, -np.inf], np.nan)
        _1 = _vol_per_unit.mean()
        _tot_return = (_price.iloc[-1] / _price.iloc[0] - 1).abs()
        _res = (_tot_return * _1) / _vol.sum()
        _res.name = _stop
        return _res
        
    rollback = ft.get_trading_days_rollback(QTD_URI, start, WEEK)
    trading_days = ft.get_trading_days(QTD_URI, rollback, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (_start, _stop) for _start, _stop in  tqdm(list(
            zip(trading_days[:-WEEK + 1], trading_days[WEEK - 1:])
        ))), axis=1).T.loc[start:stop]

def get_price_volume_corr(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _volume = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + pd.Timedelta(days=1))
        _res = _price.corrwith(_volume, axis=0).replace([np.inf, -np.inf], np.nan)
        _res.name = _date
        return _res

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=1).T.loc[start:stop]

def get_down_trend_volatility(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        res = _return.apply(lambda x: x[x < 0].pow(2).sum() / x.pow(2).sum())
        res.name = _date
        return res
    
    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=1).T.loc[start:stop]

def get_regret(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + pd.Timedelta(days=1))
        lc = _price <= _price.iloc[-1]
        hc = _price > _price.iloc[-1]
        lcvol = _vol.where(lc).sum() / _vol.sum()
        hcvol = _vol.where(hc).sum() / _vol.sum()
        lcp = (_price.where(lc) * _vol.where(lc)).sum() / _vol.where(lc).sum() / _price.iloc[-1] - 1
        hcp = (_price.where(hc) * _vol.where(hc)).sum() / _vol.where(hc).sum() / _price.iloc[-1] - 1
        res = pd.concat([lcvol, hcvol, lcp, hcp], axis=1, keys=["lcvol", "hcvol", "lcp", "hcp"])
        res.index = pd.MultiIndex.from_product([res.index, [_date]], names=["order_book_id", "date"])
        return res

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=1, backend='loky')(delayed(_get)
        (date) for date in  tqdm(list(trading_days))), axis=0
    ).sort_index().loc(axis=0)[:, start:stop]

def get_updown_volume(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        _price = ft.get_data(QTM_URI, "close", start=_date, stop=_date + pd.Timedelta(days=1))
        _vol = ft.get_data(QTM_URI, "volume", start=_date, stop=_date + pd.Timedelta(days=1))
        _return = _price.pct_change(fill_method=None)
        upvol = _vol[_return > 0].sum() / _vol.sum()
        downvol = _vol[_return < 0].sum() / _vol.sum()
        res = pd.concat([upvol, downvol], axis=1, keys=["upvol", "downvol"])
        res.index = pd.MultiIndex.from_product([res.index, [_date]], names=["order_book_id", "date"])
        return res

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=1, backend='loky')(delayed(_get)
        (date) for date in tqdm(list(trading_days))), axis=0
    ).sort_index().loc(axis=0)[:, start:stop]

def get_average_relative_price_percent(start: str, stop: str) -> pd.DataFrame:
    def _get(_date):
        qtm = quool.PanelTable(QTM_URI)
        df = qtm.read("open, high, low, close", start=_date, stop=_date + pd.Timedelta(days=1))
        twap = df.mean(axis=1).groupby(level='order_book_id').mean()
        high = df["high"].groupby(level='order_book_id').max()
        low = df["low"].groupby(level='order_book_id').min()
        arrp = (twap - low) / (high - low)
        arrp.name = _date
        return arrp

    trading_days = ft.get_trading_days(QTD_URI, start, stop)
    return pd.concat(Parallel(n_jobs=1, backend='loky')(delayed(_get)
        (date) for date in tqdm(list(trading_days))), axis=1
    ).sort_index().T.loc[:, start:stop]

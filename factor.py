__version__ = "0.3.1"


import quool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def get_data(
    datauri: str,
    field: str | list,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    pool: str = None,
    pooluri: str = None,
    dropna: bool = False,
    code_level: str | int = 0,
    date_level: str | int = 1,
):
    field = quool.parse_commastr(field)
    data_table = quool.PanelTable(datauri, code_level=code_level, date_level=date_level)
    if pooluri:
        pool_table = quool.PanelTable(pooluri, code_level=code_level, date_level=date_level)

    code = None
    if pool and pooluri:
        pool_index = pool_table.read(pool, start=start, stop=stop).dropna().index
        code = pool_index.get_level_values(code_level)
    elif pool and not pooluri:
        code = pool
    
    data = data_table.read(field, code=code, start=start, stop=stop)
    if len(field) > 1:
        return data
    
    data = data[field[0]]
    if dropna:
        data = data.dropna()
    if pool and pooluri:
        data = data.loc[data.index.isin(pool_index)]
    data = data.unstack(level=code_level)
    return data

def get_trading_days(
    uri: str, 
    start: str, 
    stop: str, 
    field: str = 'close',
) -> pd.DatetimeIndex:
    table = quool.PanelTable(uri)
    field = table.read('close', code='000001.XSHE', start=start, stop=stop)
    trading_days = field.droplevel(0).index
    return trading_days

def get_trading_days_rollback(uri: str, date: str, shift: int) -> pd.DatetimeIndex:
    if shift > 0:
        trading_days = get_trading_days(uri, start=None, stop=date)
        rollback = trading_days[trading_days <= date][-shift - 1]
    else:
        trading_days = get_trading_days(uri, start=date, stop=None)
        rollback = trading_days[min(len(trading_days), -shift)]
    return rollback

def get_price(
    uri: str,
    ptype: str | list,
    pool: str = None,
    pooluri: str = None,
    start: str | pd.Timestamp = None,
    stop: str | pd.Timestamp = None,
    filter: bool = True,
    adjust: bool = True,
    code_level: int | str = 0,
    date_level: int | str = 1,
) -> pd.DataFrame:
    ptype = quool.parse_commastr(ptype)
    if filter:
        names = ptype + ["st", "suspended"]
    if adjust:
        names = names + ["adjfactor"]
    data = get_data(datauri=uri, field=names, 
        pool=pool, pooluri=pooluri,
        start=start, stop=stop, dropna=False,
        code_level=code_level, date_level=date_level,
    )
    if filter:
        stsus = (data['st'] | data['suspended']).replace(np.nan, True)
        data.loc[:, ptype] = data.loc[:, ptype].where(~stsus)
    if adjust:
        data.loc[:, ptype] = data.loc[:, ptype].mul(data['adjfactor'], axis=0)
    if len(ptype) > 1:
        return data[ptype]
    return data[ptype[0]].unstack(level=code_level)

def save_data(
    data: pd.DataFrame | pd.Series, 
    uri: str,
    name: str = None, 
    code_level: str = 'order_book_id',
    date_level: str = 'date',
):
    table = quool.PanelTable(uri, 
        code_level=code_level, date_level=date_level)
    if isinstance(data, pd.DataFrame) and data.index.nlevels == 1:
        data = data.stack(dropna=True).swaplevel()
        data.index.names = [code_level, date_level]
    if isinstance(data, pd.Series):
        if name is None:
            raise ValueError('name cannot be None')
        data = data.to_frame(name)
    update_data = data[data.columns[data.columns.isin(table.columns)]]
    add_data = data[data.columns[~data.columns.isin(table.columns)]]
    if not update_data.empty:
        table.update(data)
    if not add_data.empty:
        table.add(data)

def zscore(df: pd.DataFrame):
    return df.sub(df.mean(axis=1), axis=0
        ).div(df.std(axis=1), axis=0)

def minmax(df: pd.DataFrame):
    return df.sub(df.min(axis=1), axis=0).div(
        df.max(axis=1) - df.min(axis=1), axis=0)

def madoutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    median = df.median(axis=1)
    ad = df.sub(median, axis=0)
    mad = ad.abs().median(axis=1)
    thresh_down = median - dev * mad
    thresh_up = median + dev * mad
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def stdoutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    thresh_down = mean - dev * std
    thresh_up = mean + dev * std
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def iqroutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    thresh_up = df.quantile(1 - dev / 2, axis=1)
    thresh_down = df.quantile(dev / 2, axis=1)
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def replace(
    df: pd.DataFrame, 
    old: int | float, 
    new: int | float
):
    return df.replace(old, new)

def log(df: pd.DataFrame, base: int = 10):
    return np.log(df) / np.log(base)

def perform_crosssection(
    factor: pd.DataFrame,
    price: pd.DataFrame = None,
    delay: int = 1,
    crossdate: str | int = -1,
    rank: bool = False,
    image: str | bool = True,
    result: str = None,
):
    price = price.shift(-delay).loc[factor.index]
    if isinstance(crossdate, int):
        crossdate = factor.index[crossdate].strftime(r"%Y-%m-%d")
    factor = factor.loc[crossdate]
    if price is not None:
        future_returns = price.shift(-1) / price - 1
        future_returns = future_returns.loc[crossdate]
    else:
        future_returns = pd.Series(index=factor.index)
    if rank:
        factor = factor.rank()
    data = pd.concat([factor, future_returns], 
        axis=1, keys=["factor", "future_returns"])
    if image is not None:
        fig, axes = plt.subplots(2, 1, figsize=(20, 20))
        data["factor"].plot.hist(bins=100, ax=axes[0])
        data.plot.scatter(ax=axes[1], title=crossdate,
            x="factor", y="future_returns")
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()
    if result is not None:
        data.to_excel(result)
    return data

def perform_inforcoef(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    delay: int = 1,
    method: str = 'pearson',
    image: str | bool = True,
    result: str = None,
):
    price = price.shift(-delay).loc[factor.index]
    future_returns = price.shift(-1) / price - 1
    inforcoef = factor.corrwith(future_returns, axis=1, method=method).dropna()
    inforcoef.name = f"infocoef"
    if image is not None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax = inforcoef.plot.bar(ax=ax, title=inforcoef.name)
        ax.set_xticks([i for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)], 
            [inforcoef.index[i].strftime(r'%Y-%m-%d') for i in range(0, inforcoef.shape[0], inforcoef.shape[0] // 10)])
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()
    if result is not None:
        inforcoef.to_excel(result)
    return inforcoef

def perform_backtest(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    delay: int = 1,
    topk: int = 100,
    benchmark: pd.Series = None,
    ngroup: int = 5,
    commission: float = 0.002,
    n_jobs: int = -1,
    image: str | bool = True,
    result: str = None,
):
    # ngroup test
    try:
        groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
    except:
        for date in factor.index:
            try:
                pd.qcut(factor.loc[date], q=ngroup, labels=False)
            except:
                raise ValueError(f"on date {date}, grouping failed")
    
    ngroup_result = Parallel(n_jobs=-1, backend='loky')(
        delayed(quool.weight_strategy)(
            (groups.where(groups == i) / groups.where(groups == i)).div(
                groups.where(groups == i).count(axis=1), axis=0).fillna(0), 
            price, delay, 'both', commission, benchmark, False, None
    ) for i in range(1, ngroup + 1))
    ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
    ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
    ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
        axis=1, keys=range(1, ngroup + 1)).add_prefix('group')

    # topk test
    topks = factor.rank(ascending=False, axis=1) < topk
    topks = factor.where(topks)
    topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
    topk_result = quool.weight_strategy(topks, price, delay, 'both', 
        commission, benchmark, None, None)
    topk_evaluation = topk_result['evaluation']
    topk_value = topk_result['value']
    topk_turnover = topk_result['turnover']

    # compute returns
    ngroup_returns = ngroup_value.pct_change().fillna(0)
    topk_returns = topk_value.pct_change().fillna(0)
    
    # longshort test
    longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
    longshort_value = (longshort_returns + 1).cumprod()
    longshort_value.name = "long-short"
    
    # merge returns
    if benchmark is not None:
        benchmark = benchmark.squeeze()
        benchmark = benchmark.pct_change().fillna(0)
        ngroup_exreturns = ngroup_returns.sub(benchmark, axis=0)
        ngroup_returns = pd.concat([ngroup_returns, benchmark], axis=1)
        topk_exreturns = topk_returns.sub(benchmark, axis=0)
        topk_returns = pd.concat([topk_returns, benchmark], axis=1)
    
    # compute exvalue
    if benchmark is not None:
        ngroup_exvalue = (ngroup_exreturns + 1).cumprod()
        topk_exvalue = (topk_exreturns + 1).cumprod()
    
    # naming
    ngroup_evaluation.name = "ngroup evaluation"
    ngroup_value.name = "ngroup value"
    topk_value.name = "topk value"
    longshort_value.name = "longshort value"
    ngroup_turnover.name = "ngroup turnover"
    topk_turnover.name = "topk turnover"
    if benchmark is not None:
        ngroup_exvalue.name = "ngroup exvalue"
        topk_exvalue.name = "topk exvalue"

    if image is not None:
        fignum = 5 + 2 * (benchmark is not None)
        fig, axes = plt.subplots(nrows=fignum, ncols=1, figsize=(20, 10 * fignum))
        ngroup_value.plot(ax=axes[0], title=ngroup_value.name)
        ngroup_turnover.plot(ax=axes[1], title=ngroup_turnover.name)
        topk_value.plot(ax=axes[2], title=topk_value.name)
        topk_turnover.plot(ax=axes[3], title=topk_turnover.name)
        longshort_value.plot(ax=axes[4], title=longshort_value.name)
        if benchmark is not None:
            ngroup_exvalue.plot(ax=axes[5], title=ngroup_exvalue.name)
            topk_exvalue.plot(ax=axes[6], title=topk_exvalue.name)
        fig.tight_layout()
        if not isinstance(image, bool):
            fig.savefig(image)
        else:
            fig.show()

    if result is not None:
        with pd.ExcelWriter(result) as writer:
            ngroup_evaluation.to_excel(writer, sheet_name=ngroup_evaluation.name)
            topk_evaluation.to_excel(writer, sheet_name=topk_evaluation.name)
            ngroup_value.to_excel(writer, sheet_name=ngroup_returns.name)
            ngroup_turnover.to_excel(writer, sheet_name=ngroup_turnover.name)
            topk_value.to_excel(writer, sheet_name=topk_returns.name)
            topk_turnover.to_excel(writer, sheet_name=topk_turnover.name)
            longshort_value.to_excel(writer, sheet_name=longshort_returns.name)
            
            if benchmark is not None:
                ngroup_exvalue.to_excel(writer, sheet_name=ngroup_exreturns.name)
                topk_exvalue.to_excel(writer, sheet_name=topk_exreturns.name)

    return {
        'ngroup_evaluation': ngroup_evaluation, 
        'ngroup_value': ngroup_value, 
        'ngroup_turnover': ngroup_turnover,
        'topk_evaluation': topk_evaluation, 
        'topk_value': topk_value, 
        'topk_turnover': topk_turnover,
        'longshort_value': longshort_value,
    }

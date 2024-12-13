import factorlab
import numpy as np
import pandas as pd
import dataforge as forge
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec


def generate_plot(
    crosssection: pd.DataFrame, 
    inforcoef: pd.DataFrame, 
    val: pd.DataFrame, 
    output: Path
):
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig)

    # Plot factor histogram
    ax_hist_factor = fig.add_subplot(gs[0, 0])
    ax_hist_factor.hist(crosssection["Returns"].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black', orientation='horizontal')
    ax_hist_factor.set_title("Returns Histogram")
    ax_hist_factor.set_xlabel("Returns Value")
    ax_hist_factor.set_ylabel("Frequency")

    # Plot factor vs returns scatter plot
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_scatter.scatter(crosssection["Factors"], crosssection["Returns"], alpha=0.6, color='green')
    ax_scatter.set_title("Factor vs Returns")
    ax_scatter.set_xlabel("Factor Value")
    ax_scatter.set_ylabel("Returns")

    # Empty subplot (for table)
    ax_blank = fig.add_subplot(gs[1, 0])
    ax_blank.axis('off')

    # Sort crosssection by factor descending, take top 10 and bottom 10
    cs_sorted = crosssection.dropna(axis=0).reset_index().sort_values("Factors", ascending=False)
    top10 = cs_sorted.head(10)
    bottom10 = cs_sorted.tail(10)
    
    # Convert to rounded list for display
    top10_list = top10[["code", 'Factors', 'Returns']].round(4).values.tolist()
    bottom10_list = bottom10[["code", 'Factors', 'Returns']].round(4).values.tolist()
    
    # Insert a separator row
    display_data = top10_list + [['...', '...', '...']] + bottom10_list

    # Create table
    table = ax_blank.table(
        cellText=display_data,
        colLabels=['Codes', 'Factros', 'Returns'],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Plot returns histogram
    ax_hist_returns = fig.add_subplot(gs[1, 1])
    ax_hist_returns.hist(crosssection["Factors"].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax_hist_returns.set_title("Factors Histogram")
    ax_hist_returns.set_xlabel("Factors")
    ax_hist_returns.set_ylabel("Frequency")

    # Plot inforcoef with rolling mean and cumulative
    ax_inforcoef = fig.add_subplot(gs[0:2, 2:])
    ax_inforcoef.plot(inforcoef["inforcoef"], label="Inforcoef")
    ax_inforcoef.plot(inforcoef["rolling_mean"], label="5D Rolling Mean", linestyle="--")
    ax_cumulative = ax_inforcoef.twinx()
    ax_cumulative.plot(inforcoef["cumulative"], label="Cumulative Inforcoef", color="orange", linestyle=":")
    ax_inforcoef.set_title("Inforcoef with Rolling Mean & Cumulative")
    ax_inforcoef.legend(loc="upper left")
    ax_cumulative.legend(loc="upper right")

    # Plot net value for groups, TopK, and LongShort
    ax_combined = fig.add_subplot(gs[2, :])
    val.plot(ax=ax_combined, label=val.columns)
    ax_combined.set_title("Net Value for Groups, TopK, and LongShort")
    ax_combined.legend(loc="best")
    ax_combined.set_xlabel("Date")
    ax_combined.set_ylabel("Net Value")

    plt.tight_layout()
    fig.savefig(out_path)


def backtest_factor(
    factor_data: pd.DataFrame,
    return_data: pd.DataFrame,
    output: str,
    period: int = 5,
    ngroup: int = 5,
    topk: int = 5,
    commission: float = 0.0001,
):
    date = return_data.dropna(how='all', axis=0).index.intersection(factor_data.dropna(how='all', axis=0).index)[-1]
    return_data = return_data / period
    
    crosssection = pd.concat([factor_data.loc[date], return_data.loc[date]], axis=1, keys=["Factors", "Returns"])
    inforcoef = factorlab.corr(factor_data, return_data, axis=1)
    direction = np.sign(inforcoef.mean())
    inforcoef = pd.DataFrame({
        "inforcoef": inforcoef,
        "rolling_mean": inforcoef.rolling(window=5).mean(),
        "cumulative": inforcoef.cumsum()
    })

    factor_data = factor_data * direction

    group = factorlab.group(factor_data, n=ngroup, axis=0)
    groupeva = []
    groupval = []
    groupret = []
    for i in range(1, ngroup + 1):
        groupi = group == i
        delta = factorlab.diff(factorlab.weightify(groupi.astype("int")))
        turnover = factorlab.sum(factorlab.absolute(delta), axis=1) / 2
        commission = turnover * commission / period
        ret = factorlab.fillna(factorlab.shift((factorlab.mean(factorlab.where(return_data, groupi, np.nan), axis=1)), 1), 0)
        val = factorlab.cumprod(ret - commission + 1)
        eva = forge.Evaluator._evaluate(val)
        groupret.append(ret)
        groupeva.append(eva)
        groupval.append(val)
    
    longshortret = groupret[-1] - groupret[0]
    longshortval = factorlab.cumprod(longshortret + 1)
    longshorteva = forge.Evaluator._evaluate(longshortval)

    groupval = pd.concat(groupval + [longshortval], axis=1, keys=[f"Group{i}" for i in range(1, ngroup + 1)] + ["LongShort"])
    groupeva = pd.concat(groupeva + [longshorteva], axis=1, keys=[f"Group{i}" for i in range(1, ngroup + 1)] + ["LongShort"])

    rank = factorlab.rank(factor_data, ascending=False, axis=1)
    toprank = rank < (topk + 1)
    delta = factorlab.diff(factorlab.weightify(toprank.astype("int")))
    turnover = factorlab.sum(factorlab.absolute(delta), axis=1) / 2
    commission = turnover * commission / period
    topkret = factorlab.fillna(factorlab.shift(factorlab.mean(factorlab.where(return_data, toprank, np.nan), axis=1), n=1), val=0)
    topkval = factorlab.cumprod(topkret - commission + 1)
    topkval.name = "Topk"
    topkeva = forge.Evaluator._evaluate(topkval)
    topkeva.name = "Topk"

    val = pd.concat([groupval, topkval], axis=1)
    eva = pd.concat([groupeva, topkeva], axis=1)
    
    generate_plot(crosssection, inforcoef, val, output)
    return eva

if __name__ == "__main__":
    # User-configurable parameters for the report generation

    factor_name = "compound_volume_first"  # The factor you want to analyze
    factor_path = "D:/Documents/DataBase/factor_lab"  # Path to the data folder (default is "data/price_volume")
    ptype = "open"
    pool_code = "000985.XSHG"

    # Performance settings
    period = 5  # Period for the performance calculation (e.g., 5 for 5 days in a period)
    ngroup = 5  # Number of groups for factor sorting
    topk = 100 # Number of top stocks to consider

    # Date range for the analysis
    begin = "2015-01-01"  # Start date
    end = "2024-12-02"  # End date
    commission = 0.0000  # Commission rate for trading
    out_path = f"out/report_{factor_name}_{ptype}_{pool_code}_{period}_{topk}.png"  # Output file path for the report

    # Generate the report with the specified parameters
    factor_data = factorlab.Factor(factor_path).read(factor_name, begin=begin, end=end)
    factor_data = factorlab.zscore(factorlab.stdoutlier(factor_data, 3))
    
    price_data = factorlab.quotes_day.read(ptype, begin=begin, end=end)
    adjfactor = factorlab.quotes_day.read(name="adjfactor", begin=begin, end=end)
    price_data = factorlab.mul(price_data, adjfactor)
    
    if pool_code is not None:
        weight = factorlab.index_weights.read(pool_code, begin=begin, end=end)
    else:
        weight = pd.DataFrame(np.ones_like(factor_data), index=factor_data.index, columns=factor_data.columns).astype("bool")

    st = factorlab.quotes_day.read("st", begin=begin, end=end).astype("bool")
    suspended = factorlab.quotes_day.read("suspended", begin=begin, end=end).astype("bool")
    limit_up = factorlab.quotes_day.read("limit_up", begin=begin, end=end)
    limit_down = factorlab.quotes_day.read("limit_down", begin=begin, end=end)
    high = factorlab.quotes_day.read("high", begin=begin, end=end)
    low = factorlab.quotes_day.read("low", begin=begin, end=end)
    feasible = ~st & ~suspended & (high < limit_up) & (low > limit_down)
    
    return_data = factorlab.shift(price_data, -1 - period) / factorlab.shift(price_data, -1) - 1
    return_data = factorlab.where(return_data, feasible, np.nan)
    return_data = factorlab.where(return_data, weight > 0, np.nan)
    factor_data = factorlab.where(factor_data, weight > 0, np.nan)
    
    evaluation = backtest_factor(
        factor_data=factor_data,
        return_data=return_data,
        output=out_path,
        period=period,
        ngroup=ngroup,
        topk=topk,
        commission=commission,
    )
    print(evaluation)

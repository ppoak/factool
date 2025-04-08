# %% [markdown]
# 因子测试
import factorlab
import numpy as np
import pandas as pd
from quool import ParquetManager, setup_logger


# %% [markdown]
# 参数设定
factor_name = "naive_weekly_return_processed"  # The factor you want to analyze
factor_path = "data/naive_return_momentum"  # Path to the data folder (default is "data/price_volume")
price_path = "D:/Documents/DataBase/quotes_day"  # Path to the data folder (default is "data/price_volume")
benchmark_path = "D:/Documents/DataBase/index_quotes_day"
log_path = (
    f"out/{factor_name}.log"  # Path to the log file (default is "out/factor_name.log")
)
begin = "2015-01-01"  # Start date
end = "now"  # End date
ptype = (
    "open"  # Price type to use for backtesting (e.g., "open", "close", "high", "low")
)
benchmark_code = "000985.XSHG"
time_col = "time"  # Name of the time column in the data
code_col = "code"  # Name of the code column in the data
freq = 5  # Frequency of rebalance (e.g., 1 for daily, 5 for weekly)
weight = None  # Portfolio weights (default is None)
topk = 100  # Number of top stocks to select (default is 100)
ic_method = "spearman"  # Method to use for information coefficient calculation (default is "spearman")
ngroup = 5  # Number of groups to split the stocks into (default is 10)
commission = 0.0000  # Commission rate for trading
out_path = f"out/report_{factor_name}_{ptype}_{benchmark_code}_{freq}_{topk}.png"  # Output file path for the report
logger = setup_logger("factor_test", file=log_path, level="INFO")

# %% [markdown]
# 价格数据读取
price_source = ParquetManager(price_path)
price_data = {}
for field in [ptype, "st", "suspended", "limit_up", "limit_down", "high", "low"]:
    price_data[field] = price_source.read(
        index=time_col,
        columns=code_col,
        pivot=field,
        **{f"{time_col}__ge": begin, f"{time_col}__le": end},
    )
feasible = (
    ~(price_data["st"].replace(np.nan, False))
    & ~(price_data["suspended"].replace(np.nan, False))
    & (price_data["high"] < price_data["limit_up"])
    & (price_data["low"] > price_data["limit_down"])
)
if benchmark_code is not None:
    benchmark = (
        ParquetManager(benchmark_path)
        .read(
            index=time_col,
            columns="close",
            **{
                f"{code_col}": benchmark_code,
                f"{time_col}__ge": begin,
                f"{time_col}__le": end,
            },
        )
        .squeeze()
    )
else:
    benchmark = None

# %% [markdown]
# 因子数据读取
factor_source = factorlab.ParquetFactorSource(
    factor_path, time_col=time_col, code_col=code_col
)
factor_data = factor_source.get_factor(factor_name, begin=begin, end=end)

# %% [markdown]
# 因子回测
evaluator = factorlab.Evaluator(
    factor=factor_data,
    price=price_data[ptype],
    logger=logger,
)
evaluator(
    method=ic_method,
    n=ngroup,
    k=topk,
    freq=freq,
    weight=weight,
    feasible=feasible,
    benchmark=benchmark,
    commission=commission,
)

# %% [markdown]
# 因子报告
# 因子报告提供一表三图，有topk+ngroup、ic的测试结果
# 分层测试可能失败，关注CRITICAL信息
pd.concat([evaluator.ic, evaluator.ic.cumsum()], axis=1, keys=["raw", "cumsum"]).plot(
    secondary_y="cumsum"
)
pd.concat([evaluator.value_topk.to_frame("topk"), evaluator.value_ngroup], axis=1).plot(
    secondary_y="topk"
)
pd.concat(
    [evaluator.evaluation_topk.to_frame("topk"), evaluator.evaluation_ngroup], axis=1
)

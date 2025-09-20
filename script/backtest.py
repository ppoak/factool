import factool
import numpy as np
import pandas as pd
from parquool import DuckParquet, setup_logger


factor_name = "log_market_size"  # The factor you want to analyze
factor_path = "D:/Documents/DataSet/factor_lab/market_size"  # Path to the data folder (default is "data/price_volume")
price_path = "D:/Documents/DataSet/quotes_day"  # Path to the data folder (default is "data/price_volume")
benchmark_path = "D:/Documents/DataSet/index_quotes_day"
log_path = (
    f"out/{factor_name}.log"  # Path to the log file (default is "out/factor_name.log")
)
begin = "2015-01-01"  # Start date
end = "now"  # End date
ptype = "open_post"  # Price type to use for backtesting (e.g., "open", "close", "high", "low")
benchmark_code = "000985.SH"
time_col = "date"  # Name of the time column in the data
code_col = "code"  # Name of the code column in the data
freq = 5  # Frequency of rebalance (e.g., 1 for daily, 5 for weekly)
weight = None  # Portfolio weights (default is None)
topk = 100  # Number of top stocks to select (default is 100)
ic_method = "spearman"  # Method to use for information coefficient calculation (default is "spearman")
ngroup = 10  # Number of groups to split the stocks into (default is 10)
commission = 0.0005  # Commission rate for trading
out_path = f"out/report_{factor_name}_{ptype}_{benchmark_code}_{freq}_{topk}.png"  # Output file path for the report
logger = setup_logger("factor_test", file=log_path, level="INFO")

price_source = factool.DuckParquetSource(
    price_path, time_col=time_col, code_col=code_col
)
price_data = {}
price = price_source.get_factor(
    name=ptype,
    begin=begin,
    end=end,
)
for field in ["st", "suspended", "limit_up", "limit_down", "high", "low"]:
    price_data[field] = price_source.get_factor(
        name=field,
        begin=begin,
        end=end,
    )
feasible = (
    ~(price_data["st"].replace(np.nan, False))
    & ~(price_data["suspended"].replace(np.nan, False))
    & (price_data["low"] < price_data["limit_up"])
    & (price_data["high"] > price_data["limit_down"])
)
if benchmark_code is not None:
    benchmark = (
        DuckParquet(benchmark_path)
        .select(
            columns=[time_col, "close"],
            where=f"{code_col} = ? AND {time_col} >= ? AND {time_col} <= ?",
            params=[benchmark_code, begin, end],
            order_by=time_col,
        )
        .set_index(time_col)
        .squeeze()
    )
else:
    benchmark = None

factor_source = factool.DuckParquetSource(
    factor_path, time_col="date", code_col=code_col
)
factor_data = factor_source.get_factor(factor_name, begin=begin, end=end)

evaluator = factool.Evaluator(
    factor=factor_data,
    price=price,
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

pd.concat([evaluator.ic, evaluator.ic.cumsum()], axis=1, keys=["raw", "cumsum"]).plot(
    secondary_y="cumsum"
)
pd.concat(
    [
        evaluator.value_topk.to_frame("topk"),
        evaluator.value_ngroup,
        (benchmark / benchmark.iloc[0]).to_frame(benchmark_code),
    ],
    axis=1,
).plot(alpha=0.7)
pd.concat(
    [
        evaluator.evaluation_topk.to_frame("topk"),
        evaluator.evaluation_ngroup,
        (benchmark / benchmark.iloc[0]).to_frame(benchmark_code),
    ],
    axis=1,
)

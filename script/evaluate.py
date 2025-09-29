import factool
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from parquool import DuckParquet, setup_logger


def evaluate(
    factor_path: str,
    price_path: str,
    output_path: str,
    benchmark_code: str = "000985.CSI",
    benchmark_path: str = None,
    begin: str = "2025-01-01",
    end: str = "now",
    ptype: str = "open_post",
    freq: int = 5,
    weight: pd.DataFrame = None,
    topk: int = 100,
    ic_method: str = "spearman",
    n_group: int = 10,
    commission: float = 0.0005,
    time_col: str = "date",
    code_col: str = "code",
) -> list[factool.Evaluator]:
    begin = pd.to_datetime(begin)
    end = pd.to_datetime(end)
    logger = setup_logger("FactorEvaluator", level="INFO")

    logger.info("开始读取行情数据")
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
    logger.info("开始构造交易可行性")
    feasible = (
        ~(price_data["st"].replace(np.nan, False))
        & ~(price_data["suspended"].replace(np.nan, False))
        & (price_data["low"] < price_data["limit_up"])
        & (price_data["high"] > price_data["limit_down"])
    )

    logger.info("开始读取基准数据")
    if benchmark_code is not None:
        benchmark = (
            DuckParquet(benchmark_path or os.getenv("INDEXQUOTESDAY_PATH"))
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

    factors = pd.Index(DuckParquet(factor_path).list_columns()).difference(
        ["date", "code", "time"]
    )
    evaluators = []
    for factor_name in factors:
        output = Path(output_path) / f"{factor_name}_{ptype}_{freq}"
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"evaluate_{factor_name}.log"
        logger = setup_logger(factor_name, file=log_path, level="INFO")

        logger.info("开始读取因子数据")
        factor_source = factool.DuckParquetSource(
            factor_path, time_col=time_col, code_col=code_col
        )
        factor_data = factor_source.get_factor(factor_name, begin=begin, end=end)

        logger.info("开始评估因子")
        evaluator = factool.Evaluator(
            factor=factor_data,
            price=price,
            logger=logger,
        )
        evaluator(
            method=ic_method,
            n=n_group,
            k=topk,
            freq=freq,
            weight=weight,
            feasible=feasible,
            benchmark=benchmark,
            commission=commission,
        )

        logger.info("评估完成，开始保存结果")
        with pd.ExcelWriter(output / f"evaluation.xlsx") as writer:
            ics = pd.concat(
                [evaluator.ic, evaluator.ic.cumsum()], axis=1, keys=["raw", "cumsum"]
            )
            ics.to_excel(writer, sheet_name="IC")
            pd.concat(
                [evaluator.topk_result["evaluation"].to_frame("topk")]
                + [
                    res["evaluation"].to_frame(f"Group{i}")
                    for i, res in enumerate(evaluator.ngroup_result, start=1)
                ],
                axis=1,
            ).to_excel(writer, sheet_name="TopK and NGroup")

        ics.plot(secondary_y="cumsum", figsize=(20, 10))
        plt.savefig(output / f"ic.png")

        pd.concat(
            [evaluator.topk_result["values"].to_frame("topk")]
            + [
                res["values"].to_frame(f"Group{i}")
                for i, res in enumerate(evaluator.ngroup_result, start=1)
            ]
            + [(benchmark / benchmark.iloc[0]).to_frame(benchmark_code)],
            axis=1,
        ).interpolate().plot(alpha=0.7, figsize=(20, 10))
        plt.savefig(output / f"values.png")

        evaluator.name = factor_name
        evaluators.append(evaluator)
    
    return evaluators


if __name__ == "__main__":
    import os
    import dotenv
    from pathlib import Path

    dotenv.load_dotenv()
    evaluate(
        factor_path=Path(os.getenv("FACTOR_DATA_PATH")) / "market_size",
        price_path=os.getenv("QUOTESDAY_PATH"),
        benchmark_path=os.getenv("INDEXQUOTESDAY_PATH"),
        output_path=os.getenv("EVAL_PATH"),
        begin="2015-01-01",
        end="now",
    )

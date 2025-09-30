import os
import argparse
from pathlib import Path

import parquool
from calc import calc
from evaluate import evaluate
from factool import DuckParquetSource


class FactorAgent(parquool.Agent):

    @staticmethod
    def get_all_tables() -> str:
        """Get all availabel duck parquet table on your environment variable `DATASET_PATH`"""
        return ", ".join([p.stem for p in Path(os.getenv("DATASET_PATH")).iterdir()])

    @staticmethod
    def get_duckparquet_schema(table_name: str) -> str:
        """Get table schema that named `table_name`, this table should exist.
        You can get all the available table by using `get_all_tables`

        Args:
            table_name (str): table name.

        Return:
            (str) the table schema in markdown table format
        """
        table_name = Path(os.getenv("DATASET_PATH")) / table_name
        if not table_name.exists():
            raise ValueError(
                f"The target table: {table_name} does not exist, "
                "please try something valid in the result of `get_all_dataset`"
            )
        duckparquet = DuckParquetSource(table_name)
        return duckparquet.get_all_factors().to_markdown()

    def pipeline(
        self,
        doc: str,
        factor_py_path: str = None,
        factor_data_path: str = None,
        price_path: str = None,
        benchmark_path: str = None,
        eval_path: str = None,
        save: str = True,
        begin: str = "2015-01-01",
        end: str = "now",
        n_jobs: int = -1,
        evaluation: str = True,
        ptype: str = "open_post",
        benchmark_code: str = None,
        freq: int = 5,
        topk: int = 100,
        ic_method: str = "spearman",
        n_group: int = 10,
        commission: float = 0.0005,
    ):
        doc = Path(doc)
        prompt = f"现在，请你根据如下因子定义，按照指示开始编写名为{doc.stem}的因子定义函数\n\n" + doc.read_text(
            encoding="utf-8"
        )
        self.run_streamed_sync(prompt)

        factor_py_path = Path(factor_py_path or os.getenv("FACTOR_PY_PATH"))
        factor_py_path.mkdir(parents=True, exist_ok=True)
        factor_py_output = factor_py_path / f"{doc.stem}.py"
        Path(factor_py_output).write_text(
            self.get_conversation()[-1]["content"][-1]["text"], encoding="utf-8"
        )
        self.logger.info(f"因子代码已保存至 {factor_py_output}")

        if save:
            factor_data_path = Path(factor_data_path or os.getenv("FACTOR_DATA_PATH"))
            price_path = Path(price_path or os.getenv("QUOTESDAY_PATH"))
            calc(
                factor_def_path=factor_py_output,
                begin=begin,
                end=end,
                n_jobs=n_jobs,
                save_path=Path(os.getenv("FACTOR_DATA_PATH")) / doc.stem,
            )
            self.logger.info(f"因子数据已保存至 {save}")

        if evaluation:
            factor_data_path = Path(factor_data_path or os.getenv("FACTOR_DATA_PATH"))
            price_path = Path(price_path or os.getenv("QUOTESDAY_PATH"))
            benchmark_path = Path(benchmark_path or os.getenv("INDEXQUOTESDAY_PATH"))
            eval_path = Path(eval_path or os.getenv("EVAL_PATH")) / doc.stem
            eval_path.mkdir(exist_ok=True, parents=True)
            evaluators = evaluate(
                factor_path=Path(os.getenv("FACTOR_DATA_PATH")) / doc.stem,
                price_path=price_path,
                output_path=eval_path,
                benchmark_code=benchmark_code,
                benchmark_path=benchmark_path,
                begin=begin,
                end=end,
                ptype=ptype,
                freq=freq,
                topk=topk,
                ic_method=ic_method,
                n_group=n_group,
                commission=commission,
            )
            self.logger.info(f"因子评估结果已保存至 {evaluation}")


def main():
    parser = argparse.ArgumentParser(description="自动生成因子定义代码")
    parser.add_argument("doc", type=str, help="因子定义的 Markdown 文件路径")
    parser.add_argument("--factor_py_path", type=str, default=None, help="输出因子的 Python 文件路径")
    parser.add_argument("--factor_data_path", type=str, default=None, help="输出因子数据的路径（例如 parquet 文件夹）")
    parser.add_argument("--price_path", type=str, default=None, help="价格数据路径")
    parser.add_argument("--benchmark_path", type=str, default=None, help="基准数据路径")
    parser.add_argument("--eval_path", type=str, default=None, help="评估结果输出路径")
    parser.add_argument("--begin", type=str, default="2015-01-01", help="因子数据的开始时间")
    parser.add_argument("--end", type=str, default="now", help="因子数据的结束时间")
    parser.add_argument("--save", action="store_true", help="是否进行因子数据的计算与保存")
    parser.add_argument("--evaluation", action="store_true", help="是否进行因子评估")
    parser.add_argument("--n_jobs", type=int, default=-1, help="并行作业数，-1 表示使用所有核心")
    parser.add_argument("--ptype", type=str, default="open_post", help="价格类型/填充方法")
    parser.add_argument("--benchmark_code", type=str, default=None, help="基准代码")
    parser.add_argument("--freq", type=int, default=5, help="因子频率（例如 5 表示 5 分钟）")
    parser.add_argument("--topk", type=int, default=100, help="排名取前 k 个")
    parser.add_argument("--ic_method", type=str, default="spearman", help="IC 计算方法")
    parser.add_argument("--n_group", type=int, default=10, help="分组数量")
    parser.add_argument("--commission", type=float, default=0.0005, help="交易佣金率")
    args = parser.parse_args()

    factool_agent.pipeline(
        doc=args.doc,
        factor_py_path=args.factor_py_path,
        factor_data_path=args.factor_data_path,
        price_path=args.price_path,
        benchmark_path=args.benchmark_path,
        eval_path=args.eval_path,
        save=args.save,
        begin=args.begin,
        end=args.end,
        n_jobs=args.n_jobs,
        evaluation=args.evaluation,
        ptype=args.ptype,
        benchmark_code=args.benchmark_code,
        freq=args.freq,
        topk=args.topk,
        ic_method=args.ic_method,
        n_group=args.n_group,
        commission=args.commission,
    )

factool_agent = FactorAgent(
    tools=[FactorAgent.get_all_tables, FactorAgent.get_duckparquet_schema],
    instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
)

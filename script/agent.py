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

    def generate(
        self,
        doc: str,
        factorpy: str = None,
        begin: str = "2015-01-01",
        end: str = "now",
        save: str = True,
        evaluation: str = True,
    ):
        doc = Path(doc)
        prompt = f"现在，请你根据如下因子定义，按照指示开始编写名为{doc.stem}的因子定义函数\n\n" + doc.read_text(
            encoding="utf-8"
        )
        self.run_streamed_sync(prompt)
        output_dir = Path(factorpy or "out")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc.stem}.py"
        Path(output_path).write_text(
            self.get_conversation()[-1]["content"][-1]["text"], encoding="utf-8"
        )
        self.logger.info(f"因子代码已保存至 {output_path}")
        if save:
            calc(
                output_path,
                begin,
                end,
                save_path=Path(os.getenv("FACTORLAB_PATH")) / doc.stem,
            )
            self.logger.info(f"因子数据已保存至 {save}")
        if evaluation:
            evaluators = evaluate(
                factor_path=Path(os.getenv("FACTORLAB_PATH")) / doc.stem,
                price_path=os.getenv("QUOTESDAY_PATH"),
                benchmark_path=os.getenv("INDEXQUOTESDAY_PATH"),
                output_path=Path(os.getenv("EVAL_PATH")),
                begin=begin,
                end=end,
            )
            self.logger.info(f"因子评估结果已保存至 {evaluation}")


def main():
    parser = argparse.ArgumentParser(description="自动生成因子定义代码")
    parser.add_argument("doc", type=str, help="因子定义的 Markdown 文件路径")
    parser.add_argument(
        "--factorpy", type=str, default=None, help="输出因子的 Python 文件路径"
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="存储对话的路径，默认采用内存存储，程序运行后自动删除。",
    )
    parser.add_argument(
        "--begin", type=str, default="2025-01-01", help="因子数据的开始时间"
    )
    parser.add_argument("--end", type=str, default="now", help="因子数据的结束时间")
    parser.add_argument(
        "--save",
        action="store_true",
        help="是否进行因子数据的计算与保存",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="是否进行因子评估",
    )
    args = parser.parse_args()

    doc = args.doc
    factorpy = args.factorpy or os.getenv("FACTORPY_PATH")
    database = args.database or os.getenv("DB_PATH")
    begin = args.begin or "2015-01-01"
    end = args.end or "now"
    save = args.save
    evaluation = args.evaluation

    FactorAgent(
        tools=[FactorAgent.get_all_tables, FactorAgent.get_duckparquet_schema],
        instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
    ).generate(doc, factorpy, database, begin, end, save, evaluation)


factool_agent = FactorAgent(
    tools=[FactorAgent.get_all_tables, FactorAgent.get_duckparquet_schema],
    instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
)

import os
import ast
import argparse
import asyncio
from pathlib import Path

from markdown_it import MarkdownIt
from agents import function_tool
from factool import DuckParquetSource
from parquool import BaseAgent


class FactorAgent(BaseAgent):

    @function_tool
    @staticmethod
    def get_all_dataset() -> str:
        """获取数据库中所有可用的数据表"""
        return ", ".join([p.stem for p in Path(os.getenv("DATASET_PATH")).iterdir()])

    @function_tool
    @staticmethod
    def get_duckparquet_schema(dp_path: str) -> str:
        """获取dp_path对应的数据表中可用的因子/数据列信息"""
        dp_path = Path(os.getenv("DATASET_PATH")) / dp_path
        if not dp_path.exists():
            raise ValueError(
                f"The target path: {dp_path} does not exist, please try something valid in the result of `get_all_dataset`"
            )
        duckparquet = DuckParquetSource(dp_path)
        return duckparquet.get_all_factors().to_markdown()

    @staticmethod
    def simple_validate(code):
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.Attribute),
            ):
                raise ValueError(f"不允许的语法节点: {type(node).__name__}")

    @staticmethod
    def get_section_content(md_text: str, factor_names: list[str]):
        md = MarkdownIt()
        tokens = md.parse(md_text)
        results = []

        for factor_name in factor_names:
            found = False
            result = []
            current_level = None
            for i, token in enumerate(tokens):
                if token.type == "heading_open":
                    current_level = int(token.tag[1])
                    heading_content = tokens[i + 1].content
                    if factor_name in heading_content:
                        result.append(heading_content)
                        found = True
                        level = current_level
                        continue
                    if found and current_level == level:
                        break
                elif found:
                    if token.type == "paragraph_open":
                        paragraph = tokens[i + 1]
                        result.append(paragraph.content)
            results.append("\n\n".join(result))
        return "\n\n".join(filter(None, results))

    async def run(
        self,
        md_path: str,
        output_path: str,
        db_path: str,
        save_path: str,
    ):
        md_path = Path(md_path)
        prompt = md_path.read_text(encoding="utf-8")
        result = await super().run_streamed(prompt, db_path=db_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{md_path.stem}_generated.py"
        Path(output_path).write_text(result, encoding="utf-8")
        self.logger.info(f"因子代码已保存至 {output_path}")
        if save_path:
            pass


def main():
    parser = argparse.ArgumentParser(description="自动生成因子定义代码")
    parser.add_argument("doc_path", type=str, help="因子定义的 Markdown 文件路径")
    parser.add_argument(
        "output_path", type=str, default=None, help="输出因子的 Python 文件路径"
    )
    parser.add_argument(
        "-d",
        "--db_path",
        type=str,
        default=None,
        help="存储对话的路径，默认采用内存存储，程序运行后自动删除。",
    )
    parser.add_argument(
        "-s", "--save_path", type=str, default=None, help="输出因子数据的路径"
    )
    args = parser.parse_args()

    doc_path = args.doc_path
    db_path = args.db_path or os.getenv("DB_PATH")
    output_path = args.output_path or os.getenv("FACTORPY_PATH")
    save_path = args.save_path or os.getenv("FACTORLAB_PATH")

    asyncio.run(
        FactorAgent(
            tools=[FactorAgent.get_all_dataset, FactorAgent.get_duckparquet_schema],
            instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
        ).run(doc_path, output_path, db_path)
    )


if __name__ == "__main__":
    main()

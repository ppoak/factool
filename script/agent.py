import os
import argparse
import asyncio
from pathlib import Path

import parquool
from calc import calc
from evaluate import evaluate
from factool import DuckParquetSource
from openai.types.responses import ResponseTextDeltaEvent


def web():
    import streamlit as st

    async def stream(prompt):
        async for event in factool_agent.stream(prompt):
            # We'll print streaming delta if available
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield event.data.delta
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    yield f"{event.item.raw_item.name} - {event.item.raw_item.arguments}\n\n"
                elif event.item.type == "tool_call_output_item":
                    yield event.item.output
                else:
                    pass

    name = "Factool Agent"
    st.title(name)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.write_stream(stream(prompt))
        st.session_state.messages.append({"role": "assistant", "content": response})


class FactorAgent(parquool.Agent):

    @staticmethod
    def get_all_dataset() -> str:
        """获取数据库中所有可用的数据表"""
        return ", ".join([p.stem for p in Path(os.getenv("DATASET_PATH")).iterdir()])

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

    def __call__(
        self,
        doc: str,
        factorpy: str,
        database: str,
        begin: str,
        end: str,
        save: str,
        evaluation: str,
    ):
        doc = Path(doc)
        prompt = f"现在，请你根据如下因子定义，按照指示开始编写名为{doc.stem}的因子定义函数\n\n" + doc.read_text(
            encoding="utf-8"
        )
        result = super().run_streamed_sync(prompt, db_path=database)
        output_dir = Path(factorpy)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{doc.stem}.py"
        Path(output_path).write_text(result.final_output, encoding="utf-8")
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

    asyncio.run(
        FactorAgent(
            tools=[FactorAgent.get_all_dataset, FactorAgent.get_duckparquet_schema],
            instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
        )(doc, factorpy, database, begin, end, save, evaluation)
    )


factool_agent = FactorAgent(
    tools=[FactorAgent.get_all_dataset, FactorAgent.get_duckparquet_schema],
    instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
)


if __name__ == "__main__":
    web()

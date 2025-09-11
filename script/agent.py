import argparse
import asyncio
import os
from pathlib import Path
import uuid

from agents import (
    Agent,
    ItemHelpers,
    ModelSettings,
    Runner,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from markdown_it import MarkdownIt
from openai import AsyncOpenAI

from factorlab import DuckParquetSource
from quool import setup_logger


def setup_environment():
    """初始化运行环境"""
    load_dotenv()
    set_default_openai_client(
        client=AsyncOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        ),
        use_for_tracing=False,
    )
    set_default_openai_api(api="chat_completions")
    set_tracing_disabled(disabled=True)


def get_all_dataset() -> str:
    """获取数据库中所有可用的数据表"""
    return ", ".join([p.stem for p in Path(os.getenv("DATASET_PATH")).iterdir()])


def get_duckparquet_schema(dp_path: str) -> str:
    """获取dp_path对应的数据表中可用的因子/数据列信息"""
    dp_path = Path(os.getenv("DATASET_PATH")) / dp_path
    if not dp_path.exists():
        raise ValueError(
            f"The target path: {dp_path} does not exist, please try something valid in the result of `get_all_dataset`"
        )
    duckparquet = DuckParquetSource(dp_path)
    return duckparquet.get_all_factors().to_markdown()


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


async def run_agent(
    md_path: str, factor_names: str, output_path: str = None, db_path: str = None
):
    logger = setup_logger("agents")
    setup_environment()
    session = SQLiteSession(uuid.uuid4().hex, db_path=db_path or ":memory:")
    codegen_agent = Agent(
        name="Code Generate Agent",
        instructions=Path("docs/code_gen_agent.md").read_text(encoding="utf-8"),
        tools=[function_tool(get_duckparquet_schema), function_tool(get_all_dataset)],
        model="v36-free.gpt-4o-mini",
        model_settings=ModelSettings(temperature=0.1),
    )

    result = Runner.run_streamed(
        codegen_agent,
        get_section_content(Path(md_path).read_text(encoding="utf-8"), factor_names),
    )
    async for event in result.stream_events():
        # We'll ignore the raw responses event deltas
        if event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            logger.debug(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                logger.info(
                    f"Tool was called: {event.item.raw_item.name}(**{event.item.raw_item.arguments})"
                )
            elif event.item.type == "tool_call_output_item":
                logger.info(f"-- Tool output: \n{event.item.output}")
            elif event.item.type == "message_output_item":
                logger.info(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                )
            else:
                pass  # Ignore other event types    if output_path is None:
        output_file_name = Path(md_path).stem + ".py"
        output_dir = Path("factor/contrib")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file_name
    Path(output_path).write_text(result.final_output, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="自动生成因子定义代码")
    parser.add_argument("path", type=str, help="因子定义的 Markdown 文件路径")
    parser.add_argument(
        "names", type=str, help="因子名称(用逗号分隔, 如 '因子A,因子B')"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="输出因子代码的文件路径，默认为 factor/contrib/与md文件同名的py后缀文件。",
    )
    parser.add_argument(
        "-d",
        "--db_path",
        type=str,
        default=None,
        help="存储对话的路径，默认采用内存存储，程序运行后自动删除。",
    )
    args = parser.parse_args()

    md_path = args.path
    factor_names = [name.strip() for name in args.names.split(",") if name.strip()]
    output_path = args.output_path
    db_path = args.db_path

    asyncio.run(run_agent(md_path, factor_names, output_path, db_path))


if __name__ == "__main__":
    main()

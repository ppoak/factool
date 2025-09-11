import asyncio
import os
from pathlib import Path
import uuid

from agents import (
    Agent,
    ModelSettings,
    Runner,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from factorlab import DuckParquetSource


def setup_environment():
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
    duckparquet = DuckParquetSource(Path(os.getenv("DATASET_PATH")) / dp_path)
    return duckparquet.get_all_factors().to_markdown()


async def main():

    setup_environment()
    session = SQLiteSession(uuid.uuid4().hex, db_path="data/test.db")
    codegen_agent = Agent(
        name="Code Generate Agent",
        instructions="""
    你现在是一个帮助量化研究员自动编写Python因子代码的助手。
    请根据给定的因子名称、功能描述和详细计算过程，按如下要求输出Python代码。

    【输出要求】
    - 只输出完整的Python代码，不添加任何解释说明或前后缀。
    - 代码需包含所有必要的import、函数定义和返回部分，并与如下示范结构一致：
    import os
    import pandas as pd
    from typing import Union
    from ..source import DuckParquetSource

    def calc_factorname(time: Union[str, pd.Timestamp]) -> Union[pd.Series, pd.DataFrame]:
        ...
        return ...

    - 不要输出markdown格式，只输出纯代码（不要```python标记，也不要文字说明）。
    - 只生成一个calc_因子名函数，每次只针对一个因子。

    【参考代码及说明】
    ```python
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from ..source import DuckParquetSource


    def calc_market_size(time: str | pd.Timestamp) -> pd.Series:
        source = DuckParquetSource(os.getenv("QUOTESDAY_PATH"), time_col="date")
        shares = source.get_factor("circulation_a", begin=time, end=time)
        price = source.get_factor("close_post", begin=time, end=time)
        log = np.log(shares * price).squeeze()
        model = sm.OLS((log**3).dropna(), sm.add_constant(log).dropna()).fit()
        nonlinear = model.resid
        return pd.concat(
            [log, nonlinear], axis=1, keys=["log_market_size", "nonlinear_market_size"]
        )
    ```

    - 在编码前请根据计算步骤明确需要的数据源有哪些，并使用工具获取可用数据源及数据源对应的信息。如果存在计算因子的数据源未在可用数据源中找到，直接告诉用户无法完成这个需求，列明原因。
    - 所有因子函数命名须符合规范：calc_<因子名>
    - 因子数据源source可获取指定时间范围内的因子值，以pd.DataFrame的形式返回宽表（列为股票代码）。
    """,
        tools=[function_tool(get_duckparquet_schema), function_tool(get_all_dataset)],
        model="v36-free.gpt-4o-mini",
        model_settings=ModelSettings(temperature=0.1),
    )

    result = Runner.run_streamed(
        codegen_agent, "请帮我完成一个成交量加权平均价格因子的代码"
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)
    Path("test/generated.py").write_text(result.final_output, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())

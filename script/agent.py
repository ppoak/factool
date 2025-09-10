import os
import re
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

CODE_GENERATE_PROMPT = """
你现在是一个帮助量化研究员自动编写Python因子代码的助手。
请根据给定的因子名称、功能描述和详细计算过程，按如下要求输出Python代码。

【输出要求】
- 只输出完整的Python代码，不添加任何解释说明或前后缀。
- 代码需包含所有必要的import、函数定义和返回部分，并与如下示范结构一致：
  import os
  import pandas as pd
  from typing import Union
  from ..source import DuckParquetSource

  def calc_因子名(time: Union[str, pd.Timestamp]) -> Union[pd.Series, pd.DataFrame]:
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

- 所有因子函数命名须符合规范：calc_<因子名>
- 因子数据源source可获取指定时间范围内的因子值，以pd.DataFrame的形式返回宽表（列为股票代码）。
- 目前可用作数据源初始化的环境变量还有QUOTESDAY_PATH、QUOTESMIN_PATH
- 请严格遵守用户提供的计算逻辑

【输入】
因子名称: {factor_name}
因子描述: {description}
详细计算过程: {details}
"""


class FactorAgent:
    
    def __init__(
        self,
        base_url=None,
        api_key=None,
        model=None,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_MODEL_NAME")

    def generate_code(self, factor_name, description, details, save_path):
        prompt = CODE_GENERATE_PROMPT.format(
            factor_name=factor_name, description=description, details=details
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        code = response.choices[0].message.content

        code = re.sub(r"^```[^\n]*\n", "", code)
        code = re.sub(r"```$", "", code)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(code)

    def calc(self):
        pass

    def save(self):
        pass

    def backtest(self):
        pass

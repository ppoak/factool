import os
from agent import create
from pathlib import Path
from typing import Union


def generate(doc: str, factor_py_path: str, db: Union[str, Path] = "out/agent.db"):
    doc = Path(doc)
    agent = create(session_id=doc.stem, session_db=db)
    prompt = (
        f"现在，请你根据如下因子定义，按照指示开始编写名为{doc.stem}的因子定义函数\n\n"
        + doc.read_text(encoding="utf-8")
    )
    agent.run_streamed_sync(prompt)
    factor_py_path = Path(factor_py_path or os.getenv("FACTOR_PY_PATH"))
    factor_py_path.mkdir(parents=True, exist_ok=True)
    factor_py_output = factor_py_path / f"{doc.stem}.py"
    Path(factor_py_output).write_text(
        agent.get_conversation()[-1]["content"][-1]["text"], encoding="utf-8"
    )
    agent.logger.info(f"因子代码已保存至 {factor_py_output}")


if __name__ == "__main__":
    doc = "docs/definitions/barra_factor/barra_momentum.md"
    factor_py_path = "generated/"
    db = ":memory:"

    generate(doc, factor_py_path, db)

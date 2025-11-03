import os
from agent import create
from pathlib import Path
from typing import Union


def generate(doc: str, factor_py_path: str, db: Union[str, Path] = "out/agent.db"):
    """
    Generate a Python factor definition file from a factor specification document using an agent.

    This function reads a factor definition document, constructs a Chinese-language prompt
    that instructs the agent to write a factor definition function named after the document's
    stem, runs the agent in a streamed synchronous mode, and saves the agent's last message
    content to a .py file. The output file is created under the specified directory with UTF-8
    encoding, and the filename will be <doc.stem>.py.

    Args:
        doc (str): Path to the factor definition document (e.g., a Markdown file).
            The file's stem (basename without extension) is used as the agent session_id
            and as the output Python filename.
        factor_py_path (str): Directory in which to save the generated Python file.
            If falsy, the function attempts to read the directory from the FACTOR_PY_PATH
            environment variable. The directory is created if it does not exist.
        db (Union[str, Path], optional): Path to the agent session database. Use ":memory:"
            to run with an in-memory database. Defaults to "out/agent.db".

    Returns:
        None

    Raises:
        FileNotFoundError: If the factor definition document does not exist.
        UnicodeDecodeError: If the document cannot be decoded as UTF-8.
        TypeError: If no output directory can be determined (factor_py_path is falsy and
            FACTOR_PY_PATH is not set).
        OSError: If creating the output directory or writing the file fails.
        IndexError or KeyError: If the agent conversation does not contain the expected content
            structure.
        Exception: Any error propagated by the agent during run_streamed_sync.

    Notes:
        - The saved file content is taken from agent.get_conversation()[-1]["content"][-1]["text"].
        - The function logs a message indicating the path to the saved file via agent.logger.

    Examples:
        generate(
            doc="docs/definitions/barra_factor/barra_momentum.md",
            factor_py_path="generated/",
            db=":memory:",
        )
    """
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

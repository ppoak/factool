import os
import parquool
from pathlib import Path
from factool import DuckParquetSource


def get_all_tables() -> str:
    """Get all availabel duck parquet table on your environment variable `DATASET_PATH`"""
    return ", ".join([p.stem for p in Path(os.getenv("DATASET_PATH")).iterdir()])


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


def create(session_id: str = None, session_db: str = "factool_agent.db"):
    """Create an agent based instance with assigned seesion_id and session_db

    Args:
        session_id (str): session id assigned.
        session_db (str): session db path assigned.
    """

    return parquool.Agent(
        tools=[get_all_tables, get_duckparquet_schema],
        instructions=Path("docs/CODEGEN_AGENT.md").read_text(encoding="utf-8"),
        session_db=session_db,
        session_id=session_id,
    )

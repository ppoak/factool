import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factool import DuckPQSource


def get_latest_date(source: 'DuckPQSource', table: str):
    source.register(table)
    if source.tables[table].empty:
        return
    date = source.query(f"SELECT MAX({source.time_col}) FROM {table}").squeeze()
    return date


def get_ealiest_date(source: 'DuckPQSource', table: str):
    source.register(table)
    if source.tables[table].empty:
        return
    date = source.query(f"SELECT MIN({source.time_col}) FROM {table}").squeeze()
    return date


def get_date_gap(
    source: 'DuckPQSource',
    target_table: str,
    base_table: str,
    default: str = "1900-01-01",
):
    target_latest = get_latest_date(source, target_table) or default
    base_latest = get_latest_date(source, base_table) or default
    return target_latest, base_latest


if __name__ == "__main__":
    from parquool import setup_logger

    logger = setup_logger("info")
    logger.info("launched")
    source = DuckPQSource("data")
    logger.info("initiated")
    print(get_latest_date(source, "quotes_day"))
    logger.info("completed")

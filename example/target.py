"""
# Factor Target

** Definition **

使用后复权开盘价、后复权收盘价、VWAP、TWAP作为因子值

** Step **

1. 取quotes_day表，取出date、code、open_post、close_post列数据
2. 取quotes_min表，计算对于每日（date）、每只股票（code）的分钟成交量与分钟收盘价乘积之和与成交量之和的比值（SUM(volume * close) / SUM(volume)）作为VWAP
2. 取quotes_min表，计算对于每日（date）、每只股票（code）的分钟分钟收盘价平均值（AVERAGE(close)）作为TWAP

"""

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv
from factool import DuckPQSource

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
FACTOR_DATA_PATH = os.getenv("FACTOR_DATA_PATH")

if not DATASET_PATH:
    raise ValueError("Missing env var: DATASET_PATH")
if not FACTOR_DATA_PATH:
    raise ValueError("Missing env var: FACTOR_DATA_PATH")

source = DuckPQSource(DATASET_PATH)
source.register("quotes_day")
source.register("quotes_min")


# -----------------------------
# Config
# -----------------------------
FACTOR_TABLE_NAME = "target"  # 入库表名
FACTOR_COLS = [
    "open",
    "high",
    "low",
    "close",
    "open_post",
    "high_post",
    "low_post",
    "close_post",
    "vwap",
    "twap",
    "volume",
]  # 最终输出列名


# -----------------------------
# Helpers
# -----------------------------
def month_ranges(begin: str, end: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generate [month_begin, next_month_begin) ranges in UTC-naive pandas timestamps.
    """
    begin_ts = pd.Timestamp(begin).normalize()
    end_ts = pd.Timestamp(end).normalize()

    if end_ts <= begin_ts:
        raise ValueError(
            f"end must be greater than begin, got begin={begin_ts}, end={end_ts}"
        )

    months = pd.date_range(begin_ts, end_ts, freq="MS")
    if len(months) == 0 or months[0] != begin_ts.replace(day=1):
        # ensure we start at month start containing begin_ts
        months = pd.date_range(begin_ts.replace(day=1), end_ts, freq="MS")

    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for m in months:
        m_begin = m
        m_end = m + pd.offsets.MonthBegin(1)
        # clip to [begin, end)
        r_begin = max(m_begin, begin_ts)
        r_end = min(m_end, end_ts)
        if r_end > r_begin:
            ranges.append((r_begin, r_end))
    return ranges


# -----------------------------
# User params (set your dates)
# -----------------------------
BEGIN = os.getenv("FACTOR_BEGIN", "2015-01-01")
END = os.getenv("FACTOR_END", "2025-12-31")  # end为开区间：[BEGIN, END)


# -----------------------------
# Step 1: quotes_day open_post/close_post
# -----------------------------
sql_day = f"""
SELECT
  CAST(date AS TIMESTAMP) AS date,
  code,
  open_post,
  high_post,
  low_post,
  close_post,
  open,
  high,
  low,
  close,
  volume  
FROM quotes_day
WHERE date >= DATE '{pd.Timestamp(BEGIN).date()}'
  AND date <  DATE '{pd.Timestamp(END).date()}'
"""
day_df = source.query(sql_day)
day_df["date"] = pd.to_datetime(day_df["date"])
day_df = day_df.sort_values(["date", "code"]).reset_index(drop=True)


# -----------------------------
# Step 2 & 3: quotes_min compute daily VWAP/TWAP (monthly to avoid huge IO)
# -----------------------------
monthly_parts: List[pd.DataFrame] = []

for m_begin, m_end in month_ranges(BEGIN, END):
    sql_min_agg = f"""
    SELECT
      CAST(date AS TIMESTAMP) AS date,
      code,
      SUM(volume * close_post) / NULLIF(SUM(volume), 0) AS vwap,
      AVG(close_post) AS twap
    FROM quotes_min
    WHERE date >= DATE '{m_begin.date()}'
      AND date <  DATE '{m_end.date()}'
    GROUP BY date, code
    """
    min_agg_df = source.query(sql_min_agg)
    min_agg_df["date"] = pd.to_datetime(min_agg_df["date"])
    min_agg_df = min_agg_df.sort_values(["date", "code"]).reset_index(drop=True)
    monthly_parts.append(min_agg_df)

min_df = (
    pd.concat(monthly_parts, axis=0, ignore_index=True)
    if monthly_parts
    else pd.DataFrame(columns=["date", "code", "vwap", "twap"])
)


# -----------------------------
# Merge: open_post/close_post + vwap/twap
# -----------------------------
factor_df = day_df.merge(min_df, on=["date", "code"], how="left")

# Ensure final format: MultiIndex (date, code), factor columns
factor_df["date"] = pd.to_datetime(factor_df["date"])
factor_df = factor_df.set_index(["date", "code"]).sort_index()
factor_df = factor_df[FACTOR_COLS].astype("float64")

# factor_df is the final long-form DataFrame with MultiIndex (date, code)
# columns: open_post, close_post, vwap, twap


# -----------------------------
# Save to factor storage
# -----------------------------
DuckPQSource(Path(FACTOR_DATA_PATH)).save(
    FACTOR_TABLE_NAME,
    factor_df,
    processors=None,
)

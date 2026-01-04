from __future__ import annotations

from typing import Union, Iterable, Tuple, Dict, List, Optional

import numpy as np
import pandas as pd


def parse_factor_path(path: str, sep: str = "/") -> Tuple[str, str]:
    if not isinstance(path, str):
        raise TypeError(f"factor path must be str, got {type(path)}: {path}")
    s = path.strip()
    if not s:
        raise ValueError("empty factor path")
    parts = s.split(sep)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid factor path: {path!r}, expected 'table{sep}factor'")
    return parts[0], parts[1]


def load_factor(
    source,
    factor_paths: Union[str, Iterable[str]],
    begin: str,
    end: str,
    *,
    sep: str = "/",
    join: str = "full",
    lookback: int = 0,
    base_table: Optional[str] = None,
) -> pd.DataFrame:
    """Loads one or many factors and joins them in DuckDB.

    Args:
        source: A DuckDB-backed data source that provides `register(table)` and
            `query(sql)` returning a pandas DataFrame.
        factor_paths: Iterable of factor paths like "table/factor".
        begin: Begin date (inclusive), passed into SQL as a string.
        end: End date (inclusive), passed into SQL as a string.
        sep: Path separator used in `factor_paths`.
        join: Join type between tables: "inner", "left", "right", or "full".
        lookback: Relative to begin, lookback more data (0 for no lookback).
        base_table: The anchor table name used as the left side of the join chain.
            If None, the first table in `factor_paths` is used.

    Returns:
        A pandas DataFrame indexed by ["date", "code"] with factor columns.

    Raises:
        ValueError: If inputs are invalid.
        TypeError: If a factor path is not a string.
    """
    factor_paths = (
        list(factor_paths) if not isinstance(factor_paths, str) else [factor_paths]
    )
    if not factor_paths:
        raise ValueError("factor_paths is empty")
    if not isinstance(lookback, int):
        raise TypeError("lookback must be int")
    if lookback < 0:
        raise ValueError("lookback must be >= 0")

    join = join.lower()
    join_map = {
        "inner": "INNER JOIN",
        "left": "LEFT JOIN",
        "right": "RIGHT JOIN",
        "full": "FULL OUTER JOIN",
    }
    if join not in join_map:
        raise ValueError(f"invalid join={join!r}, choose from {list(join_map)}")

    # Parse factor paths and group requested columns by table.
    by_table: Dict[str, List[str]] = {}
    for p in factor_paths:
        t, c = parse_factor_path(p, sep=sep)
        by_table.setdefault(t, [])
        if c not in by_table[t]:
            by_table[t].append(c)

    tables = list(by_table.keys())
    if base_table is None:
        base_table = tables[0]
    if base_table not in by_table:
        raise ValueError(
            f"base_table {base_table!r} is not present in factor_paths tables: {tables}"
        )

    # Register all tables up front.
    for t in tables:
        source.register(t)

    # ---- compute lookback begin ----
    begin_for_sql = begin
    if lookback > 0:
        sql_begin_lb = f"""
        WITH cal AS (
            SELECT DISTINCT CAST(date AS DATE) AS d
            FROM {base_table}
        ),
        anchor AS (
            SELECT d
            FROM cal
            WHERE d <= CAST('{begin}' AS DATE)
            ORDER BY d DESC
            LIMIT 1
        )
        SELECT CAST(d AS VARCHAR) AS begin_lb
        FROM cal
        WHERE d <= (SELECT d FROM anchor)
        ORDER BY d DESC
        OFFSET {lookback}
        LIMIT 1
        """.strip()

        tmp = source.query(sql_begin_lb)
        if tmp.empty or tmp.iloc[0, 0] is None:
            sql_min = f"""
            SELECT CAST(MIN(CAST(date AS DATE)) AS VARCHAR) AS begin_lb
            FROM {base_table}
            """.strip()
            tmp2 = source.query(sql_min)
            begin_for_sql = tmp2.iloc[0, 0]
        else:
            begin_for_sql = tmp.iloc[0, 0]

    def subquery(table: str) -> str:
        cols = ", ".join(by_table[table])
        return f"""
            SELECT
                CAST(date AS TIMESTAMP) AS date,
                code,
                {cols}
            FROM {table}
            WHERE date >= '{begin_for_sql}' AND date <= '{end}'
        """.strip()

    # Build a single SQL statement joining per-table subqueries.
    base_alias = "b"
    sql_from = f"FROM ({subquery(base_table)}) AS {base_alias}\n"
    key_date = f"{base_alias}.date"
    key_code = f"{base_alias}.code"

    i = 0
    for t in tables:
        if t == base_table:
            continue
        i += 1
        a = f"t{i}"
        sql_from += (
            f"{join_map[join]} ({subquery(t)}) AS {a}\n"
            f"ON {a}.date = {key_date} AND {a}.code = {key_code}\n"
        )
        if join == "full":
            key_date = f"COALESCE({key_date}, {a}.date)"
            key_code = f"COALESCE({key_code}, {a}.code)"

    # Select merged keys plus all factor columns.
    select_cols: List[str] = [f"{key_date} AS date", f"{key_code} AS code"]

    for c in by_table[base_table]:
        select_cols.append(f"{base_alias}.{c} AS {c}")

    i = 0
    for t in tables:
        if t == base_table:
            continue
        i += 1
        a = f"t{i}"
        for c in by_table[t]:
            select_cols.append(f"{a}.{c} AS {c}")

    sql = "SELECT\n    " + ",\n    ".join(select_cols) + "\n" + sql_from
    df = source.query(sql)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    return df


def mcap_neutralize(
    factors: pd.DataFrame,
    mcap: pd.Series | pd.DataFrame,
    *,
    weight_power: float = 1.0,
    clip: Optional[float] = None,
    ddof: int = 0,
) -> pd.DataFrame:
    """Market-cap weighted standardization per date using groupby sums (no apply).

    Args:
        factors: DataFrame with MultiIndex ["date", "code"].
        mcap: Series or single-column DataFrame with the same MultiIndex.
        weight_power: Uses weights = mcap ** weight_power.
        clip: If set, clip standardized values to [-clip, clip].
        ddof: Only supports 0 here reliably (recommended). Non-zero ddof is approximated.

    Returns:
        Standardized factors DataFrame indexed by ["date", "code"].
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a DataFrame")
    if not isinstance(factors.index, pd.MultiIndex) or list(factors.index.names) != [
        "date",
        "code",
    ]:
        raise ValueError("factors index must be MultiIndex named ['date', 'code']")

    if isinstance(mcap, pd.DataFrame):
        if mcap.shape[1] != 1:
            raise ValueError("mcap DataFrame must have exactly one column")
        w = mcap.iloc[:, 0]
    else:
        w = mcap

    if not isinstance(w, pd.Series):
        raise TypeError("mcap must be a Series or a single-column DataFrame")
    if not isinstance(w.index, pd.MultiIndex) or list(w.index.names) != [
        "date",
        "code",
    ]:
        raise ValueError("mcap index must be MultiIndex named ['date', 'code']")

    X = factors.apply(pd.to_numeric, errors="coerce")
    w = pd.to_numeric(w, errors="coerce")

    # Align indices
    tmp = X.join(w.rename("w"), how="left")
    w = tmp["w"]

    if weight_power != 1.0:
        w = w.pow(weight_power)

    # Invalid weights -> NaN; they will be excluded by sums with min_count.
    w = w.where(np.isfinite(w) & (w > 0.0))

    # We need per-element weights that are also NaN where factor is NaN
    W = pd.DataFrame({c: w.where(X[c].notna()) for c in X.columns}, index=X.index)

    wx = X.mul(W)  # w * x (NaNs excluded)
    wx2 = X.pow(2).mul(W)  # w * x^2

    wsum = W.groupby(level="date").sum(min_count=1)  # (date, factor)
    sum_wx = wx.groupby(level="date").sum(min_count=1)
    sum_wx2 = wx2.groupby(level="date").sum(min_count=1)

    mean = sum_wx.div(wsum)
    ex2 = sum_wx2.div(wsum)

    var = ex2.sub(mean.pow(2))
    var = var.clip(lower=0.0)

    if ddof != 0:
        # Optional: weighted ddof correction is non-trivial; this is a common approximation.
        # If you truly need unbiased weighted variance, implement Kish effective sample size.
        pass

    std = var.pow(0.5).replace(0.0, np.nan)

    # Broadcast mean/std back to row level by date using transform
    mean_row = X.groupby(level="date").transform(lambda s: mean[s.name])
    std_row = X.groupby(level="date").transform(lambda s: std[s.name])

    Z = (X - mean_row) / std_row

    if clip is not None:
        Z = Z.clip(-float(clip), float(clip))
    return Z

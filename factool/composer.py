from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union, Any

import numpy as np
import pandas as pd

from .storage import DuckPQSource


ReturnsLoader = Callable[..., pd.DataFrame]


def _ensure_midx(df_or_s: Union[pd.DataFrame, pd.Series], name: str) -> None:
    if not isinstance(df_or_s.index, pd.MultiIndex) or list(df_or_s.index.names) != [
        "date",
        "code",
    ]:
        raise ValueError(f"{name} index must be MultiIndex named ['date','code']")


@dataclass
class ComposerConfig:
    factor_paths: Union[str, Sequence[str]]
    begin: str
    end: str
    horizon: int
    sep: str = "/"
    join: str = "full"
    window: int = 0
    base_table: Optional[str] = None
    target_path: str = "target/open"


class Composer:

    def __init__(
        self,
        source: DuckPQSource,
        config: ComposerConfig,
    ):
        self.source = source
        self.config = config
        self.info: dict[str, Any] = {}

    # ---------- data loading ----------
    def load_X(self) -> pd.DataFrame:
        cfg = self.config
        X = self.source.load(
            factor_paths=cfg.factor_paths,
            begin=cfg.begin,
            end=cfg.end,
            sep=cfg.sep,
            join=cfg.join,
            pad_begin=cfg.window,
            base_table=cfg.base_table,
        )
        _ensure_midx(X, "factors")
        # ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.sort_index()
        return X

    def load_y(self) -> pd.Series:
        cfg = self.config
        y_df = self.source.load(
            factor_paths=cfg.target_path,
            begin=cfg.begin,
            end=cfg.end,
            sep=cfg.sep,
            join=cfg.join,
            pad_begin=cfg.window,
            pad_end=cfg.horizon + 1,
            base_table=cfg.base_table,
        )
        _ensure_midx(y_df, "forward_ret")
        if not isinstance(y_df, pd.DataFrame) or y_df.shape[1] != 1:
            raise ValueError(
                "returns_loader must return a single-column DataFrame like ['returns']"
            )
        y_df = (
            y_df.groupby("code").shift(-1)
            / y_df.groupby("code").shift(-1 - cfg.horizon)
            - 1
        )
        y = y_df.iloc[:, 0].rename("returns").sort_index()
        return y

    def align_Xy(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        # strict intersection to avoid mismatched rows; no forward-fill to avoid leakage
        idx = X.index.intersection(y.index)
        X2 = X.loc[idx]
        y2 = y.loc[idx]
        return X2, y2

    # ---------- hooks ----------
    def zscore(self, X: pd.DataFrame):
        Z = (X - X.groupby("date").mean()) / X.groupby("date").std()
        return Z

    def preprocess(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        # Base class: no-op; subclass can implement (winsorize/zscore/neutralize etc.)
        return X, y

    def postprocess(self, factor: pd.DataFrame) -> pd.DataFrame:
        return factor

    # ---------- main ----------
    def run(self) -> pd.DataFrame:
        X = self.load_X()
        y = self.load_y()
        X, y = self.align_Xy(X, y)
        X, y = self.preprocess(X, y)
        out = self.compose(X, y)
        out = self.postprocess(out)

        # minimal safety
        if not isinstance(out, pd.DataFrame) or out.shape[1] != 1:
            raise ValueError("compose must return a single-column DataFrame")
        _ensure_midx(out, "output factor")

        self.info = {
            "begin": self.config.begin,
            "end": self.config.end,
            "horizon": self.config.horizon,
            "ptype": self.config.target_path,
            "factor_paths": list(self.config.factor_paths),
            "n_rows": len(out),
        }
        return out

    # to be implemented by subclasses
    def compose(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

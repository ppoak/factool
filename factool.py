import math
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from logging import Logger

from parquool import DuckParquet, setup_logger


class DuckParquetSource:

    def __init__(
        self,
        dataset_path: str,
        time_col: str = "date",
        code_col: str = "code",
        name: str = None,
        db_path: str = None,
        threads: int = 4,
    ) -> None:
        self.dp = DuckParquet(dataset_path, name, db_path, threads)
        self.time_col = time_col
        self.code_col = code_col

    def get_times(self, begin: str = None, end: str = None, time_col: str = "date"):
        times = self.dp.select(
            columns=f"{time_col} AS time",
            where="time >= ? AND time <= ?",
            params=[
                pd.to_datetime(begin or "1990-01-01"),
                pd.to_datetime(end or "now"),
            ],
            distinct=True,
            order_by="time",
        ).squeeze()
        return pd.to_datetime(times)

    def get_time(self, time: str, n: int):
        if n > 0:
            return self.get_times(None, time).iloc[-n - 1]
        return self.get_times(time, None).iloc[-n]

    def get_all_factors(self) -> pd.DataFrame:
        schema = self.dp.get_schema()
        schema = schema[~schema["column_name"].isin([self.time_col, self.code_col])]
        return schema

    def get_factor(
        self,
        name: str,
        begin: pd.Timestamp | str = None,
        end: pd.Timestamp | str = None,
    ) -> pd.DataFrame:
        begin = pd.to_datetime(begin or "2000-01-01")
        end = pd.to_datetime(end or "now")
        return self.dp.dpivot(
            index=self.time_col,
            columns=self.code_col,
            values=name,
            where=f"{self.time_col} >= '{begin}' AND {self.time_col} <= '{end}'",
            order_by=self.time_col,
        ).set_index(self.time_col)

    def save(
        self,
        df: pd.DataFrame,
        name: str = "factor",
        processors: list[callable] = None,
    ):
        processors = processors or [
            Operator.zscore,
            partial(Operator.madoutlier, dev=5),
        ]
        names = "__".join(
            [
                (
                    processor.__name__
                    if not isinstance(processor, partial)
                    else processor.func.__name__
                    + "_"
                    + "_".join([f"{k}{v}" for k, v in processor.keywords.items()])
                )
                for processor in processors
            ]
        )
        if df.index.nlevels == 1:
            for processor in processors:
                processed = processor(df)
            factor = pd.concat(
                [processed.stack(), df.stack()],
                keys=[name, f"{name}__{names}"],
                axis=1,
            ).reset_index(names=[self.time_col, self.code_col])

        elif df.index.nlevels == 2:
            factors = [df[col].unstack() for col in df.columns]
            for processor in processors:
                factors = [processor(factor) for factor in factors]
            factor = pd.concat(
                [df]
                + [
                    factor.stack().to_frame(df.columns[i] + f"__{names}")
                    for i, factor in enumerate(factors)
                ],
                axis=1,
            ).reset_index(names=[self.time_col, self.code_col])

        factor["date"] = factor[self.time_col].dt.strftime("%Y-%m-%d")
        self.dp.upsert_from_df(
            factor, keys=[self.time_col, self.code_col], partition_by=["date"]
        )

    def __str__(self):
        return super().__str__() + f"(\n{self.get_all_factors()}\n)"

    def __repr__(self):
        return super().__repr__()


class Operator:

    @staticmethod
    def add(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.add(dfb, fill_value=fillna)
        return dfa + dfb

    @staticmethod
    def sub(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.sub(dfb, fill_value=fillna)
        return dfa - dfb

    @staticmethod
    def mul(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.mul(dfb, fill_value=fillna)
        return dfa * dfb

    @staticmethod
    def div(dfa: pd.DataFrame, dfb: pd.DataFrame, fillna: bool | float = False):
        if fillna:
            return dfa.div(dfb, fill_value=fillna)
        return dfa / dfb

    @staticmethod
    def mask(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
        return dfa.mask(dfb, other=np.nan)

    @staticmethod
    def where(dfa: pd.DataFrame, dfb: pd.DataFrame, dfc: pd.DataFrame | float = np.nan):
        return dfa.where(dfb, other=dfc)

    @staticmethod
    def shift(df: pd.DataFrame, n: int):
        return df.shift(n)

    @staticmethod
    def rsum(df: pd.DataFrame, n: int, axis: int = 0):
        if n < 0:
            return df.expanding(min_periods=-n, axis=axis).sum()
        elif n > 0 and n < 1:
            return df.ewm(alpha=n, axis=axis).sum()
        return df.rolling(min_periods=n, axis=axis).sum()

    @staticmethod
    def rmean(df: pd.DataFrame, n: int, axis: int = 0):
        if n < 0:
            return df.expanding(min_periods=-n, axis=axis).mean()
        elif n > 0 and n < 1:
            return df.ewm(alpha=n, axis=axis).mean()
        return df.rolling(min_periods=n, axis=axis).mean()

    @staticmethod
    def corr(dfa: pd.DataFrame, dfb: pd.DataFrame, axis: int = 0):
        return dfa.corrwith(dfb, axis=axis)

    @staticmethod
    def rank(df: pd.DataFrame, ascending: bool = False, axis: int = 0):
        return df.rank(axis=axis, ascending=ascending)

    @staticmethod
    def group(df: pd.DataFrame, n: int, axis: int = 0):
        return df.apply(lambda x: pd.qcut(x, q=n, labels=False), axis=1) + 1

    @staticmethod
    def zscore(df: pd.DataFrame):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    @staticmethod
    def wscore(df: pd.DataFrame, weight: pd.DataFrame):
        return (df.sub(np.sum(weight * df, axis=1), axis=0)).div(df.std(axis=1), axis=0)

    @staticmethod
    def minmax(df: pd.DataFrame):
        return df.sub(df.min(axis=1), axis=0).div(
            df.max(axis=1) - df.min(axis=1), axis=0
        )

    @staticmethod
    def madoutlier(df: pd.DataFrame, dev: int, drop: bool = False):
        def apply_mad(df: pd.DataFrame) -> pd.DataFrame:
            median = df.median(axis=1)
            ad = df.sub(median, axis=0)
            mad = ad.abs().median(axis=1)
            thresh_down = median - dev * mad
            thresh_up = median + dev * mad
            if not drop:
                return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
            return df.where(
                df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
                other=np.nan,
                axis=0,
            ).where(~df.isna())

        if isinstance(df.index, pd.MultiIndex):
            return df.apply(lambda x: apply_mad(x.unstack("order_book_id")).unstack())
        else:
            return apply_mad(df)

    @staticmethod
    def stdoutlier(df: pd.DataFrame, dev: int, drop: bool = False):
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        thresh_down = mean - dev * std
        thresh_up = mean + dev * std
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0), other=np.nan, axis=0
        ).where(~df.isna())

    @staticmethod
    def iqroutlier(df: pd.DataFrame, dev: int, drop: bool = False):
        thresh_up = df.quantile(1 - dev / 2, axis=1)
        thresh_down = df.quantile(dev / 2, axis=1)
        if not drop:
            return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
        return df.where(
            df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0), other=np.nan, axis=0
        ).where(~df.isna())

    @staticmethod
    def fillna(
        df: pd.DataFrame,
        val: int | str = 0,
    ):
        return df.fillna(val)

    @staticmethod
    def weightify(df: pd.DataFrame):
        return df.div(df.sum(axis=1), axis=0)

    @staticmethod
    def diff(df: pd.DataFrame, n: int = 1, axis: int = 0, nofirst: bool = False):
        if nofirst:
            df = df.copy()
            first = df.iloc[0].copy()
            df = df.diff(n, axis=axis)
            df.iloc[0] = first
            return df
        return df.diff(n, axis=axis)

    @staticmethod
    def absolute(df: pd.DataFrame):
        return df.abs()

    @staticmethod
    def sum(df: pd.DataFrame, axis: int = 0):
        return df.sum(axis=axis)

    @staticmethod
    def cumsum(df: pd.DataFrame, axis: int = 0):
        return df.cumsum(axis=axis)

    @staticmethod
    def cumprod(df: pd.DataFrame, axis: int = 0):
        return df.cumprod(axis=axis)

    @staticmethod
    def log(df: pd.DataFrame):
        return np.log((df + 1e-6).sub(df.min(axis=1), axis=0))

    @staticmethod
    def sqrt(df: pd.DataFrame):
        return np.sqrt(df.sub(df.min(axis=1), axis=0))

    @staticmethod
    def mean(df: pd.DataFrame, axis: int = 0):
        return df.mean(axis=axis)


class Evaluator:
    """Research framework for studying asset characteristics (factors).

    This class provides:
      - Grouped portfolio construction (e.g., quantile groups) and HL (high-minus-low) factor returns.
      - Rolling time-series exposures of assets to the constructed HL factor.
      - Cross-sectional regressions and Fama-MacBeth estimations.
      - Time-series regressions, GRS test, alpha anomaly tests, and GMM-based pricing.

    Notes:
      - The class is initialized with a primary factor (the research target factor) and price data.
      - For single-factor studies, methods will default to using only the primary factor. If you conduct a
        multi-factor study, pass additional factors via the `other_factors` argument as needed.
    """

    def __init__(
        self, factor: pd.DataFrame, price: pd.DataFrame, logger: Optional[Logger] = None
    ):
        """Initialize the Evaluator.

        Aligns factor and price data by dates. If factor dates are missing in price, they are dropped.
        If price dates are missing in factor, factor values are forward-filled.

        Args:
            factor: DataFrame of factor exposures (index: dates; columns: assets).
            price: DataFrame of asset prices (index: dates; columns: assets).
            logger: Optional logging.Logger; if None, a default debug logger is created.

        Raises:
            ValueError: If inputs cannot be aligned properly.
        """
        self._logger = logger or setup_logger("FactorEvaluator", level="DEBUG")
        self._factor = factor.copy()
        self._price = price.copy()
        self._align_factor_price()
        # Shifted price to anchor "t+1" to avoid look-ahead bias when computing future returns.
        self._shifted = self._price.shift(-1)

        # Placeholders for analysis outputs
        self.group_returns: Optional[pd.DataFrame] = None
        self.sorted_factor_return: Optional[pd.Series] = None
        self.ts_exposure_beta: Optional[pd.DataFrame] = None
        self.ts_exposure_t: Optional[pd.DataFrame] = None
        self.ts_exposure_alpha: Optional[pd.DataFrame] = None
        self.ic: Optional[pd.Series] = None
        self.direction: Optional[float] = None
        self.topk_result: Optional[object] = None
        self.cs_betas: Optional[pd.DataFrame] = None
        self.cs_tstats: Optional[pd.DataFrame] = None
        self.cs_r2: Optional[pd.Series] = None
        self.fmb_premia: Optional[pd.Series] = None
        self.fmb_tstats: Optional[pd.Series] = None
        self.ts_alpha: Optional[pd.Series] = None
        self.ts_alpha_t: Optional[pd.Series] = None
        self.ts_beta: Optional[pd.DataFrame] = None
        self.ts_beta_t: Optional[pd.DataFrame] = None
        self.grs_stat: Optional[float] = None
        self.grs_pval: Optional[float] = None
        self.alpha_test_result: Optional[pd.DataFrame] = None
        self.gmm_result: Optional[Dict[str, Union[np.ndarray, float]]] = None

    # ===========================
    # Internal helpers
    # ===========================

    def _align_factor_price(self) -> None:
        """Align factor and price indices, with informative logging."""
        if (factor_only := self._factor.index.difference(self._price.index)).size:
            self._logger.warning(
                f"Index {factor_only} in factor without price; these dates will be dropped."
            )
            self._factor = self._factor.drop(index=factor_only)

        if (price_only := self._price.index.difference(self._factor.index)).size:
            self._logger.warning(
                f"Index {price_only} in price without factor; factors will be forward-filled to those dates."
            )
            self._factor = self._factor.reindex(index=self._price.index, method="ffill")

    @staticmethod
    def _default_feasible_like(df: pd.DataFrame) -> pd.DataFrame:
        """Create a boolean DataFrame of the same shape as df, filled with True."""
        return pd.DataFrame(True, index=df.index, columns=df.columns)

    @staticmethod
    def _default_weight_like(df: pd.DataFrame) -> pd.DataFrame:
        """Create a float DataFrame of the same shape as df, filled with ones."""
        return pd.DataFrame(1.0, index=df.index, columns=df.columns)

    def _future_return(self, horizon: int) -> pd.DataFrame:
        """Compute future returns using an anchored shift to avoid look-ahead bias."""
        return self._shifted.shift(-horizon) / self._shifted - 1

    @staticmethod
    def _qcut_groups(series: pd.Series, q: int) -> pd.Series:
        """Quantile-cut grouping with robust handling of edge cases."""
        s = series.dropna()
        if s.empty:
            return pd.Series(index=series.index, dtype="float")
        try:
            g = pd.qcut(s, q, labels=False, duplicates="drop") + 1
        except Exception:
            return pd.Series(index=series.index, dtype="float")
        g = g.astype("float")
        return g.reindex(series.index)

    @staticmethod
    def _group_return(r: pd.Series, w: pd.Series) -> float:
        """Compute weighted group return with robust reweighting if weights are invalid."""
        r = r.dropna()
        if r.empty:
            return np.nan
        w = w.reindex(r.index).fillna(0.0).clip(lower=0)
        if (w > 0).sum() == 0 or w.sum() <= 0:
            w = pd.Series(1.0, index=r.index)
        total = w.sum()
        if total <= 0:
            return np.nan
        w = w / total
        return float((w * r).sum())

    @staticmethod
    def _combine_bucket_group_returns(
        bucket_results: List[Dict[int, float]],
        bucket_sizes: List[int],
        n: int,
        cell_weight: Literal["equal", "count"],
    ) -> Dict[str, float]:
        """Combine group returns across buckets with either equal weights or bucket-count weights."""
        g_ret = {f"G{i}": np.nan for i in range(1, n + 1)}
        if len(bucket_results) == 0:
            return g_ret

        if cell_weight == "count" and np.sum(bucket_sizes) > 0:
            bw = np.array(bucket_sizes, dtype="float")
            bw = bw / bw.sum()
        else:
            bw = np.ones(len(bucket_results), dtype="float") / len(bucket_results)

        for gi in range(1, n + 1):
            vals = np.array([br[gi] for br in bucket_results], dtype="float")
            m = np.isfinite(vals)
            if m.sum() == 0:
                g_ret[f"G{gi}"] = np.nan
            else:
                ww = bw[m]
                ww = ww / ww.sum()
                g_ret[f"G{gi}"] = float(np.dot(ww, vals[m]))
        return g_ret

    @staticmethod
    def _add_intercept(X: np.ndarray) -> np.ndarray:
        """Add an intercept column to a design matrix."""
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _white_covariance(
        X: np.ndarray, resid: np.ndarray, hc_type: str = "HC1"
    ) -> np.ndarray:
        """Compute White heteroskedasticity-robust covariance."""
        XtX_inv = np.linalg.inv(X.T @ X)
        if hc_type.upper() == "HC1":
            scale = (
                X.shape[0] / (X.shape[0] - X.shape[1])
                if (X.shape[0] - X.shape[1]) > 0
                else 1.0
            )
        else:
            scale = 1.0
        S = np.diag(resid**2)
        meat = X.T @ S @ X
        return scale * XtX_inv @ meat @ XtX_inv

    @staticmethod
    def _newey_west_covariance(
        X: np.ndarray, resid: np.ndarray, lag: int = 3
    ) -> np.ndarray:
        """Compute Newey-West HAC covariance for time-series regression."""
        T, _ = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        U = resid[:, None] * X  # T x p
        S = U.T @ U
        for l in range(1, min(lag, T - 1) + 1):
            w_l = 1.0 - l / (lag + 1)
            Gamma = U[l:].T @ U[:-l]
            S += w_l * (Gamma + Gamma.T)
        return XtX_inv @ S @ XtX_inv

    @staticmethod
    def _ols_fit(
        X: np.ndarray,
        y: np.ndarray,
        add_intercept: bool = True,
        cov_type: Literal["none", "white", "nw"] = "none",
        hc_type: str = "HC1",
        nw_lag: int = 0,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Generic OLS fit with optional intercept and robust covariance estimators.

        Args:
            X: Design matrix (n x p).
            y: Response vector (n,).
            add_intercept: Whether to add an intercept column.
            cov_type: Covariance estimator: 'none', 'white', or 'nw'.
            hc_type: White estimator type, 'HC0' or 'HC1'.
            nw_lag: Newey-West lag (if cov_type='nw').
            weights: Optional observation weights (n,). If provided, uses sqrt-weight transformation.

        Returns:
            Tuple:
              - beta: Coefficient vector (p' , p' = p + 1 if add_intercept).
              - se: Standard errors (p',).
              - t: t-statistics (p',).
              - cov: Covariance matrix (p' x p').
              - r2: R-squared.

        Notes:
            - If weights are provided, performs WLS via sqrt-weight transformation.
            - Newey-West is intended for time-series regressions; X must be ordered by time.
        """
        Xw = X.copy()
        yw = y.copy()
        if add_intercept:
            Xw = Evaluator._add_intercept(Xw)
        if weights is not None:
            sw = np.sqrt(np.asarray(weights).reshape(-1))
            Xw = Xw * sw[:, None]
            yw = yw * sw

        n, p = Xw.shape
        valid = np.isfinite(yw) & np.all(np.isfinite(Xw), axis=1)
        Xw = Xw[valid]
        yw = yw[valid]
        if Xw.shape[0] <= p:
            beta = np.full(p, np.nan)
            return (
                beta,
                np.full(p, np.nan),
                np.full(p, np.nan),
                np.full((p, p), np.nan),
                np.nan,
            )

        XtX = Xw.T @ Xw
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            beta = np.full(p, np.nan)
            return (
                beta,
                np.full(p, np.nan),
                np.full(p, np.nan),
                np.full((p, p), np.nan),
                np.nan,
            )

        beta = XtX_inv @ (Xw.T @ yw)
        resid = yw - Xw @ beta
        sse = float(resid.T @ resid)
        sst = float(((yw - yw.mean()) ** 2).sum())
        r2 = 1.0 - sse / sst if sst > 0 else np.nan

        if cov_type == "none":
            dof = Xw.shape[0] - p
            sigma2 = sse / dof if dof > 0 else np.nan
            cov = sigma2 * XtX_inv
        elif cov_type == "white":
            cov = Evaluator._white_covariance(Xw, resid, hc_type=hc_type)
        elif cov_type == "nw":
            cov = Evaluator._newey_west_covariance(Xw, resid, lag=nw_lag)
        else:
            cov = np.full((p, p), np.nan)

        se = np.sqrt(np.diag(cov))
        t = np.array([beta[i] / se[i] if se[i] > 0 else np.nan for i in range(p)])
        return beta, se, t, cov, r2

    @staticmethod
    def _rolling_single_regression(
        y: np.ndarray,
        x: np.ndarray,
        dates: pd.Index,
        window: int,
        min_obs: int,
        intercept: bool,
    ) -> Tuple[pd.Series, Optional[pd.Series], pd.Series]:
        """Run rolling OLS for a single regressor with optional intercept."""
        betas = np.full(len(dates), np.nan, dtype="float")
        alphas = np.full(len(dates), np.nan, dtype="float") if intercept else None
        tstats = np.full(len(dates), np.nan, dtype="float")

        for j in range(len(dates)):
            start = max(0, j - window + 1)
            y_win = y[start : j + 1]
            x_win = x[start : j + 1]
            valid = np.isfinite(y_win) & np.isfinite(x_win)
            if valid.sum() < min_obs:
                continue
            X_win = x_win.reshape(-1, 1)
            beta_vec, _, t_vec, _, _ = Evaluator._ols_fit(
                X=X_win, y=y_win, add_intercept=intercept, cov_type="none"
            )
            if intercept:
                alphas[j] = float(beta_vec[0])
                betas[j] = float(beta_vec[1])
                tstats[j] = float(t_vec[1])
            else:
                betas[j] = float(beta_vec[0])
                tstats[j] = float(t_vec[0])

        beta_s = pd.Series(betas, index=dates)
        alpha_s = pd.Series(alphas, index=dates) if intercept else None
        t_s = pd.Series(tstats, index=dates)
        return beta_s, alpha_s, t_s

    # ===========================
    # Grouping and HL factor
    # ===========================

    def get_group_returns(
        self,
        n: int = 10,
        horizon: int = 1,
        other_factors: Optional[
            Union[List[pd.DataFrame], Tuple[pd.DataFrame, ...]]
        ] = None,
        mode: Literal["conditional", "independent"] = "conditional",
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
        cell_weight: Literal["equal", "count"] = "equal",
        hl_mode: Literal["extreme", "first_last"] = "first_last",
    ):
        """Compute grouped portfolio returns and HL (high-minus-low) factor returns.

        Buckets assets into n quantile groups each date based on the primary factor (self._factor).
        Supports controlling for other factors using independent or conditional bucketing.

        Independent mode:
        - Construct global quantiles for the main factor and each control factor.
        - Compute returns for every cell (cross of all control-factor buckets and main-factor bucket).
        - HL is computed as:
            sum_j R[top_main_group, j] - sum_j R[bottom_main_group, j]
            where j indexes all control-factor bucket combinations.

        Conditional mode:
        - Hierarchical sorting by control factors, then local quantiles for the main factor inside each bucket.
        - HL is computed as:
            mean_j R[top_main_group in bucket j] - mean_j R[bottom_main_group in bucket j]
            with equal-weight averaging across buckets.

        Single-factor case:
        - Standard HL: 'extreme' (max minus min group) or 'first_last' (Gn - G1).

        Args:
            n: Number of quantile groups.
            horizon: Return horizon (in periods).
            other_factors: Optional list/tuple of control factor DataFrames aligned with the main factor.
            mode: Bucketing mode: 'conditional' or 'independent'.
            feasible: Optional boolean DataFrame for asset eligibility.
            weight: Optional DataFrame of within-group weights.
            cell_weight: Aggregation across buckets for reporting group_returns: 'equal' or 'count'.
            hl_mode: HL derivation for single-factor case: 'extreme' or 'first_last'.

        Returns:
            Evaluator: self with attributes:
                - group_returns: DataFrame of group returns per date, columns G1..Gn.
                - sorted_factor_return: Series of HL factor return (per definitions above).

        Raises:
            ValueError: If mode is not recognized.
            TypeError: If other_factors is not None/list/tuple.
        """
        mode = str(mode).lower()
        if mode not in ("independent", "conditional"):
            raise ValueError("mode must be 'independent' or 'conditional'")

        future = self._future_return(horizon)

        feasible = (
            feasible.reindex_like(future).fillna(False)
            if feasible is not None
            else self._default_feasible_like(future)
        )
        weight = (
            weight.reindex_like(future)
            if weight is not None
            else self._default_weight_like(future)
        )

        if other_factors is None:
            other_list: List[pd.DataFrame] = []
        elif isinstance(other_factors, (list, tuple)):
            other_list = [f.reindex_like(self._factor) for f in other_factors]
        else:
            raise TypeError("other_factors must be None, list, or tuple")

        dates = future.index
        group_cols = [f"G{i}" for i in range(1, n + 1)]
        rows: List[Dict[str, float]] = []
        hl_values: List[float] = []

        for dt in dates:
            try:
                eligible = (
                    feasible.loc[dt].astype(bool)
                    & future.loc[dt].notna()
                    & self._factor.loc[dt].notna()
                )
                r_t = future.loc[dt]
                w_t = weight.loc[dt]
                f_t = self._factor.loc[dt]

                g_ret = {c: np.nan for c in group_cols}
                if eligible.sum() == 0:
                    rows.append(g_ret)
                    hl_values.append(np.nan)
                    continue

                # Single-factor case
                if len(other_list) == 0:
                    g_t = self._qcut_groups(f_t.where(eligible), n)
                    if g_t.notna().sum() == 0:
                        rows.append(g_ret)
                        hl_values.append(np.nan)
                        continue
                    uniq = sorted(pd.unique(g_t.dropna().astype(int)))
                    for gi in uniq:
                        mask = g_t == gi
                        ret_g = self._group_return(
                            r_t.where(mask & eligible), w_t.where(mask & eligible)
                        )
                        g_ret[f"G{gi}"] = ret_g

                    # HL for single-factor case
                    if hl_mode == "extreme":
                        max_v = np.nanmax([g_ret[c] for c in group_cols])
                        min_v = np.nanmin([g_ret[c] for c in group_cols])
                        hl_dt = (
                            max_v - min_v
                            if np.isfinite(max_v) and np.isfinite(min_v)
                            else np.nan
                        )
                    else:
                        hl_dt = (
                            (g_ret[f"G{n}"] - g_ret["G1"])
                            if np.isfinite(g_ret.get(f"G{n}", np.nan))
                            and np.isfinite(g_ret.get("G1", np.nan))
                            else np.nan
                        )

                    rows.append(g_ret)
                    hl_values.append(hl_dt)
                    continue

                # Multi-factor: Independent bucketing (global groups)
                if mode == "independent":
                    # Control-factor global groups
                    other_groups = [
                        self._qcut_groups(of.loc[dt].where(eligible), n)
                        for of in other_list
                    ]
                    g_t_global = self._qcut_groups(f_t.where(eligible), n)

                    keys_df = pd.concat(other_groups, axis=1)
                    keys_df.columns = (
                        [f"g{i}" for i in range(keys_df.shape[1])]
                        if keys_df.shape[1] > 1
                        else ["g0"]
                    )

                    valid = eligible & keys_df.notna().all(axis=1) & g_t_global.notna()
                    if valid.sum() == 0:
                        rows.append(g_ret)
                        hl_values.append(np.nan)
                        continue

                    tmp = keys_df[valid].copy()
                    tmp["target_g"] = g_t_global[valid]
                    tmp["ret"] = r_t[valid]
                    tmp["w"] = w_t[valid]

                    bucket_results: List[Dict[int, float]] = []
                    bucket_sizes: List[int] = []

                    # Compute returns for each cell (each control-bucket combination)
                    for _, sub_idx in tmp.groupby(list(keys_df.columns)).groups.items():
                        sub = tmp.loc[sub_idx]
                        sub_group_ret = {i: np.nan for i in range(1, n + 1)}
                        for gi in range(1, n + 1):
                            sel = sub["target_g"] == gi
                            if sel.sum() == 0:
                                continue
                            sub_group_ret[gi] = self._group_return(
                                sub.loc[sel, "ret"], sub.loc[sel, "w"]
                            )
                        bucket_results.append(sub_group_ret)
                        bucket_sizes.append(int(sub.shape[0]))

                    # Aggregate group returns for reporting (equal/count)
                    g_ret = self._combine_bucket_group_returns(
                        bucket_results, bucket_sizes, n, cell_weight
                    )

                    # HL per your definition: sum over all cells of top group minus sum over all cells of bottom group
                    top_vals = np.array(
                        [br.get(n, np.nan) for br in bucket_results], dtype="float"
                    )
                    bot_vals = np.array(
                        [br.get(1, np.nan) for br in bucket_results], dtype="float"
                    )
                    sum_top = (
                        np.nansum(top_vals) if np.isfinite(top_vals).any() else np.nan
                    )
                    sum_bot = (
                        np.nansum(bot_vals) if np.isfinite(bot_vals).any() else np.nan
                    )
                    hl_dt = (
                        (sum_top - sum_bot)
                        if np.isfinite(sum_top) and np.isfinite(sum_bot)
                        else np.nan
                    )

                    rows.append(g_ret)
                    hl_values.append(hl_dt)
                    continue

                # Multi-factor: Conditional (hierarchical on controls, local sorting of main factor)
                buckets = [pd.Index(f_t.index[eligible])]
                for of in other_list:
                    new_buckets: List[pd.Index] = []
                    s = of.loc[dt]
                    for idxs in buckets:
                        if len(idxs) == 0:
                            continue
                        g = self._qcut_groups(s.loc[idxs], n)
                        if g.notna().sum() == 0:
                            new_buckets.append(idxs)
                            continue
                        for label in sorted(g.dropna().unique()):
                            sub_idx = g[g == label].index
                            if len(sub_idx) > 0:
                                new_buckets.append(pd.Index(sub_idx))
                    buckets = new_buckets if len(new_buckets) > 0 else buckets

                bucket_results: List[Dict[int, float]] = []
                bucket_sizes: List[int] = []

                for idxs in buckets:
                    sub_idx = pd.Index(idxs).intersection(f_t.index[eligible])
                    if len(sub_idx) == 0:
                        continue
                    g_local = self._qcut_groups(f_t.loc[sub_idx], n)
                    if g_local.notna().sum() == 0:
                        continue

                    sub_group_ret = {i: np.nan for i in range(1, n + 1)}
                    uniq = sorted(pd.unique(g_local.dropna().astype(int)))
                    for gi in uniq:
                        top_mask = g_local == gi
                        ret_g = self._group_return(
                            r_t.loc[sub_idx].where(top_mask),
                            w_t.loc[sub_idx].where(top_mask),
                        )
                        sub_group_ret[gi] = ret_g

                    bucket_results.append(sub_group_ret)
                    bucket_sizes.append(int(len(sub_idx)))

                # Aggregate group returns for reporting (equal/count)
                g_ret = self._combine_bucket_group_returns(
                    bucket_results, bucket_sizes, n, cell_weight
                )

                # HL per your definition: equal-weight average across buckets of top minus bottom
                top_vals = np.array(
                    [br.get(n, np.nan) for br in bucket_results], dtype="float"
                )
                bot_vals = np.array(
                    [br.get(1, np.nan) for br in bucket_results], dtype="float"
                )
                mean_top = (
                    np.nanmean(top_vals) if np.isfinite(top_vals).any() else np.nan
                )
                mean_bot = (
                    np.nanmean(bot_vals) if np.isfinite(bot_vals).any() else np.nan
                )
                hl_dt = (
                    (mean_top - mean_bot)
                    if np.isfinite(mean_top) and np.isfinite(mean_bot)
                    else np.nan
                )

                rows.append(g_ret)
                hl_values.append(hl_dt)

            except Exception as e:
                self._logger.error(f"get_group_returns error on {dt}: {e}")
                rows.append({c: np.nan for c in group_cols})
                hl_values.append(np.nan)

        self.group_returns = pd.DataFrame(rows, index=dates, columns=group_cols)
        self.sorted_factor_return = pd.Series(
            hl_values, index=dates, name=f"HL_n{n}_h{horizon}_{mode}"
        )
        self._logger.info(
            f"Group returns computed: n={n}, horizon={horizon}, mode={mode}"
        )
        return self

    def get_factor_exposure(
        self,
        n: int = 10,
        horizon: int = 1,
        other_factors: Optional[
            Union[List[pd.DataFrame], Tuple[pd.DataFrame, ...]]
        ] = None,
        mode: Literal["conditional", "independent"] = "conditional",
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
        cell_weight: Literal["equal", "count"] = "equal",
        window: int = 252,
        min_obs: int = 60,
        intercept: bool = True,
        standardize_factor: bool = False,
        n_jobs: int = -1,
        hl_mode: Literal["extreme", "first_last"] = "first_last",
    ):
        """Rolling time-series regression: asset returns vs. constructed HL factor.

        Steps:
          1) Build HL factor via grouped portfolios (get_group_returns).
          2) Perform rolling OLS of each asset's future returns on the HL factor.

        Args:
            n: Number of groups for HL construction.
            horizon: Return horizon for dependent variable.
            other_factors: Optional list/tuple of control factor DataFrames for grouping (multi-factor study).
            mode: Grouping mode used in HL construction: 'conditional' or 'independent'.
            feasible: Optional eligibility DataFrame for grouping.
            weight: Optional weights for group construction.
            cell_weight: Bucket aggregation weight: 'equal' or 'count'.
            window: Rolling window size for time-series regression.
            min_obs: Minimum valid observations in a window to compute regression.
            intercept: Whether to include intercept.
            standardize_factor: Whether to standardize the HL factor prior to regression.
            n_jobs: Number of parallel jobs for asset-wise regression; -1 uses all CPUs.
            hl_mode: HL derivation mode: 'extreme' or 'first_last'.

        Returns:
            Evaluator: self with attributes:
                - ts_exposure_beta: DataFrame of rolling betas.
                - ts_exposure_t: DataFrame of rolling t-statistics of beta.
                - ts_exposure_alpha: DataFrame of rolling alphas (if intercept=True).
        """
        if self.group_returns is None or self.sorted_factor_return is None:
            self.get_group_returns(
                n=n,
                horizon=horizon,
                other_factors=other_factors,
                mode=mode,
                feasible=feasible,
                weight=weight,
                cell_weight=cell_weight,
                hl_mode=hl_mode,
            )

        f_ret = self.sorted_factor_return.copy()
        future = self._future_return(horizon)
        common_idx = f_ret.index.intersection(future.index)
        f_ret = f_ret.loc[common_idx]
        y_df = future.loc[common_idx].copy()

        if feasible is not None:
            feasible = feasible.reindex(index=common_idx, columns=y_df.columns)
            y_df = y_df.where(feasible.astype(bool))

        if standardize_factor:
            std_val = f_ret.std(ddof=1)
            f_std = (f_ret - f_ret.mean()) / (std_val if std_val > 0 else 1.0)
            x_series = f_std
        else:
            x_series = f_ret

        dates = common_idx
        assets = list(y_df.columns)

        def _asset_rolling(col: str):
            y = y_df[col].values
            x = x_series.values
            beta_s, alpha_s, t_s = self._rolling_single_regression(
                y=y,
                x=x,
                dates=dates,
                window=window,
                min_obs=min_obs,
                intercept=intercept,
            )
            return (
                beta_s.rename(col),
                (alpha_s.rename(col) if intercept else None),
                t_s.rename(col),
            )

        try:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_asset_rolling)(col) for col in assets
            )
        except Exception as e:
            self._logger.error(f"get_factor_exposure parallel error: {e}")
            results = [_asset_rolling(col) for col in assets]

        beta_df = pd.concat([r[0] for r in results], axis=1)
        t_df = pd.concat([r[2] for r in results], axis=1)
        alpha_df = pd.concat([r[1] for r in results], axis=1) if intercept else None

        self.ts_exposure_beta = beta_df
        self.ts_exposure_t = t_df
        self.ts_exposure_alpha = alpha_df
        self._logger.info(
            f"TS exposure evaluated: window={window}, min_obs={min_obs}, intercept={intercept}, "
            f"n={n}, horizon={horizon}, mode={mode}, hl_mode={hl_mode}"
        )
        return self

    def get_info_coef(self, freq: int = 1, method: str = "spearman"):
        """Evaluate Information Coefficient (IC) between factor exposures and future returns.

        Args:
            freq: Return horizon in periods.
            method: Correlation method ('pearson', 'spearman', 'kendall').

        Returns:
            Evaluator: self with attributes:
                - ic: Series of daily IC values.
                - direction: Sign of mean IC to be used for portfolio direction.
        """
        future = self._future_return(freq)
        self.ic = self._factor.corrwith(future, axis=1, method=method)
        self.direction = np.sign(self.ic.mean())
        return self

    # ===========================
    # Cross-sectional analytics
    # ===========================

    def orthogonalize_factor(
        self,
        target: pd.DataFrame,
        other_factors: List[pd.DataFrame],
        mode: Literal["cross_sectional", "time_series"] = "cross_sectional",
        intercept: bool = True,
    ) -> pd.DataFrame:
        """Orthogonalize a factor with respect to other factors.

        Performs regression of the target factor on other_factors and returns residuals.
        Supports cross-sectional (per date across assets) or time-series (per asset across time).

        Args:
            target: DataFrame of the target factor to be orthogonalized.
            other_factors: List of control factor DataFrames aligned with the target.
            mode: 'cross_sectional' or 'time_series'.
            intercept: Whether to include intercept in the orthogonalization regression.

        Returns:
            DataFrame: Residual factor exposures orthogonal to other_factors.

        Raises:
            ValueError: If other_factors are not provided or alignment fails.
        """
        if other_factors is None or len(other_factors) == 0:
            raise ValueError("other_factors must be provided for orthogonalization")

        aligned = [target] + list(other_factors)
        idx = aligned[0].index
        cols = aligned[0].columns
        aligned = [a.reindex(index=idx, columns=cols) for a in aligned]
        target_aligned = aligned[0]
        controls = aligned[1:]

        res_df = pd.DataFrame(index=idx, columns=cols, dtype="float")

        if mode == "cross_sectional":
            for dt in idx:
                y = target_aligned.loc[dt].values
                X = np.column_stack([c.loc[dt].values for c in controls])
                mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                min_req = X.shape[1] + (1 if intercept else 0) + 1
                if mask.sum() < min_req:
                    res_df.loc[dt] = np.nan
                    continue
                beta, _, _, _, _ = self._ols_fit(
                    X[mask], y[mask], add_intercept=intercept, cov_type="none"
                )
                Xw = self._add_intercept(X) if intercept else X
                fitted = Xw @ beta
                r = np.full_like(y, np.nan, dtype="float")
                r[mask] = y[mask] - fitted[mask]
                res_df.loc[dt] = r
        else:
            for asset in cols:
                y = target_aligned[asset].values
                X = np.column_stack([c[asset].values for c in controls])
                mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
                min_req = X.shape[1] + (1 if intercept else 0) + 1
                if mask.sum() < min_req:
                    res_df[asset] = np.nan
                    continue
                beta, _, _, _, _ = self._ols_fit(
                    X[mask], y[mask], add_intercept=intercept, cov_type="none"
                )
                Xw = self._add_intercept(X) if intercept else X
                fitted = Xw @ beta
                r = np.full_like(y, np.nan, dtype="float")
                r[mask] = y[mask] - fitted[mask]
                res_df[asset] = r

        return res_df

    def cross_sectional_regression(
        self,
        horizon: int = 1,
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
        add_intercept: bool = True,
        cov_type: Literal["none", "white"] = "white",
        white_type: str = "HC1",
        orthogonalize: bool = False,
        other_factors: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Run cross-sectional regressions of future returns on characteristics (factors).

        For single-factor study, this uses the primary factor only.
        For multi-factor study, supply `other_factors` to include additional regressors alongside the primary factor.

        Args:
            horizon: Return horizon in periods.
            feasible: Optional boolean eligibility mask DataFrame.
            weight: Optional cross-sectional weights for assets at each date.
            add_intercept: Whether to include intercept in the cross-sectional regression.
            cov_type: Covariance estimator: 'none' or 'white'.
            white_type: White estimator type ('HC0' or 'HC1').
            orthogonalize: Whether to orthogonalize regressors with respect to `other_factors`.
            other_factors: Dict of name -> DataFrame for additional regressors (multi-factor case).

        Returns:
            Evaluator: self with attributes:
                - cs_betas: DataFrame (dates x factors) of cross-sectional betas.
                - cs_tstats: DataFrame (dates x factors) of t-stats.
                - cs_r2: Series of cross-sectional R-squared per date.
        """
        future = self._future_return(horizon)
        idx, cols = future.index, future.columns

        # Build regressor dictionary: always include the primary factor
        X_dict: Dict[str, pd.DataFrame] = {
            "factor": self._factor.reindex(index=idx, columns=cols)
        }
        if other_factors is not None and len(other_factors) > 0:
            for k, df in other_factors.items():
                X_dict[k] = df.reindex(index=idx, columns=cols)

        feasible = (
            feasible.reindex(index=idx, columns=cols).fillna(False)
            if feasible is not None
            else self._default_feasible_like(future)
        )
        if weight is not None:
            weight = weight.reindex(index=idx, columns=cols).fillna(0.0)

        # Optional orthogonalization (residualizing each regressor against supplied other_factors)
        if orthogonalize and other_factors is not None and len(other_factors) > 0:
            Z_list = [
                df.reindex(index=idx, columns=cols) for df in other_factors.values()
            ]
            for k in list(X_dict.keys()):
                try:
                    X_dict[k] = self.orthogonalize_factor(
                        target=X_dict[k], other_factors=Z_list, mode="cross_sectional"
                    )
                except Exception as e:
                    self._logger.error(f"Orthogonalization failed for factor {k}: {e}")

        factor_names = list(X_dict.keys())
        dates = idx
        betas_rows, t_rows, r2_vals = [], [], []

        for dt in dates:
            y = future.loc[dt].values
            Xi = np.column_stack([X_dict[k].loc[dt].values for k in factor_names])

            valid = (
                feasible.loc[dt].astype(bool).values
                & np.isfinite(y)
                & np.all(np.isfinite(Xi), axis=1)
            )
            min_req = Xi.shape[1] + (1 if add_intercept else 0) + 1
            if valid.sum() < min_req:
                betas_rows.append({k: np.nan for k in factor_names})
                t_rows.append({k: np.nan for k in factor_names})
                r2_vals.append(np.nan)
                continue

            w = None
            if weight is not None:
                w = weight.loc[dt].values
                w = np.where(valid, w, 0.0)
                if (w > 0).sum() == 0:
                    w = None

            beta, se, t, _, r2 = self._ols_fit(
                X=Xi[valid],
                y=y[valid],
                add_intercept=add_intercept,
                cov_type=cov_type,
                hc_type=white_type,
                weights=w[valid] if w is not None else None,
            )
            # Map coefficients: intercept is first (if present)
            start_idx = 1 if add_intercept else 0
            betas_rows.append(
                {
                    factor_names[i]: float(beta[start_idx + i])
                    for i in range(len(factor_names))
                }
            )
            t_rows.append(
                {
                    factor_names[i]: float(t[start_idx + i])
                    for i in range(len(factor_names))
                }
            )
            r2_vals.append(r2)

        self.cs_betas = pd.DataFrame(betas_rows, index=dates, columns=factor_names)
        self.cs_tstats = pd.DataFrame(t_rows, index=dates, columns=factor_names)
        self.cs_r2 = pd.Series(r2_vals, index=dates, name="R2_CS")
        self._logger.info("Cross-sectional regression completed.")
        return self

    def fama_macbeth(
        self,
        horizon: int = 1,
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
        add_intercept: bool = True,
        nw_lag: int = 3,
        orthogonalize: bool = False,
        other_factors: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Perform Fama-MacBeth regression to estimate factor risk premia.

        Step 1: For each date, run cross-sectional regression of future returns on exposures.
        Step 2: Average the coefficients over time; compute Newey-West HAC t-stats across time.

        Args:
            horizon: Return horizon in periods.
            feasible: Optional eligibility mask DataFrame.
            weight: Optional cross-sectional weights at each date.
            add_intercept: Whether to include intercept in cross-sectional regressions.
            nw_lag: Newey-West lag for time-series of coefficients.
            orthogonalize: Whether to orthogonalize factors in step 1 w.r.t. `other_factors`.
            other_factors: Dict of name -> DataFrame for additional regressors (multi-factor case).

        Returns:
            Evaluator: self with attributes:
                - fmb_premia: Series of average factor premia across time.
                - fmb_tstats: Series of Newey-West HAC t-stats for premia.
        """
        self.cross_sectional_regression(
            horizon=horizon,
            feasible=feasible,
            weight=weight,
            add_intercept=add_intercept,
            cov_type="white",
            white_type="HC1",
            orthogonalize=orthogonalize,
            other_factors=other_factors,
        )
        beta_ts = self.cs_betas  # dates x factors
        premia = beta_ts.mean(axis=0)

        # HAC t-stats for time-series of coefficients (mean-only regression)
        tstats = {}
        for k in beta_ts.columns:
            x = np.ones((beta_ts[k].shape[0], 1))
            y = beta_ts[k].values
            beta_vec, _, t_vec, _, _ = self._ols_fit(
                X=x, y=y, add_intercept=False, cov_type="nw", nw_lag=nw_lag
            )
            tstats[k] = float(t_vec[0])

        self.fmb_premia = premia.rename("FMB_Premia")
        self.fmb_tstats = pd.Series(tstats, name="FMB_t")
        self._logger.info(
            "Fama-MacBeth regression completed with Newey-West HAC t-stats."
        )
        return self

    # ===========================
    # Time-series regressions and tests
    # ===========================

    def ts_regression(
        self,
        asset_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        add_intercept: bool = True,
        cov_type: Literal["none", "white", "nw"] = "nw",
        nw_lag: int = 3,
        hc_type: str = "HC1",
    ):
        """Run time-series regressions of asset returns on factor returns.

        Computes alphas, betas, t-stats, and supports robust covariance via White or Newey-West.

        Args:
            asset_returns: DataFrame of asset returns (dates x assets). Defaults to next-period returns from prices.
            factor_returns: DataFrame of factor returns (dates x K). Defaults to constructed HL factor if available.
            add_intercept: Whether to include intercept (alpha).
            cov_type: Covariance type: 'none', 'white', or 'nw'.
            nw_lag: Newey-West lag for HAC (used if cov_type='nw').
            hc_type: White estimator type ('HC0' or 'HC1').

        Returns:
            Evaluator: self with attributes:
                - ts_alpha: Series of alphas per asset.
                - ts_beta: DataFrame of betas (assets x factors).
                - ts_alpha_t: Series of t-stats for alphas.
                - ts_beta_t: DataFrame of t-stats for betas.
        """
        if asset_returns is None:
            asset_returns = self._price.pct_change(fill_method=None).shift(-1)
        if factor_returns is None:
            if (
                hasattr(self, "sorted_factor_return")
                and self.sorted_factor_return is not None
            ):
                factor_returns = self.sorted_factor_return.to_frame("HL")
            else:
                raise ValueError(
                    "factor_returns must be provided if HL factor is not computed."
                )

        idx = asset_returns.index.intersection(factor_returns.index)
        asset_returns = asset_returns.loc[idx]
        factor_returns = factor_returns.loc[idx]
        X = factor_returns.values  # T x K
        assets = list(asset_returns.columns)

        alpha_vals: Dict[str, float] = {}
        alpha_t: Dict[str, float] = {}
        beta_vals: Dict[str, np.ndarray] = {}
        beta_t: Dict[str, np.ndarray] = {}

        for asset in assets:
            y = asset_returns[asset].values
            beta, _, t, _, _ = self._ols_fit(
                X=X,
                y=y,
                add_intercept=add_intercept,
                cov_type=cov_type,
                hc_type=hc_type,
                nw_lag=nw_lag,
            )
            if add_intercept:
                alpha_vals[asset] = float(beta[0])
                alpha_t[asset] = float(t[0])
                beta_vals[asset] = beta[1:].astype(float)
                beta_t[asset] = t[1:].astype(float)
            else:
                alpha_vals[asset] = np.nan
                alpha_t[asset] = np.nan
                beta_vals[asset] = beta.astype(float)
                beta_t[asset] = t.astype(float)

        self.ts_alpha = pd.Series(alpha_vals, name="alpha")
        self.ts_alpha_t = pd.Series(alpha_t, name="alpha_t")
        self.ts_beta = pd.DataFrame(beta_vals, index=factor_returns.columns).T
        self.ts_beta_t = pd.DataFrame(beta_t, index=factor_returns.columns).T
        self._logger.info("Time-series regression completed.")
        return self

    def grs_test(
        self,
        asset_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        add_intercept: bool = True,
    ) -> Tuple[float, float]:
        """Compute the Gibbons-Ross-Shanken (GRS) test for joint alpha = 0.

        Args:
            asset_returns: DataFrame of asset returns (dates x N). If None, uses next-period returns from prices.
            factor_returns: DataFrame of factor returns (dates x K). Defaults to constructed HL factor if available.
            add_intercept: Whether time-series regressions include intercepts (required for GRS).

        Returns:
            Tuple of (GRS F-statistic, p-value).
        """
        if asset_returns is None:
            asset_returns = self._price.pct_change(fill_method=None).shift(-1)
        if factor_returns is None:
            if (
                hasattr(self, "sorted_factor_return")
                and self.sorted_factor_return is not None
            ):
                factor_returns = self.sorted_factor_return.to_frame("HL")
            else:
                raise ValueError(
                    "factor_returns must be provided if HL factor is not computed."
                )

        idx = asset_returns.index.intersection(factor_returns.index)
        R = asset_returns.loc[idx].values  # T x N
        F = factor_returns.loc[idx].values  # T x K
        Tn, N = R.shape
        K = F.shape[1]

        # TS regression for each asset: R_i = alpha_i + beta_i' F + eps_i
        alphas, residuals = [], []
        for i in range(N):
            y = R[:, i]
            beta, _, _, _, _ = self._ols_fit(
                X=F, y=y, add_intercept=True, cov_type="none"
            )
            alphas.append(beta[0])
            resid = y - (self._add_intercept(F) @ beta)
            residuals.append(resid)

        a = np.array(alphas).reshape(-1, 1)  # N x 1
        Eps = np.column_stack(residuals)  # T x N
        Sigma_e = np.cov(Eps, rowvar=False, ddof=1)  # N x N
        mu_F = F.mean(axis=0).reshape(-1, 1)  # K x 1
        Sigma_F = np.cov(F, rowvar=False, ddof=1)  # K x K

        try:
            Sigma_e_inv = np.linalg.inv(Sigma_e)
            Sigma_F_inv = np.linalg.inv(Sigma_F)
        except np.linalg.LinAlgError:
            self._logger.error("GRS test failed: singular covariance.")
            return np.nan, np.nan

        top = (Tn - N - K) / N if (Tn - N - K) > 0 else Tn / N
        denom = 1.0 + float(mu_F.T @ Sigma_F_inv @ mu_F)
        grs = float(top * (a.T @ Sigma_e_inv @ a) / denom)

        # p-value from F-distribution with df1=N, df2=T-N-K
        df1 = N
        df2 = Tn - N - K if (Tn - N - K) > 0 else max(Tn - K - 1, 1)
        try:
            from scipy import stats

            p_val = float(1.0 - stats.f.cdf(grs, df1=df1, df2=df2))
        except Exception:
            p_val = np.nan

        self.grs_stat = grs
        self.grs_pval = p_val
        self._logger.info("GRS test computed.")
        return grs, p_val

    def alpha_tests(
        self,
        asset_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        cov_type: Literal["none", "white", "nw"] = "nw",
        nw_lag: int = 3,
        hc_type: str = "HC1",
    ) -> pd.DataFrame:
        """Conduct individual alpha anomaly tests for assets.

        Runs time-series regressions with intercept and returns alpha estimates and t-stats.

        Args:
            asset_returns: DataFrame of asset returns (dates x N).
            factor_returns: DataFrame of factor returns (dates x K).
            cov_type: Covariance type: 'none', 'white', or 'nw'.
            nw_lag: Newey-West lag if cov_type='nw'.
            hc_type: White estimator type ('HC0' or 'HC1').

        Returns:
            DataFrame: Columns ['alpha', 't'] indexed by asset names.
        """
        self.ts_regression(
            asset_returns=asset_returns,
            factor_returns=factor_returns,
            add_intercept=True,
            cov_type=cov_type,
            nw_lag=nw_lag,
            hc_type=hc_type,
        )
        res = pd.DataFrame({"alpha": self.ts_alpha, "t": self.ts_alpha_t})
        self.alpha_test_result = res
        self._logger.info("Alpha tests completed.")
        return res

    def gmm_linear_pricing(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        two_step: bool = True,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Estimate linear factor risk premia via GMM under SDF m_t = 1 - lambda' F_t.

        Moment conditions: E[m_t R_t] = 0 => E[(1 - lambda' F_t) R_t] = 0.
        Solve for lambda minimizing g(lambda)' W g(lambda), where
        g(lambda) = mean_t[(1 - lambda' F_t) R_t], W is a weighting matrix.

        Args:
            asset_returns: DataFrame of asset returns (dates x N).
            factor_returns: DataFrame of factor returns (dates x K).
            two_step: Whether to run two-step GMM (second step uses an estimated optimal weighting).

        Returns:
            Dict with keys:
                - 'lambda': Estimated risk premia (K,).
                - 'J': J-statistic for overidentifying restrictions.
                - 'pval': p-value of J (chi-square with df=N-K).
                - 'W': Weighting matrix used (N x N).
        """
        R = asset_returns.values  # T x N
        F = factor_returns.values  # T x K
        Tn, N = R.shape
        K = F.shape[1]

        # Helper for moments at lambda
        def g_lambda(lmbd: np.ndarray) -> np.ndarray:
            mt = 1.0 - F @ lmbd  # T x 1
            return (R * mt[:, None]).mean(axis=0)  # N,

        # First-step: W = I, linearization using E[R F']
        # E[R F'] ~ mean over T of outer products; we build A (N x K) and b (N x 1)
        A = np.einsum("ti,tk->ik", R, F) / Tn  # N x K
        b = R.mean(axis=0).reshape(N, 1)  # N x 1
        try:
            lambda_1 = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)  # K,
        except Exception:
            lambda_1 = np.zeros(K)

        # Two-step GMM with estimated optimal weighting
        if two_step:
            mt = 1.0 - F @ lambda_1
            moments_t = R * mt[:, None]  # T x N
            S = np.cov(moments_t, rowvar=False, ddof=1)  # N x N
            try:
                W = np.linalg.pinv(S)
            except Exception:
                W = np.eye(N)
            AwA = A.T @ W @ A
            try:
                inv_AwA = np.linalg.inv(AwA)
            except np.linalg.LinAlgError:
                inv_AwA = np.linalg.pinv(AwA)
            lambda_hat = (inv_AwA @ (A.T @ W @ b)).reshape(-1)
        else:
            W = np.eye(N)
            lambda_hat = lambda_1

        g_hat = g_lambda(lambda_hat).reshape(-1, 1)  # N x 1
        J = float(Tn * (g_hat.T @ W @ g_hat))  # scalar
        df = max(N - K, 1)
        try:
            from scipy import stats

            pval = float(1.0 - stats.chi2.cdf(J, df=df))
        except Exception:
            pval = np.nan

        self.gmm_result = {"lambda": lambda_hat, "J": J, "pval": pval, "W": W}
        self._logger.info("GMM linear pricing estimation completed.")
        return self.gmm_result

    # ===========================
    # Statistical utilities
    # ===========================

    def t_test(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
        alternative: str = "two-sided",
    ) -> Tuple[float, float]:
        """One-sample t-test for H0: mean(data) == 0.

        Args:
            data: 1-D data (pandas Series, DataFrame, or numpy array). NaNs are dropped.
            alternative: 'two-sided' (default), 'greater' (mean > 0), or 'less' (mean < 0).

        Returns:
            Tuple of (t_statistic, p_value).
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            arr = data.values.ravel()
        else:
            arr = np.asarray(data)
        arr = arr[~pd.isna(arr)]
        if arr.size < 2:
            return np.nan, np.nan

        n = arr.size
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        se = std / np.sqrt(n) if std > 0 else 0.0

        if se == 0.0:
            if mean == 0.0:
                return 0.0, 1.0
            t_stat = np.inf if mean > 0 else -np.inf
            return t_stat, 0.0

        t_stat = mean / se

        try:
            from scipy import stats

            if alternative == "two-sided":
                p_val = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
            elif alternative == "greater":
                p_val = float(stats.t.sf(t_stat, df=n - 1))
            elif alternative == "less":
                p_val = float(stats.t.cdf(t_stat, df=n - 1))
            else:
                raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
        except Exception:
            # Normal approximation fallback
            z = t_stat
            if alternative == "two-sided":
                p_val = float(math.erfc(abs(z) / np.sqrt(2)))
            elif alternative == "greater":
                p_val = float(0.5 * math.erfc(-z / np.sqrt(2)))
            elif alternative == "less":
                p_val = float(0.5 * math.erfc(z / np.sqrt(2)))
            else:
                raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")

        return t_stat, p_val

    def white_test(
        self, X: np.ndarray, resid: np.ndarray, add_intercept: bool = True
    ) -> Tuple[float, float]:
        """White's heteroskedasticity test (LM) for regression residuals.

        Regress squared residuals on regressors, their squares, and cross terms, then compute LM statistic.

        Args:
            X: Design matrix (n x p).
            resid: Residuals vector (n,).
            add_intercept: Whether to include intercept in the auxiliary regression.

        Returns:
            Tuple of (LM statistic, p-value) with chi-square approximation.
        """
        try:
            n, p = X.shape
            # Build polynomial features up to degree 2
            Z_list = [X, X**2]
            for i in range(p):
                for j in range(i + 1, p):
                    Z_list.append((X[:, i] * X[:, j]).reshape(-1, 1))
            Z = np.column_stack(Z_list)
            if add_intercept:
                Z = self._add_intercept(Z)

            y_aux = resid**2
            _, _, _, _, r2 = self._ols_fit(
                Z, y_aux, add_intercept=False, cov_type="none"
            )
            df = Z.shape[1]
            LM = n * r2 if np.isfinite(r2) else np.nan
            try:
                from scipy import stats

                p_val = float(1.0 - stats.chi2.cdf(LM, df=df))
            except Exception:
                p_val = np.nan
            return float(LM), p_val
        except Exception as e:
            self._logger.error(f"White test failed: {e}")
            return np.nan, np.nan

    def attribute_portfolio(
        self,
        weights: pd.DataFrame,
        horizon: int = 1,
        feasible: Optional[pd.DataFrame] = None,
        other_factors: Optional[Dict[str, pd.DataFrame]] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        standardize_exposures: bool = False,
        normalize_weights: bool = True,
        normalize_mode: Literal["sum", "abs"] = "sum",
        build_factor_returns: bool = True,
        group_n: int = 10,
        group_mode: Literal["conditional", "independent"] = "conditional",
        group_cell_weight: Literal["equal", "count"] = "equal",
        hl_mode: Literal["first_last", "extreme"] = "first_last",
    ) -> "Evaluator":
        """Attribute a portfolio's returns to factor exposures and factor returns.

        This method performs characteristics-based attribution:
        - Portfolio factor exposure at time t: E_{p,k,t} = sum_i w_{t,i} * X_{k,t,i}
            where X_{k,t,i} is the cross-sectional exposure of asset i to factor k at time t.
        - Factor return contribution: C_{k,t} = E_{p,k,t} * F_{k,t}, where F_{k,t} is the factor return at t.
        - Residual (idiosyncratic): R_{p,t} - sum_k C_{k,t}.

        Factor returns:
        - If factor_returns is provided, it should be a DataFrame (dates x K) and columns should match:
            'factor' for the primary factor and the keys in other_factors (if provided).
        - If factor_returns is None and build_factor_returns=True:
            - The HL factor for the primary factor is built via grouped portfolios (using group_n, group_mode, hl_mode).
            - For each other factor (if provided), a temporary Evaluator is created to build its own HL using the same price
                and the same grouping params (single-factor HL per control factor).
        - If factor_returns is None and build_factor_returns=False: contributions will not be computed (NaN).

        Weights and alignment:
        - weights is a wide DataFrame (dates x assets). We assume weights at t are applied to returns from t to t+horizon.
        - If normalize_weights=True, weights are normalized per date:
            - normalize_mode='sum': divide by sum of weights (can be negative for long/short).
            - normalize_mode='abs': divide by sum of absolute weights (keeps total gross at 1).
        - Weights are zeroed where assets are infeasible or have NaN returns.

        Args:
            weights: Portfolio weights (dates x assets).
            horizon: Return horizon (in periods) for realized portfolio returns and factor contributions.
            feasible: Optional eligibility mask (dates x assets). Infeasible assets receive zero weight.
            other_factors: Optional dict of name -> exposure DataFrame for multi-factor exposure attribution.
                        Exposures must be aligned by dates/assets to the primary factor and price universe.
            factor_returns: Optional DataFrame (dates x K) of factor returns. Columns should be:
                            'factor' for the primary factor, and keys matching other_factors.
            standardize_exposures: Cross-sectionally standardize each factor's exposures per date (z-score).
            normalize_weights: Normalize weights per date (after masking infeasible/NaN returns).
            normalize_mode: 'sum' for sum-to-1; 'abs' for gross = 1 using sum of absolute weights.
            build_factor_returns: If True and factor_returns is None, build HL returns for the primary and other factors.
            group_n: n-quantiles used if building HL factor returns.
            group_mode: 'conditional' or 'independent' sorting mode used when building HL.
            group_cell_weight: Aggregation across buckets when reporting group_returns for HL building.
            hl_mode: 'first_last' or 'extreme' for HL construction when building HL.

        Returns:
            Evaluator: self with attribute `portfolio_attribution` containing:
                - 'portfolio_return': Series of realized portfolio returns.
                - 'factor_exposure': DataFrame (dates x K) of portfolio factor exposures.
                - 'factor_returns': DataFrame (dates x K) of factor returns used.
                - 'factor_contribution': DataFrame (dates x K) of factor contributions.
                - 'total_factor_contribution': Series sum across factors of contributions.
                - 'residual': Series of idiosyncratic residual = portfolio_return - total_factor_contribution.

        Notes:
            - Exposures use a characteristics model (weighted average of cross-sectional exposures).
            - Contributions multiply exposures at t by factor returns over t to t+horizon, aligned by index.
            - If factor returns are not supplied and cannot be built, contributions will be NaN.
        """
        # 1) Prepare returns and align
        future = self._future_return(horizon)  # asset returns from t to t+h
        # Align weights to price/returns universe
        weights = weights.reindex(index=future.index, columns=future.columns).fillna(
            0.0
        )
        # Mask out assets with NaN returns on the date
        weights = weights.where(future.notna(), 0.0)

        # Apply feasibility mask if provided
        if feasible is not None:
            feasible = feasible.reindex_like(future).fillna(False)
            weights = weights.where(feasible.astype(bool), 0.0)

        # Normalize weights per date if requested
        if normalize_weights:
            if normalize_mode == "abs":
                denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
            else:
                denom = weights.sum(axis=1).replace(0.0, np.nan)
            weights = weights.div(denom, axis=0).fillna(0.0)

        # 2) Realized portfolio return from t to t+h
        port_ret = (weights * future).sum(axis=1)

        # 3) Build factor exposure matrices
        factor_expo_dict: Dict[str, pd.DataFrame] = {
            "factor": self._factor.reindex_like(future)
        }
        if other_factors is not None:
            for name, df in other_factors.items():
                factor_expo_dict[name] = df.reindex_like(future)

        # Optionally standardize exposures per date (z-score)
        if standardize_exposures:
            for name, X in factor_expo_dict.items():
                mu = X.mean(axis=1)
                sd = X.std(axis=1, ddof=1).replace(0.0, np.nan)
                factor_expo_dict[name] = (X.sub(mu, axis=0)).div(sd, axis=0)

        # 4) Portfolio factor exposures over time: E_{p,k,t} = sum_i w_{t,i} * X_{k,t,i}
        expo_df = pd.DataFrame(
            index=future.index, columns=list(factor_expo_dict.keys()), dtype="float"
        )
        for name, X in factor_expo_dict.items():
            X = X.where(future.notna())  # ignore assets without returns
            expo_df[name] = (weights * X).sum(axis=1)

        # 5) Factor returns matrix (dates x K)
        fac_ret_df: Optional[pd.DataFrame] = None
        if factor_returns is not None:
            # Align to working index
            fac_ret_df = factor_returns.reindex(index=future.index)
        elif build_factor_returns:
            # Build HL for the primary factor using this evaluator
            fac_cols: List[str] = []
            fac_list: List[pd.Series] = []

            # Primary factor HL
            self.get_group_returns(
                n=group_n,
                horizon=horizon,
                other_factors=None,
                mode=group_mode,
                feasible=feasible,
                weight=None,
                cell_weight=group_cell_weight,
                hl_mode=hl_mode,
            )
            hl_main = self.sorted_factor_return.rename("factor")
            fac_cols.append("factor")
            fac_list.append(hl_main)

            # Other factors HL (single-factor HL per control factor)
            if other_factors is not None and len(other_factors) > 0:
                for name, df in other_factors.items():
                    try:
                        tmp_eval = Evaluator(
                            factor=df, price=self._price, logger=self._logger
                        )
                        tmp_eval.get_group_returns(
                            n=group_n,
                            horizon=horizon,
                            other_factors=None,
                            mode=group_mode,
                            feasible=feasible,
                            weight=None,
                            cell_weight=group_cell_weight,
                            hl_mode=hl_mode,
                        )
                        fac_cols.append(name)
                        fac_list.append(tmp_eval.sorted_factor_return.rename(name))
                    except Exception as e:
                        self._logger.error(
                            f"Failed to build HL for factor '{name}': {e}"
                        )

            if len(fac_list) > 0:
                fac_ret_df = pd.concat(fac_list, axis=1).reindex(index=future.index)
            else:
                fac_ret_df = None

        # 6) Factor contributions: C_{k,t} = E_{p,k,t} * F_{k,t}
        if fac_ret_df is not None:
            common_cols = [c for c in expo_df.columns if c in fac_ret_df.columns]
            fac_ret_used = fac_ret_df[common_cols]
            expo_used = expo_df[common_cols]
            factor_contrib = expo_used.mul(fac_ret_used, axis=0)
            total_factor_contrib = factor_contrib.sum(axis=1)
            residual = port_ret - total_factor_contrib
        else:
            # Cannot compute contributions without factor returns
            factor_contrib = pd.DataFrame(
                index=future.index, columns=expo_df.columns, dtype="float"
            )
            factor_contrib[:] = np.nan
            total_factor_contrib = pd.Series(
                np.nan, index=future.index, name="TotalFactor"
            )
            residual = pd.Series(np.nan, index=future.index, name="Residual")
            fac_ret_used = None

        # 7) Store results
        self.portfolio_attribution = {
            "portfolio_return": port_ret.rename("PortfolioReturn"),
            "factor_exposure": expo_df,
            "factor_returns": fac_ret_used if fac_ret_df is not None else None,
            "factor_contribution": factor_contrib,
            "total_factor_contribution": total_factor_contrib.rename("TotalFactor"),
            "residual": residual.rename("Residual"),
            "weights_used": weights,
        }
        self._logger.info("Portfolio attribution completed.")
        return self

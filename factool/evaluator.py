import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from logging import Logger

from parquool import setup_logger


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
        self,
        factor: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
        price: pd.DataFrame,
        logger: Optional[Logger] = None,
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
        self._price = price.copy()

        self._factors: Dict[str, pd.DataFrame] = self._normalize_factors_input(factor)
        self._names: List[str] = list(self._factors.keys())
        self._name: str = self._names[0]
        self._factor: pd.DataFrame = self._factors[self._name]
        self._align_factor_price()

        # Shifted price to anchor "t+1" to avoid look-ahead bias when computing future returns.
        self._shifted = self._price.shift(-1)

        # Placeholders for analysis outputs
        self.group_returns: Dict[str, pd.DataFrame] = {}
        self.sorted_factor_return: pd.DataFrame = None
        self.factor_exposure: Optional[pd.DataFrame] = None
        self.ts_intercept: Optional[pd.DataFrame] = None
        self.factor_exposure_t: Optional[pd.DataFrame] = None
        self.ic: Optional[pd.Series] = None
        self.direction: Optional[float] = None
        self.factor_premia: Optional[pd.DataFrame] = None
        self.factor_premia_t: Optional[pd.DataFrame] = None
        self.factor_r2: Optional[pd.Series] = None
        self.fmb_premia: Optional[pd.Series] = None
        self.fmb_tstats: Optional[pd.Series] = None
        self.grs_stat: Optional[float] = None
        self.grs_pval: Optional[float] = None
        self.gmm_result: Optional[Dict[str, Union[np.ndarray, float]]] = None

    # ===========================
    # Internal helpers
    # ===========================

    @staticmethod
    def _ensure_name(df: pd.DataFrame, fallback: str) -> Tuple[str, pd.DataFrame]:
        nm = df.attrs.get("name", None)
        if nm is None or not isinstance(nm, str) or len(nm.strip()) == 0:
            nm = fallback
        df = df.copy()
        df.attrs["name"] = nm
        return nm, df

    def _normalize_factors_input(
        self,
        factor_in: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """Normalize factor input into format: {name: DataFrame} dictionary, ensuring unique name"""
        out: Dict[str, pd.DataFrame] = {}
        if isinstance(factor_in, pd.DataFrame):
            name, df = self._ensure_name(factor_in, "factor")
            out[name] = df.copy()
        elif isinstance(factor_in, (list, tuple)):
            used = set()
            for i, df in enumerate(factor_in, start=1):
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(
                        "All items in factor list must be pandas DataFrame."
                    )
                name, dfi = self._ensure_name(df, f"factor_{i}")
                base = name
                k = 1
                while name in used:
                    name = f"{base}_{k}"
                    k += 1
                used.add(name)
                dfi.attrs["name"] = name
                out[name] = dfi.copy()
        elif isinstance(factor_in, dict):
            used = set()
            for name, df in factor_in.items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(
                        "All values in factor dict must be pandas DataFrame."
                    )
                nm = str(name)
                if len(nm.strip()) == 0:
                    raise ValueError("Empty factor name is not allowed.")
                base = nm
                k = 1
                while nm in used:
                    nm = f"{base}_{k}"
                    k += 1
                used.add(nm)
                dfi = df.copy()
                dfi.attrs["name"] = nm
                out[nm] = dfi
        else:
            raise TypeError(
                "factor must be a DataFrame, a list/tuple of DataFrames, or a dict[name -> DataFrame]."
            )
        return out

    def _align_factor_price(self) -> None:
        """Align factor and price indices, with informative logging."""
        cidx = self._price.index
        ccol = self._price.columns
        for f in self._factors.values():
            cidx = cidx.union(f.index)
            ccol = ccol.union(f.columns)
        self._logger.info(
            f"Added {cidx.difference(self._price.index).size} rows to price matrix"
        )
        self._price = self._price.reindex(index=cidx)
        self._logger.info(
            f"Added {ccol.difference(self._price.columns).size} cols to price matrix"
        )
        self._price = self._price.reindex(columns=ccol)
        for nm, f in self._factors.items():
            self._logger.info(
                f"Added {cidx.difference(f.index).size} rows to {nm} matrix"
            )
            self._factors[nm] = f.reindex(index=cidx)
            self._logger.info(
                f"Added {ccol.difference(f.columns).size} cols to {nm} matrix"
            )
            self._factors[nm] = f.reindex(columns=ccol)

    @staticmethod
    def _default_feasible_like(df: pd.DataFrame) -> pd.DataFrame:
        """Create a boolean DataFrame of the same shape as df, filled with True."""
        return pd.DataFrame(True, index=df.index, columns=df.columns)

    @staticmethod
    def _default_weight_like(df: pd.DataFrame) -> pd.DataFrame:
        """Create a float DataFrame of the same shape as df, filled with ones."""
        return pd.DataFrame(1.0, index=df.index, columns=df.columns)

    def _future_return(self, horizon: int, skip: bool = True) -> pd.DataFrame:
        """Compute future returns using an anchored shift to avoid look-ahead bias."""
        if not skip:
            return self._shifted.shift(-horizon) / self._shifted - 1
        nonreturn_days = pd.DataFrame(
            np.zeros_like(self._price),
            index=self._price.index,
            columns=self._price.columns,
            dtype="bool",
        )
        nonreturn_days.iloc[::horizon] = True
        return (self._shifted.shift(-horizon) / self._shifted - 1).where(nonreturn_days)

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

        _, p = Xw.shape
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
    def _rolling_regression(
        y: np.ndarray,
        x: np.ndarray,
        dates: pd.Index,
        window: int,
        min_obs: int,
        intercept: bool,
    ) -> Tuple[pd.Series, pd.Series]:
        """Run rolling OLS for regressor with optional intercept."""
        betas = np.full(
            (len(dates), x.shape[1] + int(intercept)), np.nan, dtype="float"
        )
        tstats = np.full(
            (len(dates), x.shape[1] + (1 if intercept else 0)), np.nan, dtype="float"
        )

        for j in range(len(dates)):
            start = max(0, j - window + 1)
            y_win = y[start : j + 1]
            x_win = x[start : j + 1]
            valid = np.isfinite(y_win) & np.isfinite(x_win).all(axis=1)
            if valid.sum() < min_obs:
                continue
            beta_vec, _, t_vec, _, _ = Evaluator._ols_fit(
                X=x_win, y=y_win, add_intercept=intercept, cov_type="none"
            )
            beta_vec = beta_vec.astype(np.float64)
            t_vec = t_vec.astype(np.float64)
            betas[j] = beta_vec
            tstats[j] = t_vec

        beta_s = pd.DataFrame(betas, index=dates)
        t_s = pd.DataFrame(
            tstats,
            index=dates,
        )
        return beta_s, t_s

    # ===========================
    # Grouping and HL factor
    # ===========================

    def get_group_returns(
        self,
        n: int = 10,
        horizon: int = 1,
        skip_horizon: bool = True,
        mode: Literal["single", "conditional", "independent"] = "single",
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
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

        Args:
            n: Number of quantile groups.
            horizon: Return horizon (in periods).
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
            mode: Bucketing mode: 'single', 'conditional' or 'independent'.
            feasible: Optional boolean DataFrame for asset eligibility.
            weight: Optional DataFrame of within-group weights.

        Returns:
            Evaluator: self with attributes:
                - group_returns: DataFrame of group returns per date, columns G1..Gn.
                - sorted_factor_return: Series of HL factor return (per definitions above).

        Raises:
            ValueError: If mode is not recognized.
            TypeError: If other_factors is not None/list/tuple.
        """
        mode = str(mode).lower()
        if mode not in ("single", "independent", "conditional"):
            self._logger.error("Mode must be 'independent' or 'conditional'")
            return self

        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for group return"
        )
        asset_returns = self._future_return(horizon, skip=skip_horizon)

        feasible = (
            feasible.reindex_like(asset_returns).fillna(False)
            if feasible is not None
            else self._default_feasible_like(asset_returns)
        )
        weight = (
            weight.reindex_like(asset_returns)
            if weight is not None
            else self._default_weight_like(asset_returns)
        )

        dates = asset_returns.index

        for name, factor in self._factors.items():
            g_rets = {}
            other_names = list(filter(lambda x: x != name, self._names))
            other_factors: List[pd.DataFrame] = [
                self._factors[nm] for nm in other_names
            ]
            for dt in dates:
                try:
                    eligible = (
                        feasible.loc[dt].astype(bool)
                        & asset_returns.loc[dt].notna()
                        & factor.loc[dt].notna()
                    )
                    r_t: pd.Series = asset_returns.loc[dt]
                    w_t: pd.Series = weight.loc[dt]
                    f_t: pd.Series = factor.loc[dt]

                    if eligible.sum() == 0:
                        self._logger.warning(
                            f"No eligible asset on {dt} when computing group return"
                        )
                        if mode == "single" or len(other_factors) == 0:
                            g_rets[dt] = pd.Series(
                                {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                name=dt,
                            )
                        else:
                            # Multi-factor sorting, a DataFrame should be appended
                            g_rets[dt] = (
                                pd.Series(
                                    {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                    name=dt,
                                )
                                .to_frame()
                                .T
                            )
                        continue

                    if mode == "single" or len(other_factors) == 0:
                        g_ret = pd.Series(
                            {f"{name}({i})": np.nan for i in range(1, n + 1)}, name=dt
                        )
                        g_t = self._qcut_groups(f_t.where(eligible), n)
                        if g_t.notna().sum() == 0:
                            g_rets[dt] = g_ret
                            continue
                        uniq = sorted(pd.unique(g_t.dropna().astype(int)))
                        for gi in uniq:
                            mask = g_t == gi
                            g_ret[f"{name}({gi})"] = self._group_return(
                                r_t.where(mask & eligible), w_t.where(mask & eligible)
                            )
                        g_rets[dt] = g_ret
                        continue

                    # Independent bucketing (global groups)
                    elif mode == "independent":
                        g_ret = (
                            pd.Series(
                                {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                name=dt,
                            )
                            .to_frame()
                            .T
                        )
                        # Control-factor global groups
                        other_groups = [
                            self._qcut_groups(of.loc[dt].where(eligible), n)
                            for of in other_factors
                        ]
                        g_t_global = self._qcut_groups(f_t.where(eligible), n)

                        keys_df = pd.concat(other_groups, axis=1)
                        keys_df.columns = other_names

                        valid = (
                            eligible & keys_df.notna().all(axis=1) & g_t_global.notna()
                        )
                        if valid.sum() == 0:
                            g_rets[dt] = g_ret
                            continue

                        tmp = keys_df[valid].copy()
                        tmp["target_g"] = g_t_global[valid]
                        tmp["ret"] = r_t[valid]
                        tmp["w"] = w_t[valid]

                        bucket_results: List[Dict[int, float]] = []

                        # Compute returns for each cell (each control-bucket combination)
                        for gn, sub_idx in tmp.groupby(
                            list(keys_df.columns)
                        ).groups.items():
                            # For multi-factor and single factor, gn is not the same data type
                            if not isinstance(gn, tuple):
                                gn = (gn,)
                            sub = tmp.loc[sub_idx]
                            sub_group_ret = pd.Series(
                                {i: np.nan for i in range(1, n + 1)},
                                name="/".join(
                                    [
                                        f"{on}({int(g)})"
                                        for on, g in zip(other_names, gn)
                                    ]
                                ),
                            )
                            for gi in range(1, n + 1):
                                sel = sub["target_g"] == gi
                                if sel.sum() == 0:
                                    continue
                                sub_group_ret[gi] = self._group_return(
                                    sub.loc[sel, "ret"], sub.loc[sel, "w"]
                                )
                            bucket_results.append(sub_group_ret)

                        # Aggregate group returns for reporting (equal/count)
                        g_ret = pd.DataFrame(bucket_results)
                        g_ret.columns = [f"{name}({i})" for i in range(1, n + 1)]
                        g_rets[dt] = g_ret
                        continue

                    elif mode == "conditional":
                        # Conditional (hierarchical on controls, local sorting of main factor)
                        buckets = [pd.Index(f_t.index[eligible], name="")]
                        for nm, of in zip(other_names, other_factors):
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
                                        new_buckets.append(
                                            pd.Index(
                                                sub_idx,
                                                name=idxs.name + f"/{nm}({label})",
                                            )
                                        )
                            buckets = new_buckets if len(new_buckets) > 0 else buckets

                        bucket_results: List[Dict[int, float]] = []
                        for idxs in buckets:
                            sub_idx = pd.Index(idxs).intersection(f_t.index[eligible])
                            if len(sub_idx) == 0:
                                continue
                            g_local = self._qcut_groups(f_t.loc[sub_idx], n)
                            if g_local.notna().sum() == 0:
                                continue

                            sub_group_ret = pd.Series(
                                {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                name=idxs.name[1:],
                            )
                            uniq = sorted(pd.unique(g_local.dropna().astype(int)))
                            for gi in uniq:
                                top_mask = g_local == gi
                                ret_g = self._group_return(
                                    r_t.loc[sub_idx].where(top_mask),
                                    w_t.loc[sub_idx].where(top_mask),
                                )
                                sub_group_ret[f"{name}({gi})"] = ret_g
                            bucket_results.append(sub_group_ret)
                        g_ret = pd.concat(bucket_results, axis=1).T
                        g_rets[dt] = g_ret

                except Exception as e:
                    self._logger.error(f"get_group_returns error on {dt}: {e}")

            self.group_returns[name] = pd.concat(
                g_rets.values(),
                keys=g_rets.keys(),
                axis=int(isinstance(g_rets[dt], pd.Series)),
            ).dropna(how="all", axis=1)
            self.group_returns[name] = (
                self.group_returns[name].T
                if isinstance(g_rets[dt], pd.Series)
                else self.group_returns[name]
            )
        self.sorted_factor_return = pd.concat(
            [
                (
                    sgr.dropna(axis=0, how="all").iloc[:, -1].groupby(level=0).sum()
                    - sgr.dropna(axis=0, how="all").iloc[:, 0].groupby(level=0).sum()
                )
                for sgr in self.group_returns.values()
            ],
            keys=self.group_returns.keys(),
            axis=1,
        )
        self._logger.info(
            f"Group returns computed: n={n}, horizon={horizon}"
            + (f", mode={mode}" if len(self._factors) - 1 else "")
        )
        return self

    def time_series_regression(
        self,
        horizon: Optional[int] = 1,
        rolling: bool = False,
        window: int = 252,
        min_obs: int = 60,
        add_intercept: bool = True,
        cov_type: Literal["none", "white", "nw"] = "nw",
        nw_lag: int = 3,
        hc_type: str = "HC1",
        feasible: Optional[pd.DataFrame] = None,
        n_jobs: int = -1,
    ) -> "Evaluator":
        """Run time-series regressions of asset returns on factor returns.

        This unified implementation supports:
        - Full-sample OLS for multi-factor setups.
        - Rolling OLS for single-factor setups (used by get_factor_exposure).

        Args:
            horizon: Return horizon for dependent variable when asset_returns is None.
            other_factor_returns: Optional DataFrame of factor returns (dates x K). If None,
                uses the constructed HL factor (sorted_factor_return) if available.
            rolling: Whether to perform rolling regression (single factor only).
            window: Rolling window size for time-series regression (if rolling=True).
            min_obs: Minimum valid observations in a window to compute regression (rolling).
            add_intercept: Whether to include an intercept (alpha) in the regression.
            cov_type: Covariance estimator for full-sample regression: 'none', 'white', or 'nw'.
            nw_lag: Newey-West lag (if cov_type='nw').
            hc_type: White estimator type, 'HC0' or 'HC1'.
            feasible: Optional eligibility DataFrame for masking asset returns (dates x assets).
            n_jobs: Number of parallel jobs for asset-wise rolling regression; -1 uses all CPUs.

        Returns:
            Evaluator: self with attributes populated:
            - factor_exposure: DataFrame of rolling betas (dates x assets).
            - factor_exposure_t: DataFrame of rolling t-stats for beta (dates x assets).
            - ts_intercept: DataFrame of rolling alphas (dates x assets; if add_intercept=True).

        Raises:
            ValueError: If factor returns are not available, cannot be aligned, or rolling is
                requested with multiple factors.
        """
        # Prepare asset returns
        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for time series regression"
        )
        asset_returns = self._future_return(horizon, skip=False)

        # Prepare factor returns
        if not (
            hasattr(self, "sorted_factor_return")
            and self.sorted_factor_return is not None
        ):
            self._logger.error(
                "No factor return founded, run `get_group_return` first."
            )
            return self

        factor_returns = self.sorted_factor_return.copy()
        # Align indices
        idx = asset_returns.index.intersection(factor_returns.index)
        if idx.empty:
            self._logger.error("No available date found.")
            return self
        asset_returns = asset_returns.loc[idx]
        factor_returns = factor_returns.loc[idx]

        # Apply feasible mask if provided
        if feasible is not None:
            feasible = feasible.reindex(index=idx, columns=asset_returns.columns)
            asset_returns = asset_returns.where(feasible.astype(bool))
            self._logger.debug("Feasible mask used.")

        # Rolling single-factor regression
        if rolling:
            x_series = factor_returns.copy()
            dates = idx
            assets = list(asset_returns.columns)

            def _asset_rolling(col: str):
                y = asset_returns[col].values
                x = x_series.values
                beta_s, t_s = self._rolling_regression(
                    y=y,
                    x=x,
                    dates=dates,
                    window=window,
                    min_obs=min_obs,
                    intercept=add_intercept,
                )
                return (beta_s, t_s)

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_asset_rolling)(col) for col in assets
            )

            beta_df = pd.concat([r[0] for r in results], axis=0, keys=assets)
            t_df = pd.concat([r[1] for r in results], axis=0, keys=assets)
            beta_df.columns = (["intercept"] if add_intercept else []) + self._names
            t_df.columns = (["intercept"] if add_intercept else []) + self._names
            t_df = t_df.add_suffix("-t")
            self.factor_exposure = beta_df
            self.factor_exposure_t = t_df
            self._logger.info(
                f"Rolling TS regression completed: window={window}, min_obs={min_obs}, "
                f"intercept={add_intercept}, horizon={horizon}, factors={list(factor_returns.columns)}"
            )
            return self

        # Full-sample multi-factor regression
        x = factor_returns.values  # T x K
        assets = list(asset_returns.columns)
        alpha_vals: Dict[str, float] = {}
        beta_vals: Dict[str, np.ndarray] = {}
        tstats: Dict[str, np.ndarray] = {}

        for asset in assets:
            y = asset_returns[asset].values
            beta, _, t, _, _ = self._ols_fit(
                X=x,
                y=y,
                add_intercept=add_intercept,
                cov_type=cov_type,
                hc_type=hc_type,
                nw_lag=nw_lag,
            )
            if add_intercept:
                alpha_vals[asset] = float(beta[0])
                beta_vals[asset] = beta[1:].astype(float)
                tstats[asset] = t.astype(float)
            else:
                alpha_vals[asset] = np.nan
                beta_vals[asset] = beta.astype(float)
                tstats[asset] = t.astype(float)

        self.factor_exposure = pd.DataFrame(beta_vals, index=factor_returns.columns).T
        self.ts_intercept = pd.Series(alpha_vals, name="intercept")
        self.factor_exposure_t = pd.DataFrame(
            tstats,
            index=factor_returns.columns,
        ).T
        self._logger.info(
            f"Full-sample TS regression completed: intercept={add_intercept}, cov_type={cov_type}, "
            f"nw_lag={nw_lag}, factors={list(factor_returns.columns)}"
        )
        return self

    def get_factor_exposure(
        self,
        horizon: int = 1,
        feasible: Optional[pd.DataFrame] = None,
        window: int = 252,
        min_obs: int = 60,
        intercept: bool = True,
        n_jobs: int = 1,
    ) -> "Evaluator":
        """Rolling time-series regression: asset returns vs. constructed HL factor.

        This method builds the HL factor via grouped portfolios (requires `get_group_return`
        to have been run), and then delegates the rolling regression to
        `time_series_regression`.

        Args:
            horizon: Return horizon for the dependent variable.
            feasible: Optional eligibility DataFrame for masking asset returns.
            window: Rolling window size for time-series regression.
            min_obs: Minimum valid observations in a window to compute regression.
            intercept: Whether to include intercept in rolling regression.
            standardize_factor: Whether to standardize the HL factor prior to regression.
            n_jobs: Number of parallel jobs for asset-wise regression; -1 uses all CPUs.

        Returns:
            Evaluator: self with attributes:
                - ts_exposure_beta: DataFrame of rolling betas.
                - ts_exposure_t: DataFrame of rolling t-statistics of beta.
                - ts_exposure_alpha: DataFrame of rolling alphas (if intercept=True).

        Raises:
            ValueError: If no factor return is available (run `get_group_return` first).
        """
        if self.group_returns is None or self.sorted_factor_return is None:
            self._logger.error(
                "No factor return founded, please run `get_group_return` first."
            )
            return self

        # Delegate to the unified time_series_regression (rolling single-factor)
        self.time_series_regression(
            horizon=horizon,
            rolling=True,
            window=window,
            min_obs=min_obs,
            add_intercept=intercept,
            cov_type="none",  # rolling uses plain OLS for speed/stability
            feasible=feasible,
            n_jobs=n_jobs,
        )

        self._logger.info(
            f"TS exposure evaluated (window={window}, min_obs={min_obs}, intercept={intercept}, horizon={horizon}) "
        )
        return self

    def get_info_coef(
        self, horizon: int = 1, skip_horizon: bool = True, method: str = "spearman"
    ):
        """Evaluate Information Coefficient (IC) between factor exposures and future returns.

        Args:
            horizon: Return horizon in periods.
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
            method: Correlation method ('pearson', 'spearman', 'kendall').

        Returns:
            Evaluator: self with attributes:
                - ic: Series of daily IC values.
                - direction: Sign of mean IC to be used for portfolio direction.
        """
        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for information coefficiency"
        )
        asset_returns = self._future_return(horizon, skip=skip_horizon)
        self.ic = [
            factor.corrwith(asset_returns, axis=1, method=method).dropna(
                axis=0, how="all"
            )
            for factor in self._factors.values()
        ]
        self.ic = pd.concat(self.ic, keys=self._names, axis=1)

        self.direction = np.sign(self.ic.mean())
        return self

    # ===========================
    # Cross-sectional analytics
    # ===========================

    def cross_sectional_regression(
        self,
        horizon: int = 1,
        skip_horizon: bool = True,
        feasible: Optional[pd.DataFrame] = None,
        weight: Optional[pd.DataFrame] = None,
        add_intercept: bool = True,
        cov_type: Literal["none", "white"] = "white",
        white_type: str = "HC1",
    ):
        """Run cross-sectional regressions of future returns on characteristics (factors).

        For single-factor study, this uses the primary factor only.
        For multi-factor study, supply `other_factors` to include additional regressors alongside the primary factor.

        Args:
            horizon: Return horizon in periods.
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
            feasible: Optional boolean eligibility mask DataFrame.
            weight: Optional cross-sectional weights for assets at each date.
            add_intercept: Whether to include intercept in the cross-sectional regression.
            cov_type: Covariance estimator: 'none' or 'white'.
            white_type: White estimator type ('HC0' or 'HC1').

        Returns:
            Evaluator: self with attributes:
                - cs_betas: DataFrame (dates x factors) of cross-sectional betas.
                - cs_tstats: DataFrame (dates x factors) of t-stats.
                - cs_r2: Series of cross-sectional R-squared per date.
        """
        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for cross sectional regression"
        )
        asset_returns = self._future_return(horizon, skip=skip_horizon)
        idx, cols = asset_returns.index, asset_returns.columns
        for fct in self._factors.values():
            idx = idx.intersection(fct.index)
            cols = cols.intersection(fct.columns)
        asset_returns = asset_returns.loc[idx, cols]

        feasible = (
            feasible.reindex(index=idx, columns=cols).fillna(False)
            if feasible is not None
            else self._default_feasible_like(asset_returns)
        )
        if weight is not None:
            weight = weight.reindex(index=idx, columns=cols).fillna(0.0)

        dates = asset_returns.dropna(axis=0, how="all").index
        default = np.full(len(self._names) + int(add_intercept), np.nan)
        betas, tstats, r2vals = [], [], []

        for dt in dates:
            y = asset_returns.loc[dt].values
            Xi = np.column_stack([fct.loc[dt].values for fct in self._factors.values()])

            valid = (
                feasible.loc[dt].astype(bool).values
                & np.isfinite(y)
                & np.all(np.isfinite(Xi), axis=1)
            )
            min_req = Xi.shape[1] + (1 if add_intercept else 0) + 1
            if valid.sum() < min_req:
                self._logger.warning(
                    f"Valid asset ({valid.sum()}) is less than min_req ({min_req}) on {dt}"
                )
                betas.append(
                    pd.Series(
                        default,
                        index=(["intercept"] if add_intercept else []) + self._names,
                    )
                )
                tstats.append(
                    pd.Series(
                        default,
                        index=(["intercept"] if add_intercept else []) + self._names,
                    ).add_suffix("-t")
                )
                r2vals.append(np.nan)
                continue

            w = None
            if weight is not None:
                w = weight.loc[dt].values
                w = np.where(valid, w, 0.0)
                if (w > 0).sum() == 0:
                    w = None

            beta, _, t, _, r2 = self._ols_fit(
                X=Xi[valid],
                y=y[valid],
                add_intercept=add_intercept,
                cov_type=cov_type,
                hc_type=white_type,
                weights=w[valid] if w is not None else None,
            )
            # Map coefficients: intercept is first (if present)
            betas.append(
                pd.Series(
                    beta, index=(["intercept"] if add_intercept else []) + self._names
                )
            )
            tstats.append(
                pd.Series(
                    t, index=(["intercept"] if add_intercept else []) + self._names
                ).add_suffix("-t")
            )
            r2vals.append(r2)

        self.factor_premia = pd.concat(betas, keys=dates, axis=1).T
        self.factor_premia_t = pd.concat(tstats, keys=dates, axis=1).T
        self.factor_r2 = pd.Series(r2vals, index=dates, name="R2_CS")
        self._logger.info(
            f"Cross-sectional regression completed (horizon={horizon}, add_intercept={add_intercept}, cov_type={cov_type})"
        )
        return self

    def fama_macbeth(
        self,
        nw_lag: int = 3,
    ):
        """Perform Fama-MacBeth regression to estimate factor risk premia.

        Step 1: For each date, run cross-sectional regression of future returns on exposures.
        Step 2: Average the coefficients over time; compute Newey-West HAC t-stats across time.

        Args:
            nw_lag: Newey-West lag for time-series of coefficients.

        Returns:
            Evaluator: self with attributes:
                - fmb_premia: Series of average factor premia across time.
                - fmb_tstats: Series of Newey-West HAC t-stats for premia.
        """
        if not (hasattr(self, "factor_premia") and self.factor_premia is not None):
            self._logger.error(
                "Cross sectional regression not performed, please run `cross_sectional_regression` first."
            )

        beta_ts = self.factor_premia  # dates x factors
        premia = beta_ts.mean(axis=0)

        # HAC t-stats for time-series of coefficients (mean-only regression)
        tstats = {}
        for k in beta_ts.columns:
            x = np.ones((beta_ts[k].shape[0], 1))
            y = beta_ts[k].values
            _, _, t_vec, _, _ = self._ols_fit(
                X=x, y=y, add_intercept=False, cov_type="nw", nw_lag=nw_lag
            )
            tstats[k] = float(t_vec[0])

        self.fmb_premia = premia.rename("FMB_Premia")
        self.fmb_tstats = pd.Series(tstats, name="FMB_t")
        self._logger.info(
            f"Fama-MacBeth regression completed with Newey-West HAC t-stats. (nw_lag={nw_lag})"
        )
        return self

    def grs_test(
        self,
        horizon: int = 1,
        skip_horizon: bool = True,
        add_intercept: bool = True,
    ) -> "Evaluator":
        """Compute the Gibbons-Ross-Shanken (GRS) test for joint alpha = 0.

        Args:
            horizon: Return horizon in periods.
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
            add_intercept: Whether time-series regressions include intercepts (required for GRS).

        Returns:
            Tuple of (GRS F-statistic, p-value).
        """
        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for Gibbons-Ross-Shanken test"
        )
        asset_returns = self._future_return(horizon, skip=skip_horizon)
        if not (
            hasattr(self, "sorted_factor_return")
            and self.sorted_factor_return is not None
        ):
            self._logger.error(
                "No factor return founded, run `get_group_return` first."
            )
            return self
        factor_returns = self.sorted_factor_return

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
                X=F, y=y, add_intercept=add_intercept, cov_type="none"
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
            p_val = float(1.0 - stats.f.cdf(grs, df1=df1, df2=df2))
        except Exception:
            p_val = np.nan

        self.grs_stat = grs
        self.grs_pval = p_val
        self._logger.info(
            f"GRS test computed. (horizon={horizon}, add_intercept={add_intercept})"
        )
        return self

    def gmm_linear_pricing(
        self,
        horizon: int = 1,
        skip_horizon: bool = True,
        two_step: bool = True,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Estimate linear factor risk premia via GMM under SDF m_t = 1 - lambda' F_t.

        Moment conditions: E[m_t R_t] = 0 => E[(1 - lambda' F_t) R_t] = 0.
        Solve for lambda minimizing g(lambda)' W g(lambda), where
        g(lambda) = mean_t[(1 - lambda' F_t) R_t], W is a weighting matrix.

        Args:
            horizon: Return horizon in periods.
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
            two_step: Whether to run two-step GMM (second step uses an estimated optimal weighting).

        Returns:
            Dict with keys:
                - 'lambda': Estimated risk premia (K,).
                - 'J': J-statistic for overidentifying restrictions.
                - 'pval': p-value of J (chi-square with df=N-K).
                - 'W': Weighting matrix used (N x N).
        """
        self._logger.debug(
            f"Apply {'future' if horizon > 0 else 'past'} {abs(horizon)} day return for GMM pricing"
        )
        asset_returns = self._future_return(horizon, skip=skip_horizon)
        if not (
            hasattr(self, "sorted_factor_return")
            and self.sorted_factor_return is not None
        ):
            self._logger.error(
                "No factor return founded, run `get_group_return` first."
            )
            return self
        factor_returns = self.sorted_factor_return.copy()
        # Align indices
        idx = asset_returns.index.intersection(factor_returns.index)
        if idx.empty:
            self._logger.error("No available date found.")
            return self
        asset_returns = asset_returns.loc[idx]
        factor_returns = factor_returns.loc[idx]
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
            pval = float(1.0 - stats.chi2.cdf(J, df=df))
        except Exception:
            pval = np.nan

        self.gmm_result = pd.Series(
            {"lambda": lambda_hat, "J": J, "pval": pval, "W": W}, name="GMM"
        )
        self._logger.info(
            f"GMM linear pricing estimation completed. (horizon={horizon}, two_step={two_step})"
        )
        return self

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
        skip_horizon: bool = True,
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
            skip_horizon: Whtere to skip the horizon in the weights DataFrame.
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
        future = self._future_return(
            horizon, skip=skip_horizon
        )  # asset returns from t to t+h
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

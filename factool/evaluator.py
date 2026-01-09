import math
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from logging import Logger
from scipy import stats

from parquool import setup_logger


MI = pd.MultiIndex


def _is_mi_date_code_index(idx: pd.Index) -> bool:
    return isinstance(idx, pd.MultiIndex) and idx.nlevels == 2


def _mi_names_or_default(idx: MI) -> Tuple[str, str]:
    n0 = idx.names[0] if idx.names and idx.names[0] else "date"
    n1 = idx.names[1] if idx.names and idx.names[1] else "code"
    return n0, n1


def _as_mi_series(x: Union[pd.Series, pd.DataFrame], name: str) -> pd.Series:
    """Convert x to MultiIndex Series (date, code) with a given name."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"{name} DataFrame must have exactly 1 column, got {x.shape[1]}"
            )
        s = x.iloc[:, 0]
        s.name = name
    elif isinstance(x, pd.Series):
        s = x
        s.name = name
    else:
        raise TypeError(f"{name} must be a pandas Series or single-column DataFrame.")
    if not _is_mi_date_code_index(s.index):
        raise TypeError(f"{name} index must be MultiIndex([date, code]).")
    return s


def _ensure_sorted_mi(
    df: Union[pd.Series, pd.DataFrame],
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(df.index, pd.MultiIndex) and not df.index.is_monotonic_increasing:
        return df.sort_index()
    return df


class Evaluator:
    """
    Research framework for studying asset characteristics (factors), based on long-panel data.

    IO contract:
      - factor: MultiIndex([date, code]) Series (single factor) OR DataFrame (multi-factor columns)
      - future: MultiIndex([date, code]) Series OR single-column DataFrame
        Each (date, code) return corresponds to buy at date+1 and sell at date+1+horizon,
        already aligned to factor at `date` by the caller (may be missing / unaligned).
    """

    def __init__(
        self,
        factor: Union[pd.Series, pd.DataFrame],
        future: Union[pd.Series, pd.DataFrame],
        feasible: Optional[Union[pd.Series, pd.DataFrame]] = None,
        weight: Optional[Union[pd.Series, pd.DataFrame]] = None,
        logger: Optional[Logger] = None,
    ):
        self._logger = logger or setup_logger("FactorEvaluator", level="DEBUG")

        # normalize factor(s) to DataFrame columns
        self._factor_df = self._normalize_factor_input(
            factor
        )  # MI index, columns factors
        self._factor_df = _ensure_sorted_mi(self._factor_df)

        # normalize future return to Series named 'ret'
        ret_s = _as_mi_series(future, name="future").copy()
        ret_s = _ensure_sorted_mi(ret_s)
        ret_s.name = "ret"
        self._ret = ret_s

        if not _is_mi_date_code_index(self._factor_df.index):
            raise TypeError("factor index must be MultiIndex([date, code]).")

        self._date_name, self._code_name = _mi_names_or_default(self._factor_df.index)

        # build working panel: inner join on index (no implicit alignment magic)
        self._panel = self._factor_df.join(self._ret, how="inner")
        if self._panel.empty:
            self._logger.warning(
                "Joined panel is empty after aligning factor and future on (date, code)."
            )

        if feasible is not None:
            self._feasible = _as_mi_series(feasible, name="feasible").reindex(
                self._panel.index
            )
        else:
            self._feasible = pd.Series(
                [1] * len(self._panel), name="feasible", index=self._panel.index
            )
        if weight is not None:
            self._weight = _as_mi_series(weight, name="weight").copy(self._panel.index)
        else:
            self._weight = pd.Series(
                [1] * len(self._panel), name="weight", index=self._panel.index
            )
            self._weight = self._weight / self._weight.groupby("date").sum()

        self._names: List[str] = list(self._factor_df.columns)
        self._name: str = self._names[0] if self._names else "factor"
        self._logger.info(
            f"Evaluator initialized with factors={self._names}, panel_rows={len(self._panel)}"
        )

        # Placeholders for analysis outputs
        self.group_returns: Dict[str, pd.DataFrame] = {}
        self.sorted_factor_return: Optional[pd.DataFrame] = None

        self.factor_exposure: Optional[pd.DataFrame] = None
        self.ts_intercept: Optional[pd.Series] = None
        self.factor_exposure_t: Optional[pd.DataFrame] = None

        self.factor_coverage: Optional[pd.DataFrame] = None
        self.info_coef: Optional[pd.DataFrame] = None
        self.ic_direction: Optional[pd.Series] = None
        self.factor_corr: Optional[pd.DataFrame] = None

        self.factor_premia: Optional[pd.DataFrame] = None
        self.factor_premia_t: Optional[pd.DataFrame] = None
        self.factor_r2: Optional[pd.Series] = None

        self.fmb_premia: Optional[pd.Series] = None
        self.fmb_tstats: Optional[pd.Series] = None

        self.grs_stat: Optional[float] = None
        self.grs_pval: Optional[float] = None
        self.gmm_result: Optional[pd.Series] = None

        self.portfolio_attribution: Optional[Dict[str, object]] = None

    # ===========================
    # Input normalization
    # ===========================
    @staticmethod
    def _normalize_factor_input(
        factor_in: Union[pd.Series, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Return a MultiIndex([date, code]) DataFrame with columns = factor names.
        Single-factor Series: name is the factor name.
        """
        if isinstance(factor_in, pd.Series):
            if factor_in.name is None or str(factor_in.name).strip() == "":
                raise ValueError("Single-factor Series must have a non-empty name.")
            df = factor_in.to_frame(name=str(factor_in.name))
        elif isinstance(factor_in, pd.DataFrame):
            if factor_in.shape[1] == 0:
                raise ValueError("Factor DataFrame must have at least 1 column.")
            df = factor_in.copy()
            df.columns = [str(c) for c in df.columns]
        else:
            raise TypeError("factor must be a Series or DataFrame.")

        if not _is_mi_date_code_index(df.index):
            raise TypeError("factor index must be MultiIndex([date, code]).")

        # drop duplicated columns by making unique
        used = set()
        new_cols = []
        for c in df.columns:
            base = str(c)
            nm = base
            i = 1
            while nm in used:
                nm = f"{base}_{i}"
                i += 1
            used.add(nm)
            new_cols.append(nm)
        df.columns = new_cols
        return df

    # ===========================
    # Internal helpers
    # ===========================
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
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    @staticmethod
    def _white_covariance(
        X: np.ndarray, resid: np.ndarray, hc_type: str = "HC1"
    ) -> np.ndarray:
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
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        betas = np.full(
            (len(dates), x.shape[1] + int(intercept)), np.nan, dtype="float"
        )
        tstats = np.full(
            (len(dates), x.shape[1] + int(intercept)), np.nan, dtype="float"
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
            betas[j] = beta_vec.astype(np.float64)
            tstats[j] = t_vec.astype(np.float64)

        beta_s = pd.DataFrame(betas, index=dates)
        t_s = pd.DataFrame(tstats, index=dates)
        return beta_s, t_s

    # ===========================
    # Grouping and HL factor
    # ===========================
    def get_group_returns(
        self,
        n: int = 10,
        mode: Literal["single", "conditional", "independent"] = "single",
    ) -> "Evaluator":
        """
        Compute grouped portfolio returns and HL factor returns.
        """
        mode = str(mode).lower()
        if mode not in ("single", "independent", "conditional"):
            self._logger.error("Mode must be 'single', 'independent' or 'conditional'")
            return self

        self._logger.info(f"[group_returns] n={n} mode={mode}")

        # only keep rows with return
        base = self._panel.join(self._feasible).join(self._weight)
        base = base[base["ret"].notna() & base["feasible"].astype(bool)]

        if base.empty:
            self._logger.warning(
                "No data available after applying return non-NaN and feasible mask."
            )
            self.group_returns = {nm: pd.DataFrame() for nm in self._names}
            self.sorted_factor_return = pd.DataFrame(
                index=pd.Index([], name=self._date_name), columns=self._names
            )
            return self

        # per factor build returns
        self.group_returns = {}
        for name in self._names:
            other_names = [x for x in self._names if x != name]
            # per date compute group returns
            g_rets: Dict[pd.Timestamp, Union[pd.Series, pd.DataFrame]] = {}

            for dt, sub in base.groupby(level=0, sort=True):
                try:
                    # sub index: (dt, codes)
                    # keep rows with factor available
                    if sub[name].notna().sum() == 0:
                        # no eligible factor values
                        if mode == "single" or len(other_names) == 0:
                            g_rets[dt] = pd.Series(
                                {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                name=dt,
                            )
                        else:
                            g_rets[dt] = (
                                pd.Series(
                                    {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                    name=dt,
                                )
                                .to_frame()
                                .T
                            )
                        continue

                    r_t = sub["ret"]
                    w_t = sub["weight"]
                    f_t = sub[name]

                    if mode == "single" or len(other_names) == 0:
                        g_ret = pd.Series(
                            {f"{name}({i})": np.nan for i in range(1, n + 1)}, name=dt
                        )
                        g_t = self._qcut_groups(f_t, n)
                        if g_t.notna().sum() == 0:
                            g_rets[dt] = g_ret
                            continue
                        for gi in sorted(pd.unique(g_t.dropna().astype(int))):
                            mask = g_t == gi
                            g_ret[f"{name}({gi})"] = self._group_return(
                                r_t[mask], w_t[mask]
                            )
                        g_rets[dt] = g_ret
                        continue

                    # multi-factor controls
                    if mode == "independent":
                        # global qcut for each control factor and target factor
                        other_groups = []
                        ok = pd.Series(True, index=sub.index)
                        for on in other_names:
                            g_on = self._qcut_groups(sub[on], n)
                            other_groups.append(g_on.rename(on))
                            ok &= g_on.notna()
                        g_target = self._qcut_groups(f_t, n).rename("target_g")
                        ok &= g_target.notna()

                        if ok.sum() == 0:
                            g_rets[dt] = (
                                pd.Series(
                                    {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                    name=dt,
                                )
                                .to_frame()
                                .T
                            )
                            continue

                        tmp = (
                            pd.concat(
                                other_groups
                                + [g_target, r_t.rename("ret"), w_t.rename("w")],
                                axis=1,
                            )
                            .loc[ok]
                            .copy()
                        )

                        bucket_results: List[pd.Series] = []
                        # group by control buckets, compute within each cell target-bucket returns
                        for gn, sub_idx in tmp.groupby(other_names).groups.items():
                            if not isinstance(gn, tuple):
                                gn = (gn,)
                            cell = tmp.loc[sub_idx]
                            cell_name = "/".join(
                                [f"{on}({int(g)})" for on, g in zip(other_names, gn)]
                            )
                            sub_group_ret = pd.Series(
                                {i: np.nan for i in range(1, n + 1)}, name=cell_name
                            )
                            for gi in range(1, n + 1):
                                sel = cell["target_g"] == gi
                                if sel.sum() == 0:
                                    continue
                                sub_group_ret[gi] = self._group_return(
                                    cell.loc[sel, "ret"], cell.loc[sel, "w"]
                                )
                            bucket_results.append(sub_group_ret)

                        g_ret_df = pd.DataFrame(bucket_results)
                        g_ret_df.columns = [f"{name}({i})" for i in range(1, n + 1)]
                        g_rets[dt] = g_ret_df
                        continue

                    if mode == "conditional":
                        # hierarchical on control factors, then local qcut on target
                        buckets = [pd.Index(sub.index, name="")]
                        for on in other_names:
                            new_buckets: List[pd.Index] = []
                            s = sub[on]
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
                                                name=idxs.name + f"/{on}({label})",
                                            )
                                        )
                            buckets = new_buckets if len(new_buckets) > 0 else buckets

                        bucket_results: List[pd.Series] = []
                        for idxs in buckets:
                            if len(idxs) == 0:
                                continue
                            g_local = self._qcut_groups(f_t.loc[idxs], n)
                            if g_local.notna().sum() == 0:
                                continue
                            sub_group_ret = pd.Series(
                                {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                name=idxs.name[1:],
                            )
                            for gi in sorted(pd.unique(g_local.dropna().astype(int))):
                                sel = g_local == gi
                                sub_group_ret[f"{name}({gi})"] = self._group_return(
                                    r_t.loc[idxs][sel], w_t.loc[idxs][sel]
                                )
                            bucket_results.append(sub_group_ret)

                        if len(bucket_results) == 0:
                            g_rets[dt] = (
                                pd.Series(
                                    {f"{name}({i})": np.nan for i in range(1, n + 1)},
                                    name=dt,
                                )
                                .to_frame()
                                .T
                            )
                        else:
                            g_rets[dt] = pd.concat(bucket_results, axis=1).T
                        continue

                except Exception as e:
                    self._logger.error(f"get_group_returns error on {dt}: {e}")

            # concat per factor
            first = next(iter(g_rets.values())) if len(g_rets) else None
            if first is None:
                self.group_returns[name] = pd.DataFrame()
            else:
                axis = int(isinstance(first, pd.Series))
                df = pd.concat(g_rets.values(), keys=g_rets.keys(), axis=axis).dropna(
                    how="all", axis=1
                )
                if isinstance(first, pd.Series):
                    df = df.T  # date x groups
                self.group_returns[name] = df

        # build HL return per factor: sum top - sum bottom (summing across conditional/independent cells)
        hl_list = []
        for nm, sgr in self.group_returns.items():
            if sgr is None or len(sgr) == 0:
                hl = pd.Series(dtype=float, name=nm)
            else:
                if isinstance(sgr.index, pd.MultiIndex):
                    # independent/conditional: index may include date in level0
                    top = (
                        sgr.dropna(axis=0, how="all").iloc[:, -1].groupby(level=0).sum()
                    )
                    bot = (
                        sgr.dropna(axis=0, how="all").iloc[:, 0].groupby(level=0).sum()
                    )
                    hl = (top - bot).rename(nm)
                else:
                    hl = (sgr.iloc[:, -1] - sgr.iloc[:, 0]).rename(nm)
            hl_list.append(hl)

        self.sorted_factor_return = pd.concat(hl_list, axis=1)
        self._logger.info(f"Group returns computed: n={n}, mode={mode}")
        return self

    # ===========================
    # IC and factor correlations
    # ===========================
    def get_coverage(self) -> "Evaluator":
        num = self._panel.groupby("date").count()
        self.factor_coverage = num.iloc[:, :-1].div(num.iloc[:, -1], axis=0)
        return self

    def get_info_coef(self, method: str = "spearman") -> "Evaluator":
        """
        IC between factor exposures at date and provided future returns at date.
        horizon/skip_horizon kept for signature compatibility (ignored).
        """
        self._logger.info(f"[IC] method={method}")

        base = self._panel[self._panel["ret"].notna()]
        if base.empty:
            self.info_coef = pd.DataFrame(columns=self._names, dtype=float)
            self.ic_direction = pd.Series(dtype=float)
            return self

        ic_map: Dict[str, pd.Series] = {}
        for nm in self._names:
            # per date correlation between factor and ret
            def _corr(sub: pd.DataFrame) -> float:
                x = sub[nm]
                y = sub["ret"]
                ok = x.notna() & y.notna()
                if ok.sum() < 2:
                    return np.nan
                try:
                    return float(x[ok].corr(y[ok], method=method))
                except Exception:
                    return np.nan

            ic_s = base.groupby(level=0, sort=True).apply(_corr).rename(nm)
            ic_map[nm] = ic_s

        self.info_coef = pd.concat(ic_map.values(), axis=1)
        self.ic_direction = np.sign(self.info_coef.mean(axis=0))
        return self

    def get_correlation(self, method: str = "spearman") -> "Evaluator":
        self._logger.info(f"[Corr] method={method}")
        corr = self._panel.iloc[:, :-1].groupby("date").corr()
        corr.index.names = ["date", "factor"]
        self.factor_corr = corr
        return self

    # ===========================
    # Cross-sectional analytics
    # ===========================
    def cross_sectional_regression(
        self,
        add_intercept: bool = True,
        cov_type: Literal["none", "white"] = "white",
        white_type: str = "HC1",
    ) -> "Evaluator":
        """
        For each date: regress ret on factor exposures (cross-section).
        """
        self._logger.info(
            f"[CS reg] add_intercept={add_intercept}, cov_type={cov_type}"
        )

        base = self._panel.join(self._feasible).join(self._weight)
        base = base[base["ret"].notna() & base["feasible"].astype(bool)]

        dates = base.index.get_level_values(0).unique()
        default = np.full(len(self._names) + int(add_intercept), np.nan)

        betas, tstats, r2vals, used_dates = [], [], [], []
        for dt in dates:
            sub = base.xs(dt, level=0, drop_level=False)
            y = sub["ret"].values
            Xi = sub[self._names].values

            valid = np.isfinite(y) & np.all(np.isfinite(Xi), axis=1)
            min_req = Xi.shape[1] + (1 if add_intercept else 0) + 1
            if valid.sum() < min_req:
                self._logger.warning(
                    f"Valid asset ({valid.sum()}) < min_req ({min_req}) on {dt}"
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
                used_dates.append(dt)
                continue

            w = None
            wv = sub["weight"].values
            wv = np.where(valid, wv, 0.0)
            if (wv > 0).sum() > 0:
                w = wv[valid]

            beta, _, t, _, r2 = self._ols_fit(
                X=Xi[valid],
                y=y[valid],
                add_intercept=add_intercept,
                cov_type=cov_type,
                hc_type=white_type,
                weights=w,
            )

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
            used_dates.append(dt)

        self.factor_premia = pd.DataFrame(
            betas, index=pd.Index(used_dates, name=self._date_name)
        )
        self.factor_premia_t = pd.DataFrame(
            tstats, index=pd.Index(used_dates, name=self._date_name)
        )
        self.factor_r2 = pd.Series(
            r2vals, index=pd.Index(used_dates, name=self._date_name), name="R2_CS"
        )
        self._logger.info(
            f"Cross-sectional regression completed add_intercept={add_intercept}, cov_type={cov_type})"
        )
        return self

    def fama_macbeth(self, nw_lag: int = 3) -> "Evaluator":
        if self.factor_premia is None or len(self.factor_premia) == 0:
            self._logger.error(
                "Cross sectional regression not performed, run `cross_sectional_regression` first."
            )
            return self

        beta_ts = self.factor_premia.copy()  # dates x (intercept + K)
        premia = beta_ts.mean(axis=0)

        tstats = {}
        for k in beta_ts.columns:
            y = beta_ts[k].values
            x = np.ones((len(y), 1))
            _, _, t_vec, _, _ = self._ols_fit(
                X=x, y=y, add_intercept=False, cov_type="nw", nw_lag=nw_lag
            )
            tstats[k] = float(t_vec[0])

        self.fmb_premia = premia.rename("FMB_Premia")
        self.fmb_tstats = pd.Series(tstats, name="FMB_t")
        self._logger.info(f"Fama-MacBeth regression completed (nw_lag={nw_lag})")
        return self

    # ===========================
    # Time-series regression: R_{t,i} ~ F_t
    # ===========================
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
        n_jobs: int = -1,
    ) -> "Evaluator":
        """
        TS regressions of asset returns on factor returns (HL returns).
        Asset returns are from provided future (panel['ret']).
        """
        self._logger.info(
            f"[TS reg] horizon={horizon}, rolling={rolling}, cov_type={cov_type}"
        )

        if self.sorted_factor_return is None or len(self.sorted_factor_return) == 0:
            self._logger.error("No factor return found; run `get_group_returns` first.")
            return self

        # Build R (date x code) from long panel ret
        base = self._panel[["ret"]].copy()
        base["ret"] = base["ret"].where(self._feasible)

        R = base["ret"].unstack(level=1)  # date x code
        F = self.sorted_factor_return.copy()  # date x K
        idx = R.index.intersection(F.index)
        if len(idx) == 0:
            self._logger.error(
                "No overlapping dates between asset returns and factor returns."
            )
            return self

        R = R.loc[idx]
        F = F.loc[idx]

        if rolling:
            dates = idx
            assets = list(R.columns)
            x = F.values  # T x K

            def _asset_rolling(col: str):
                y = R[col].values
                beta_s, t_s = self._rolling_regression(
                    y=y,
                    x=x,
                    dates=dates,
                    window=window,
                    min_obs=min_obs,
                    intercept=add_intercept,
                )
                return beta_s, t_s

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_asset_rolling)(col) for col in assets
            )

            beta_df = pd.concat([r[0] for r in results], axis=0, keys=assets)
            t_df = pd.concat([r[1] for r in results], axis=0, keys=assets)

            beta_df.columns = (["intercept"] if add_intercept else []) + list(F.columns)
            t_df.columns = (["intercept"] if add_intercept else []) + list(F.columns)
            t_df = t_df.add_suffix("-t")

            self.factor_exposure = beta_df
            self.factor_exposure_t = t_df
            self._logger.info(
                f"Rolling TS regression completed: window={window}, min_obs={min_obs}, intercept={add_intercept}"
            )
            return self

        # Full-sample asset-wise regressions
        x = F.values  # T x K
        assets = list(R.columns)
        alpha_vals: Dict[str, float] = {}
        beta_vals: Dict[str, np.ndarray] = {}
        tstats: Dict[str, np.ndarray] = {}

        for asset in assets:
            y = R[asset].values
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
            else:
                alpha_vals[asset] = np.nan
                beta_vals[asset] = beta.astype(float)
            tstats[asset] = t.astype(float)

        self.factor_exposure = pd.DataFrame(beta_vals, index=F.columns).T
        self.ts_intercept = pd.Series(alpha_vals, name="intercept")
        self.factor_exposure_t = pd.DataFrame(
            tstats, index=(["intercept"] if add_intercept else []) + list(F.columns)
        ).T
        self._logger.info(
            f"Full-sample TS regression completed: intercept={add_intercept}, cov_type={cov_type}, nw_lag={nw_lag}"
        )
        return self

    def get_factor_exposure(
        self,
        horizon: int = 1,
        window: int = 252,
        min_obs: int = 60,
        intercept: bool = True,
        n_jobs: int = 1,
    ) -> "Evaluator":
        if self.sorted_factor_return is None or len(self.sorted_factor_return) == 0:
            self._logger.error("No factor return found; run `get_group_returns` first.")
            return self

        self.time_series_regression(
            horizon=horizon,
            rolling=True,
            window=window,
            min_obs=min_obs,
            add_intercept=intercept,
            cov_type="none",
            n_jobs=n_jobs,
        )
        return self

    # ===========================
    # GRS / GMM pricing
    # ===========================
    def grs_test(
        self,
        horizon: int = 1,
        skip_horizon: bool = True,
        add_intercept: bool = True,
    ) -> "Evaluator":
        self._logger.info(
            f"[GRS] horizon={horizon}, skip_horizon={skip_horizon}, add_intercept={add_intercept}"
        )

        if self.sorted_factor_return is None or len(self.sorted_factor_return) == 0:
            self._logger.error("No factor return found; run `get_group_returns` first.")
            return self

        R = self._panel["ret"].unstack(level=1)  # date x N
        F = self.sorted_factor_return  # date x K
        idx = R.index.intersection(F.index)
        if len(idx) == 0:
            self._logger.error(
                "No overlapping dates between asset returns and factor returns."
            )
            return self

        Rm = R.loc[idx].values
        Fm = F.loc[idx].values
        Tn, N = Rm.shape
        K = Fm.shape[1]

        # TS regression for each asset with plain OLS (as original)
        alphas, residuals = [], []
        X = Fm
        X_int = self._add_intercept(X) if add_intercept else X

        for i in range(N):
            y = Rm[:, i]
            beta, _, _, _, _ = self._ols_fit(
                X=X, y=y, add_intercept=add_intercept, cov_type="none"
            )
            if add_intercept:
                alphas.append(beta[0])
                resid = y - (X_int @ beta)
            else:
                # if no intercept, alpha=0
                alphas.append(0.0)
                resid = y - (X @ beta)
            residuals.append(resid)

        a = np.array(alphas).reshape(-1, 1)  # N x 1
        Eps = np.column_stack(residuals)  # T x N
        Sigma_e = np.cov(Eps, rowvar=False, ddof=1)  # N x N
        mu_F = Fm.mean(axis=0).reshape(-1, 1)  # K x 1
        Sigma_F = np.cov(Fm, rowvar=False, ddof=1)  # K x K

        try:
            Sigma_e_inv = np.linalg.inv(Sigma_e)
            Sigma_F_inv = np.linalg.inv(Sigma_F)
        except np.linalg.LinAlgError:
            self._logger.error("GRS test failed: singular covariance.")
            self.grs_stat, self.grs_pval = np.nan, np.nan
            return self

        top = (Tn - N - K) / N if (Tn - N - K) > 0 else Tn / N
        denom = 1.0 + float(mu_F.T @ Sigma_F_inv @ mu_F)
        grs = float(top * (a.T @ Sigma_e_inv @ a) / denom)

        df1 = N
        df2 = Tn - N - K if (Tn - N - K) > 0 else max(Tn - K - 1, 1)
        try:
            p_val = float(1.0 - stats.f.cdf(grs, df1=df1, df2=df2))
        except Exception:
            p_val = np.nan

        self.grs_stat = grs
        self.grs_pval = p_val
        return self

    def gmm_linear_pricing(
        self,
        horizon: int = 1,
        skip_horizon: bool = True,
        two_step: bool = True,
    ) -> "Evaluator":
        self._logger.info(
            f"[GMM] horizon={horizon}, skip_horizon={skip_horizon}, two_step={two_step}"
        )

        if self.sorted_factor_return is None or len(self.sorted_factor_return) == 0:
            self._logger.error("No factor return found; run `get_group_returns` first.")
            return self

        R = self._panel["ret"].unstack(level=1)
        F = self.sorted_factor_return.copy()
        idx = R.index.intersection(F.index)
        if len(idx) == 0:
            self._logger.error(
                "No overlapping dates between asset returns and factor returns."
            )
            return self

        Rm = R.loc[idx].values  # T x N
        Fm = F.loc[idx].values  # T x K
        Tn, N = Rm.shape
        K = Fm.shape[1]

        def g_lambda(lmbd: np.ndarray) -> np.ndarray:
            mt = 1.0 - Fm @ lmbd
            return (Rm * mt[:, None]).mean(axis=0)

        A = np.einsum("ti,tk->ik", Rm, Fm) / Tn  # N x K
        b = Rm.mean(axis=0).reshape(N, 1)

        try:
            lambda_1 = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)
        except Exception:
            lambda_1 = np.zeros(K)

        if two_step:
            mt = 1.0 - Fm @ lambda_1
            moments_t = Rm * mt[:, None]
            S = np.cov(moments_t, rowvar=False, ddof=1)
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

        g_hat = g_lambda(lambda_hat).reshape(-1, 1)
        J = float(Tn * (g_hat.T @ W @ g_hat))
        df = max(N - K, 1)
        try:
            pval = float(1.0 - stats.chi2.cdf(J, df=df))
        except Exception:
            pval = np.nan

        self.gmm_result = pd.Series(
            {"lambda": lambda_hat, "J": J, "pval": pval, "W": W}, name="GMM"
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
        try:
            n, p = X.shape
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

import os
import dotenv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional, Literal

import numpy as np
import pandas as pd

from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule

import quool
from parquool import setup_logger

from factool import DuckPQSource, Evaluator

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

dotenv.load_dotenv()


FactorPath = str
OneTestSpec = Union[
    FactorPath,  # "table/factor" (single factor)
    List[FactorPath],  # ["t/f1","t/f2"] (multi-factor one test)
    Tuple[FactorPath, ...],  # ("t/f1","t/f2") (multi-factor one test)
]
MultiTestSpec = Union[
    OneTestSpec,
    List[FactorPath],  # multiple single-factor tests OR one multi-factor test
    List[Tuple[FactorPath, ...]],  # multiple multi-factor tests
]


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        return float(np.asarray(x))
    except Exception:
        return np.nan


def get_trade_mask(
    data_source: DuckPQSource, begin: str, end: str, min_list_days: int = 90
):
    sql = f"""
    SELECT
        q.date AS date,
        q.code AS code,
        (
            q.high > q.limit_down
            AND q.low  < q.limit_up
            AND COALESCE(q.st, false) = false
            AND COALESCE(q.suspended, false) = false
            AND datediff('day', i.listed_date, q.date) > {min_list_days}
        ) AS tradable_mask
    FROM quotes_day AS q
    JOIN instruments_info AS i
        ON q.code = i.code
    WHERE q.date >= '{begin}' AND q.date <= '{end}'
    """

    data = data_source.query(sql)
    data["date"] = pd.to_datetime(data["date"])

    feasible = data.set_index(["date", "code"]).sort_index()
    return feasible


@dataclass
class BacktestParams:
    factor_paths: List[str]
    begin: str
    end: str
    target_path: str = "target/open"
    horizon: int = 5
    baseline_factors: Optional[Union[str, List[str]]] = None

    # Tradability mask and weight
    min_list_days: int = 90
    weight_path: str = None

    # IC and grouping
    ic_method: str = "spearman"
    n_groups: int = 10
    bucketing_mode: str = "single"

    # Cross-sectional regression
    cs_add_intercept: bool = True
    cs_cov_type: str = "white"
    cs_white_type: str = "HC1"

    # IC stability settings
    ic_roll_window: int = 60
    ic_acf_lags: int = 10
    ic_break_k: int = (
        1  # number of candidate breakpoints to test (simple single-break scan)
    )

    # Monotonicity settings
    monotonicity_use_excess: bool = (
        False  # if True, use group returns minus cross-sectional mean
    )


def grenerate_test_key(param: BacktestParams) -> str:
    paths = (
        param.factor_paths
        if isinstance(param.factor_paths, list)
        else [param.factor_paths]
    )
    bases = []
    if param.baseline_factors is not None:
        bases = (
            param.baseline_factors
            if isinstance(param.baseline_factors, list)
            else [param.baseline_factors]
        )
    return "+".join([path.split("/", 1)[-1] for path in paths]) + (
        ""
        if not bases
        else (f"__" + "+".join([base.split("/", 1)[-1] for base in bases]))
    )


def newey_west_tstat(x: pd.Series, lags: Optional[int] = None) -> float:
    """
    Newey-West t-stat for mean(x) with HAC covariance.
    Uses statsmodels OLS of x on constant.
    """
    s = pd.Series(x).dropna()
    if s.shape[0] < 10:
        return np.nan
    y = s.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X)
    if lags is None:
        # Common rule-of-thumb
        lags = int(np.floor(4 * (len(y) / 100) ** (2 / 9)))
    try:
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(lags)})
        return float(fit.tvalues[0])
    except Exception:
        return np.nan


def ic_summary_stats(ic: pd.Series) -> Dict[str, Any]:
    """
    IC summary including win rate, ICIR, tails, and HAC t-stat.
    """
    s = pd.Series(ic).dropna()
    n = int(s.shape[0])
    if n == 0:
        return {
            "ic_n": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_t": np.nan,
            "ic_t_nw": np.nan,
            "ic_win_rate": np.nan,
            "ic_p05": np.nan,
            "ic_p50": np.nan,
            "ic_p95": np.nan,
            "ic_skew": np.nan,
            "ic_kurt": np.nan,
            "ic_tail_gt_2std": np.nan,
        }

    mu = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else np.nan
    ir = mu / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    t = mu / (sd / np.sqrt(n)) if sd and np.isfinite(sd) and sd > 0 else np.nan
    t_nw = newey_west_tstat(s)

    win_rate = float((s > 0).mean())
    p05, p50, p95 = [float(v) for v in np.nanpercentile(s.values, [5, 50, 95])]
    skew = float(stats.skew(s.values, nan_policy="omit")) if n >= 3 else np.nan
    kurt = (
        float(stats.kurtosis(s.values, fisher=True, nan_policy="omit"))
        if n >= 4
        else np.nan
    )
    tail_gt_2std = float((np.abs(s - mu) > 2 * sd).mean()) if sd and sd > 0 else np.nan

    return {
        "ic_n": n,
        "ic_mean": mu,
        "ic_std": sd,
        "ic_ir": ir,
        "ic_t": t,
        "ic_t_nw": t_nw,
        "ic_win_rate": win_rate,
        "ic_p05": p05,
        "ic_p50": p50,
        "ic_p95": p95,
        "ic_skew": skew,
        "ic_kurt": kurt,
        "ic_tail_gt_2std": tail_gt_2std,
    }


def ic_stability_tests(
    ic: pd.Series, roll_window: int, acf_lags: int, break_k: int = 1
) -> Dict[str, Any]:
    """
    Stability/drift tests:
    - Rolling mean/IR variability
    - ACF(1) and Ljung-Box p-value (autocorrelation)
    - Simple single-break scan (max abs difference in pre/post mean), report best split and p-value
    """
    s = pd.Series(ic).dropna()
    out: Dict[str, Any] = {}
    if s.shape[0] < max(50, roll_window + 5):
        out.update(
            {
                "ic_roll_mean_std": np.nan,
                "ic_roll_ir_std": np.nan,
                "ic_acf1": np.nan,
                "ic_lb_pvalue": np.nan,
                "ic_best_break_date": None,
                "ic_break_stat": np.nan,
                "ic_break_pvalue": np.nan,
            }
        )
        return out

    roll_mean = s.rolling(roll_window).mean()
    roll_std = s.rolling(roll_window).std(ddof=1)
    roll_ir = roll_mean / roll_std

    out["ic_roll_mean_std"] = safe_float(roll_mean.dropna().std(ddof=1))
    out["ic_roll_ir_std"] = safe_float(roll_ir.dropna().std(ddof=1))

    # Autocorrelation tests
    try:
        out["ic_acf1"] = safe_float(s.autocorr(lag=1))
    except Exception:
        out["ic_acf1"] = np.nan

    try:
        lb = acorr_ljungbox(s.values, lags=[min(acf_lags, len(s) - 1)], return_df=True)
        out["ic_lb_pvalue"] = safe_float(lb["lb_pvalue"].iloc[-1])
    except Exception:
        out["ic_lb_pvalue"] = np.nan

    # Simple best single break scan (choose split that maximizes |mean1 - mean2|)
    idx = s.index
    values = s.values
    n = len(s)
    min_seg = max(30, n // 10)
    best = {"stat": -np.inf, "split": None, "pvalue": np.nan}

    # Candidate splits
    split_candidates = list(range(min_seg, n - min_seg))
    if break_k is not None and break_k > 0 and len(split_candidates) > 0:
        # To limit compute, subsample candidates
        step = max(1, len(split_candidates) // 200)
        split_candidates = split_candidates[::step]

    for k in split_candidates:
        a = values[:k]
        b = values[k:]
        if len(a) < min_seg or len(b) < min_seg:
            continue
        # Two-sample t-test (Welch)
        stat, pval = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        diff = np.nanmean(a) - np.nanmean(b)
        score = np.abs(diff)
        if np.isfinite(score) and score > best["stat"]:
            best = {"stat": score, "split": k, "pvalue": pval}

    if best["split"] is None:
        out["ic_best_break_date"] = None
        out["ic_break_stat"] = np.nan
        out["ic_break_pvalue"] = np.nan
    else:
        out["ic_best_break_date"] = str(idx[best["split"]].date())
        out["ic_break_stat"] = safe_float(best["stat"])
        out["ic_break_pvalue"] = safe_float(best["pvalue"])

    return out


def _extract_factor_group_columns(
    group_returns: pd.DataFrame, factor_name: str, n_groups: int
) -> List[str]:
    return [f"{factor_name}({i + 1})" for i in range(n_groups)]


def group_monotonicity_tests(
    e: Evaluator,
    factor_name: str,
    n_groups: int,
    use_excess: bool = False,
) -> Dict[str, Any]:
    """
    Uses evaluator's group returns time series.
    Tests:
    - Spearman correlation between group index and group mean returns (per date), then t-test over time.
    - Per-date OLS slope of group returns on group index; report mean slope and NW t-stat.
    - Jonckheere–Terpstra-like proxy using Kendall tau between group index and returns (per date).
      (True JT test requires raw observations; here we approximate using ordinal association.)
    """
    gr = e.group_returns.get(factor_name)
    if gr is None or gr.empty:
        return {
            "mono_spearman_mean": np.nan,
            "mono_spearman_t": np.nan,
            "mono_spearman_win_rate": np.nan,
            "mono_slope_mean": np.nan,
            "mono_slope_t_nw": np.nan,
            "mono_kendall_mean": np.nan,
            "mono_kendall_t": np.nan,
        }

    # Convert to date x group matrix
    gr_mat = gr.groupby(level=0).mean()
    cols = _extract_factor_group_columns(gr_mat, factor_name, n_groups)
    gr_mat = gr_mat[cols].copy()

    if use_excess:
        gr_mat = gr_mat.sub(gr_mat.mean(axis=1), axis=0)

    x = np.arange(1, n_groups + 1)

    spears = []
    slopes = []
    kendalls = []

    for dt, row in gr_mat.iterrows():
        y = row.values.astype(float)
        if np.all(np.isnan(y)):
            continue
        if np.sum(np.isfinite(y)) < max(5, n_groups // 2):
            continue

        r_s, _ = stats.spearmanr(x, y, nan_policy="omit")
        spears.append(r_s)

        # OLS slope of y on x
        mask = np.isfinite(y)
        if mask.sum() >= 3:
            X = sm.add_constant(x[mask])
            fit = sm.OLS(y[mask], X).fit()
            slopes.append(fit.params[1])
        else:
            slopes.append(np.nan)

        r_k, _ = stats.kendalltau(x, y, nan_policy="omit")
        kendalls.append(r_k)

    spears = pd.Series(spears).dropna()
    slopes = pd.Series(slopes).dropna()
    kendalls = pd.Series(kendalls).dropna()

    def mean_t(s: pd.Series) -> Tuple[float, float]:
        if len(s) < 30:
            return (safe_float(s.mean()), np.nan)
        mu = float(s.mean())
        sd = float(s.std(ddof=1))
        t = mu / (sd / np.sqrt(len(s))) if sd > 0 else np.nan
        return mu, t

    spearman_mean, spearman_t = mean_t(spears)
    kendall_mean, kendall_t = mean_t(kendalls)
    slope_mean = safe_float(slopes.mean())
    slope_t_nw = newey_west_tstat(slopes)

    return {
        "mono_spearman_mean": spearman_mean,
        "mono_spearman_t": spearman_t,
        "mono_spearman_win_rate": (
            safe_float((spears > 0).mean()) if len(spears) else np.nan
        ),
        "mono_slope_mean": slope_mean,
        "mono_slope_t_nw": slope_t_nw,
        "mono_kendall_mean": kendall_mean,
        "mono_kendall_t": kendall_t,
    }


def pairwise_corr_and_stability(
    test_factor: pd.DataFrame,
    base_factor: pd.DataFrame,
    feasible_mask: Optional[pd.DataFrame] = None,
    roll_window: int = 120,
) -> Dict[str, Any]:
    """
    Computes cross-sectional correlation between two factors per date, then summarizes:
    - mean corr, std corr
    - rolling mean corr std (stability)
    """
    f1 = test_factor
    f2 = base_factor

    if feasible_mask is not None:
        mask = feasible_mask.reindex_like(f1).astype(bool)
        f1 = f1.where(mask)
        f2 = f2.where(mask)

    cors = []
    for dt in f1.index:
        a = f1.loc[dt]
        b = f2.loc[dt]
        valid = a.notna() & b.notna()
        if valid.sum() < 30:
            cors.append(np.nan)
            continue
        cors.append(a[valid].corr(b[valid], method="spearman"))
    s = pd.Series(cors, index=f1.index).dropna()

    if s.empty:
        return {"corr_mean": np.nan, "corr_std": np.nan, "corr_roll_mean_std": np.nan}

    roll = s.rolling(roll_window).mean()
    return {
        "corr_mean": safe_float(s.mean()),
        "corr_std": safe_float(s.std(ddof=1)),
        "corr_roll_mean_std": safe_float(roll.dropna().std(ddof=1)),
    }


def cs_r2_uplift(
    e_test: Evaluator,
    e_base: Optional[Evaluator],
) -> Dict[str, Any]:
    """
    Uses evaluator cross-sectional regression outputs:
    - base R2 (if provided)
    - test R2 (model with test factor(s))
    - uplift: test - base
    """
    out = {"cs_r2_mean": np.nan, "cs_r2_base_mean": np.nan, "cs_r2_uplift": np.nan}

    try:
        r2_test = e_test.factor_r2
        if isinstance(r2_test, pd.DataFrame):
            # If multiple columns exist, use the first numeric column
            r2_test_series = r2_test.select_dtypes(include=[np.number]).iloc[:, 0]
        else:
            r2_test_series = pd.Series(r2_test)
        out["cs_r2_mean"] = safe_float(pd.Series(r2_test_series).dropna().mean())
    except Exception:
        out["cs_r2_mean"] = np.nan

    if e_base is not None:
        try:
            r2_base = e_base.factor_r2
            if isinstance(r2_base, pd.DataFrame):
                r2_base_series = r2_base.select_dtypes(include=[np.number]).iloc[:, 0]
            else:
                r2_base_series = pd.Series(r2_base)
            out["cs_r2_base_mean"] = safe_float(
                pd.Series(r2_base_series).dropna().mean()
            )
        except Exception:
            out["cs_r2_base_mean"] = np.nan

    if np.isfinite(out["cs_r2_mean"]) and np.isfinite(out["cs_r2_base_mean"]):
        out["cs_r2_uplift"] = out["cs_r2_mean"] - out["cs_r2_base_mean"]

    return out


def load_factors_for_one_test(test_cfg: List[Dict[str, str]]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for c in test_cfg:
        df = factor_source.get_factor(
            table=c["table_name"],
            name=c["factor_name"],
            begin=params.begin,
            end=params.end,
        ).sort_index()
        dfs.append(df)
    return dfs


def run(
    factor_source: DuckPQSource,
    data_source: DuckPQSource,
    backtest_params: BacktestParams,
) -> Dict[str, Any]:
    logger = setup_logger("Evaluator")

    factor_data = factor_source.load(
        backtest_params.factor_paths,
        begin=backtest_params.begin,
        end=backtest_params.end,
    )
    logger.info(f"Loaded factor_data {factor_data.shape}")
    future_return = factor_source.load(
        backtest_params.target_path,
        begin=backtest_params.begin,
        end=backtest_params.end,
        pad_end=backtest_params.horizon + 1,
    ).iloc[:, 0]
    future_return = (
        future_return.groupby("code").shift(-1)
        / future_return.groupby("code").shift(-1 - backtest_params.horizon)
        - 1
    ).loc[backtest_params.begin : backtest_params.end]
    logger.info(f"Loaded future_return {future_return.shape}")

    weight = None
    if backtest_params.weight_path is not None:
        weight = factor_source.load(
            backtest_params.weight_path,
            begin=backtest_params.begin,
            end=backtest_params.end,
        )
    feasible = get_trade_mask(
        data_source=data_source,
        begin=backtest_params.begin,
        end=backtest_params.end,
        min_list_days=backtest_params.min_list_days,
    )
    logger.info(f"Feasible and weighted loaded")

    evaluator = Evaluator(
        factor=factor_data,
        future=future_return,
        weight=weight,
        feasible=feasible,
    )

    result: Dict[str, Any] = {}
    factor_names = factor_data.columns.to_list()

    # Coverage
    evaluator.get_coverage()
    result["mean_coverage"] = evaluator.factor_coverage.mean()

    # Correlation
    evaluator.get_correlation(backtest_params.ic_method)
    result["corr_mean"] = evaluator.factor_corr.groupby(level="factor").mean()
    result["corr_std"] = evaluator.factor_corr.groupby(level="factor").std()

    # IC and IC tests
    evaluator.get_info_coef(
        method=backtest_params.ic_method,
    )
    ic_df = evaluator.info_coef.copy()
    result["ic_raw"] = ic_df

    ic_summ_rows = []
    ic_stab_rows = []
    for col in ic_df.columns:
        s = ic_df[col]
        summ = ic_summary_stats(s)
        stab = ic_stability_tests(
            s,
            backtest_params.ic_roll_window,
            backtest_params.ic_acf_lags,
            backtest_params.ic_break_k,
        )
        ic_summ_rows.append(pd.Series(summ, name=col))
        ic_stab_rows.append(pd.Series(stab, name=col))

    result["ic_summary"] = pd.DataFrame(ic_summ_rows)
    result["ic_stability"] = pd.DataFrame(ic_stab_rows)

    # Group returns
    evaluator.get_group_returns(
        n=backtest_params.n_groups,
        mode=backtest_params.bucketing_mode,
    )

    factor_return = evaluator.sorted_factor_return
    group_returns = pd.concat(
        [gr.groupby(level=0).mean() for gr in evaluator.group_returns.values()]
        + [factor_return],
        axis=1,
    )
    group_values = (group_returns.fillna(0) + 1).cumprod()
    group_values.index.name = "date"
    group_performance = group_values.apply(quool.Evaluator.evaluate)

    group_returns_mean = group_returns.mean()
    group_returns_t = group_returns_mean / (
        group_returns.std(ddof=1) / np.sqrt(group_returns.shape[0])
    )
    result["group_return_summary"] = pd.concat(
        [group_returns_mean, group_returns_t],
        axis=1,
        keys=["mean", "t"],
    )
    result["group_values"] = group_values
    result["group_performance"] = group_performance

    # Monotonicity tests (per factor)
    mono_rows = []
    for factor_name in factor_names:
        mono = group_monotonicity_tests(
            evaluator,
            factor_name=factor_name,
            n_groups=backtest_params.n_groups,
            use_excess=backtest_params.monotonicity_use_excess,
        )
        mono_rows.append(pd.Series(mono, name=factor_name))
    result["monotonicity"] = pd.DataFrame(mono_rows)

    # Cross-sectional regression (test model)
    evaluator.cross_sectional_regression(
        add_intercept=backtest_params.cs_add_intercept,
        cov_type=backtest_params.cs_cov_type,
        white_type=backtest_params.cs_white_type,
    )
    result["cs_premia"] = evaluator.factor_premia
    result["cs_premia_t"] = evaluator.factor_premia_t
    result["cs_r2"] = evaluator.factor_r2

    # Incremental tests vs baseline factors
    incremental = {}
    if backtest_params.baseline_factors:
        # Build baseline evaluator (regression with baseline factors only)
        baseline_factor_data = factor_source.load(
            backtest_params.baseline_factors,
            begin=backtest_params.begin,
            end=backtest_params.end,
        )
        full_data = pd.concat([factor_data, baseline_factor_data], axis=1)
        evaluator_base = Evaluator(
            factor=full_data, future=future_return, feasible=feasible, weight=weight
        )
        evaluator_base.cross_sectional_regression(
            add_intercept=backtest_params.cs_add_intercept,
            cov_type=backtest_params.cs_cov_type,
            white_type=backtest_params.cs_white_type,
        )

        # Correlation and stability
        evaluator_base.get_correlation(method=backtest_params.ic_method)
        corr_mean = evaluator_base.factor_corr.groupby("factor").mean()
        corr_std = evaluator_base.factor_corr.groupby("factor").std()
        incremental["corr_mean"] = corr_mean
        incremental["corr_std"] = corr_std

        # R2 uplift (test vs baseline)
        incremental["r2_uplift"] = pd.Series(
            cs_r2_uplift(e_test=evaluator, e_base=evaluator_base), name="r2_uplift"
        )

    else:
        incremental["corr_mean"] = pd.DataFrame()
        incremental["corr_std"] = pd.DataFrame()
        incremental["r2_uplift"] = pd.Series(
            {
                "cs_r2_mean": (
                    safe_float(
                        pd.Series(
                            evaluator.factor_r2.select_dtypes(include=[np.number]).iloc[
                                :, 0
                            ]
                        ).mean()
                    )
                    if isinstance(evaluator.factor_r2, pd.DataFrame)
                    else np.nan
                ),
                "cs_r2_base_mean": np.nan,
                "cs_r2_uplift": np.nan,
            },
            name="r2_uplift",
        )

    result["incremental"] = incremental

    return result


def _to_1col_df(x, col_name: str) -> pd.DataFrame:
    """Series/array-like -> DataFrame(1 col). If already DF, return as-is."""
    if x is None:
        return pd.DataFrame()
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, pd.Series):
        return x.to_frame(col_name)
    # scalar
    if np.isscalar(x):
        return pd.DataFrame({col_name: [x]})
    # list/np array
    return pd.DataFrame({col_name: list(x)})


def save(save_path: Union[Path, str], results: Dict[str, Dict]):
    blocks: List[pd.DataFrame] = []

    for test_key, res in results.items():
        cov = res.get("mean_coverage")  # often Series
        ic_summ = res.get("ic_summary")  # DF
        ic_stab = res.get("ic_stability")  # DF
        mono = res.get("monotonicity")  # DF
        inc_r2 = res.get("incremental", {}).get("r2_uplift")  # dict or DF

        cov_df = _to_1col_df(cov, "coverage")

        ic_summ_df = (
            ic_summ.copy() if isinstance(ic_summ, pd.DataFrame) else pd.DataFrame()
        )
        ic_stab_df = (
            ic_stab.copy() if isinstance(ic_stab, pd.DataFrame) else pd.DataFrame()
        )
        mono_df = mono.copy() if isinstance(mono, pd.DataFrame) else pd.DataFrame()

        if isinstance(inc_r2, pd.DataFrame):
            inc_r2_df = inc_r2.copy()
        elif isinstance(inc_r2, dict):
            inc_r2_df = pd.DataFrame([inc_r2])
        else:
            inc_r2_df = pd.DataFrame()

        base_index = None
        for df_ in [ic_summ_df, ic_stab_df, mono_df, cov_df]:
            if isinstance(df_, pd.DataFrame) and len(df_) > 0:
                base_index = df_.index
                break
        if base_index is None:
            continue

        big = pd.concat(
            [cov_df, ic_summ_df, ic_stab_df, mono_df],
            axis=1,
            join="outer",
        )

        if len(inc_r2_df) > 0:
            inc_row = inc_r2_df.iloc[0]
            for c in inc_row.index:
                big[c] = inc_row[c]

        big.insert(0, "test_key", test_key)

        big = big.reset_index().rename(columns={"index": "factor"})

        blocks.append(big)

    if not blocks:
        return pd.DataFrame()

    out = pd.concat(blocks, axis=0, ignore_index=True)
    out.index = pd.RangeIndex(1, len(out) + 1, step=1, name="index")

    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="summary")
        for i, (test_key, res) in enumerate(results.items(), start=1):
            if not res["incremental"]["corr_mean"].empty:
                res["incremental"]["corr_mean"].to_excel(
                    writer, sheet_name=f"factor_correlation_{i}"
                )
            else:
                res["corr_mean"].to_excel(writer, sheet_name=f"factor_correlation_{i}")
            res["ic_raw"].to_excel(writer, sheet_name=f"info_coef_{i}")
            res["group_values"].to_excel(writer, sheet_name=f"group_value_{i}")
            res["group_performance"].to_excel(
                writer, sheet_name=f"group_performance_{i}"
            )
            if not res["incremental"]["corr_mean"].empty:
                res["incremental"]["r2_uplift"].to_excel(
                    writer, sheet_name=f"r2_uplift_{i}"
                )

            # Drawing charts in original excel file
            wb = writer.book

            # Correlation rule color
            ws = wb[f"factor_correlation_{i}"]
            min_row, min_col = 2, 2
            max_row, max_col = ws.max_row, ws.max_column
            rule = ColorScaleRule(
                start_type="num",
                start_value=-1,
                start_color="2F5597",
                mid_type="num",
                mid_value=0,
                mid_color="FFFFFF",
                end_type="num",
                end_value=1,
                end_color="C00000",
            )
            cell_range = f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"
            ws.conditional_formatting.add(cell_range, rule)
            ws.freeze_panes = "B2"

            # IC bar chart
            ws = wb[f"info_coef_{i}"]
            chart = BarChart()
            chart.type = "col"
            chart.title = f"Information Coefficient"
            chart.y_axis.title = "IC"
            chart.x_axis.title = "Date"

            data = Reference(ws, min_col=2, min_row=1, max_row=ws.max_row)
            cats = Reference(ws, min_col=1, min_row=2, max_row=ws.max_row)
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            chart.gapWidth = 50
            ws.add_chart(chart, "E2")

            # Group value line chart
            ws = wb[f"group_value_{i}"]
            max_row = ws.max_row
            max_col = ws.max_column

            chart = LineChart()
            chart.title = f"Group Net Value"
            chart.y_axis.title = "Net Value"

            cats = Reference(ws, min_col=1, min_row=2, max_row=max_row)

            data = Reference(ws, min_col=2, min_row=1, max_col=max_col, max_row=max_row)

            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)

            ws.add_chart(chart, "N2")


if __name__ == "__main__":
    DATASET_PATH = os.getenv("DATASET_PATH")
    FACTOR_DATA_PATH = os.getenv("FACTOR_DATA_PATH")
    if not DATASET_PATH or not FACTOR_DATA_PATH:
        raise EnvironmentError(
            "Missing env vars: DATASET_PATH and/or FACTOR_DATA_PATH. "
            "Please set them (e.g. in .env) before running."
        )
    data_source = DuckPQSource(Path(DATASET_PATH))
    data_source.register("quotes_day")
    data_source.register("instruments_info")

    params = [
        BacktestParams(
            factor_paths="barra_momentum/barra_mom_st_63d",  # CONFIG: Placeholder for factor paths
            baseline_factors="barra_size/mcap_float_a",
            begin="2015-01-01",
            end="2025-06-30",
        )
    ]
    output_path = "factor_test.xlsx"  # CONFIG: Placeholder for output path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    factor_source = DuckPQSource(Path(FACTOR_DATA_PATH))
    results = {}
    for param in params:
        result = run(factor_source, data_source, param)
        results[grenerate_test_key(param)] = result
    save(output_path, results)

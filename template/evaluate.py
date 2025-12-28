# %%
# Evaluation Script for Factor Testing
import os
import dotenv
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quool
from IPython.display import display, display_markdown, Markdown

from factool import DuckPQSource, Evaluator

# Optional but recommended for some stats tests
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

dotenv.load_dotenv()


# %%
# Utilities: parsing factor paths and normalizing test configs
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


def normalize_tests(
    tests: MultiTestSpec,
    *,
    sep: str = "/",
    treat_list_str_as: str = "many_single",  # "many_single" or "one_multi"
) -> List[List[Dict[str, str]]]:
    """
    Normalize input into: list_of_tests, where each test is a list of factor configs:
        [
          [ {"table_name":..., "factor_name":...}, ... ],  # test1 (1..n factors)
          [ {"table_name":..., "factor_name":...}, ... ],  # test2
        ]
    """
    if isinstance(tests, list) and tests and all(isinstance(x, tuple) for x in tests):
        out: List[List[Dict[str, str]]] = []
        for tup in tests:
            if not all(isinstance(p, str) for p in tup):
                raise TypeError("list[tuple] must be list[tuple[str,...]]")
            one: List[Dict[str, str]] = []
            for p in tup:
                table, factor = parse_factor_path(p, sep=sep)
                one.append({"table_name": table, "factor_name": factor})
            out.append(one)
        return out

    if isinstance(tests, tuple):
        if not all(isinstance(x, str) for x in tests):
            raise TypeError("tuple must be tuple[str,...]")
        one = []
        for p in tests:
            table, factor = parse_factor_path(p, sep=sep)
            one.append({"table_name": table, "factor_name": factor})
        return [one]

    if isinstance(tests, str):
        table, factor = parse_factor_path(tests, sep=sep)
        return [[{"table_name": table, "factor_name": factor}]]

    if isinstance(tests, list):
        if not tests:
            return []
        if not all(isinstance(x, str) for x in tests):
            raise TypeError("list must be list[str] or list[tuple[str,...]]")
        if treat_list_str_as == "one_multi":
            one = []
            for p in tests:
                table, factor = parse_factor_path(p, sep=sep)
                one.append({"table_name": table, "factor_name": factor})
            return [one]
        if treat_list_str_as == "many_single":
            out = []
            for p in tests:
                table, factor = parse_factor_path(p, sep=sep)
                out.append([{"table_name": table, "factor_name": factor}])
            return out
        raise ValueError("treat_list_str_as must be 'many_single' or 'one_multi'")

    raise TypeError(f"unsupported tests type: {type(tests)}")


def factor_path_from_cfg(c: Dict[str, str]) -> str:
    return f'{c["table_name"]}/{c["factor_name"]}'


def pretty_test_key(test_cfg: List[Dict[str, str]]) -> str:
    """
    Test key rule:
    - Single factor: "table/factor"
    - Multi-factor: "t/f1+t/f2+..."
    """
    paths = [factor_path_from_cfg(c) for c in test_cfg]
    if len(paths) == 1:
        return paths[0]
    return "+".join(paths)


def safe_float(x: Any) -> float:
    try:
        if x is None:
            return np.nan
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        return float(np.asarray(x))
    except Exception:
        return np.nan


# %%
# Parameters: backtest settings and new tests configuration
@dataclass
class BacktestParams:
    factors: Union[str, List[str], Tuple, List[Tuple]] = ""  # CONFIG: Placeholder for factor paths
    treat_list_str_as: str = "many_single"  # "many_single" or "one_multi"
    begin: str = "2015-01-01"
    end: str = "2025-06-30"
    ptype: str = "open_post"
    horizon: int = 5
    skip_horizon: bool = True
    ic_method: str = "spearman"
    n_groups: int = 10
    bucketing_mode: str = "single"

    # Cross-sectional regression
    cs_add_intercept: bool = True
    cs_cov_type: str = "white"
    cs_white_type: str = "HC1"

    # Tradability mask
    min_list_days: int = 90

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

    # Incremental tests
    incremental_baseline_factors: Optional[List[FactorPath]] = None
    incremental_rolling_corr_window: int = 120

    # Output
    output_dir: str = "../out"
    output_excel_name: str = "factor_test.xlsx"

    # Plot
    plot: bool = True


P = BacktestParams()

DATASET_PATH = os.getenv("DATASET_PATH")
FACTOR_DATA_PATH = os.getenv("FACTOR_DATA_PATH")
if not DATASET_PATH or not FACTOR_DATA_PATH:
    raise EnvironmentError(
        "Missing env vars: DATASET_PATH and/or FACTOR_DATA_PATH. "
        "Please set them (e.g. in .env) before running."
    )

Path(P.output_dir).mkdir(parents=True, exist_ok=True)

# %%
# Data preparation: load price and feasible trading mask from dataset
ds = DuckPQSource(Path(DATASET_PATH))
ds.register("quotes_day")
ds.register("instruments_info")

sql = f"""
SELECT
    q.date AS date,
    q.code AS code,
    q.{P.ptype} AS price,
    (
        q.high > q.limit_down
        AND q.low  < q.limit_up
        AND COALESCE(q.st, false) = false
        AND COALESCE(q.suspended, false) = false
        AND datediff('day', i.listed_date, q.date) > {P.min_list_days}
    ) AS tradable_mask
FROM quotes_day AS q
JOIN instruments_info AS i
    ON q.code = i.code
WHERE q.date >= '{P.begin}' AND q.date <= '{P.end}'
"""

ds_data = ds.query(sql)
ds_data["date"] = pd.to_datetime(ds_data["date"])

feasible = ds_data.pivot(
    index="date", columns="code", values="tradable_mask"
).sort_index()
price = ds_data.pivot(index="date", columns="code", values="price").sort_index()
weight = None

display(pd.DataFrame({"price_rows": [price.shape[0]], "price_cols": [price.shape[1]]}))


# %%
# Factor loading: load one or multiple factors from factor storage
factor_source = DuckPQSource(Path(FACTOR_DATA_PATH))


def load_factor_df(factor_path: str) -> pd.DataFrame:
    table, name = parse_factor_path(factor_path)
    df = factor_source.get_factor(table=table, name=name, begin=P.begin, end=P.end)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_factors_for_one_test(test_cfg: List[Dict[str, str]]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for c in test_cfg:
        dfs.append(
            factor_source.get_factor(
                table=c["table_name"],
                name=c["factor_name"],
                begin=P.begin,
                end=P.end,
            ).sort_index()
        )
    return dfs


def align_like_price(df: pd.DataFrame) -> pd.DataFrame:
    return df.reindex(index=price.index, columns=price.columns)


# %%
# Stats helpers: Newey-West t-stat, IC distribution/tails, stability/drift tests
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


# %%
# Monotonicity helpers: Spearman, layer-index regression, and non-parametric ordered trend test
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


# %%
# Incremental helpers: correlation vs baseline factors and R2 uplift tests
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
    f1 = align_like_price(test_factor)
    f2 = align_like_price(base_factor)

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


# %%
# Core runner: run one test with enhanced IC, monotonicity, and incremental checks
def run_one_test(test_cfg: List[Dict[str, str]]) -> Dict[str, Any]:
    dfs = load_factors_for_one_test(test_cfg)
    e = Evaluator(factor=dfs, price=price)

    res: Dict[str, Any] = {}
    factor_names = list(e._factors.keys())

    # Coverage
    cov_res = {}
    for factor_name, df in e._factors.items():
        aligned = align_like_price(df)
        coverage = aligned.count(axis=1) / price.count(axis=1)
        cov_res[factor_name] = safe_float(coverage.mean())
        if P.plot:
            coverage.plot(title=f"Coverage of {factor_name}", figsize=(20, 4))
    res["coverage"] = pd.Series(cov_res, name="coverage")

    # IC and IC tests
    e.get_info_coef(horizon=P.horizon, skip_horizon=P.skip_horizon, method=P.ic_method)
    ic_df = e.ic.copy()
    res["ic_raw"] = ic_df

    ic_summ_rows = []
    ic_stab_rows = []
    for col in ic_df.columns:
        s = ic_df[col]
        summ = ic_summary_stats(s)
        stab = ic_stability_tests(s, P.ic_roll_window, P.ic_acf_lags, P.ic_break_k)
        ic_summ_rows.append(pd.Series(summ, name=col))
        ic_stab_rows.append(pd.Series(stab, name=col))

    res["ic_summary"] = pd.DataFrame(ic_summ_rows)
    res["ic_stability"] = pd.DataFrame(ic_stab_rows)

    if P.plot:
        pd.concat(
            [
                ic_df,
                ic_df.rolling(20).mean().add_suffix(" roll20_mean"),
                ic_df.cumsum().add_suffix(" cumsum"),
            ],
            axis=1,
        ).plot(
            figsize=(20, 5),
            secondary_y=[c for c in (ic_df.columns + " cumsum")],
            title="IC diagnostics",
        )
        ic_df.resample("YE").mean().plot.bar(figsize=(20, 4), title="Yearly IC mean")
        ic_df.resample("ME").mean().plot.bar(figsize=(35, 4), title="Monthly IC mean")

    # Group returns
    e.get_group_returns(
        n=P.n_groups,
        horizon=P.horizon,
        skip_horizon=P.skip_horizon,
        mode=P.bucketing_mode,
        feasible=feasible,
        weight=weight,
    )

    factor_return = e.sorted_factor_return
    group_returns = pd.concat(
        [gr.groupby(level=0).mean() for gr in e.group_returns.values()]
        + [factor_return],
        axis=1,
    )
    group_values = (group_returns.fillna(0) + 1).cumprod()
    group_values_stat = group_values.apply(quool.Evaluator.evaluate)

    group_returns_mean = group_returns.mean()
    group_returns_t = group_returns_mean / (
        group_returns.std(ddof=1) / np.sqrt(group_returns.shape[0])
    )
    res["group_return_summary"] = pd.concat(
        [group_returns_mean, group_returns_t],
        axis=1,
        keys=["mean", "t"],
    )
    res["group_value_summary"] = group_values_stat

    # Monotonicity tests (per factor)
    mono_rows = []
    for factor_name in factor_names:
        mono = group_monotonicity_tests(
            e,
            factor_name=factor_name,
            n_groups=P.n_groups,
            use_excess=P.monotonicity_use_excess,
        )
        mono_rows.append(pd.Series(mono, name=factor_name))
    res["monotonicity"] = pd.DataFrame(mono_rows)

    if P.plot:
        fig, ax = plt.subplots(figsize=(20, 4))
        for factor_name in factor_names:
            cols = _extract_factor_group_columns(
                group_returns_mean, factor_name, P.n_groups
            )
            group_returns_mean[cols].plot.bar(ax=ax, label=factor_name)


        group_values.iloc[:, :-1].plot(
            title="Group cumulative returns", figsize=(20, 5)
        )

        ax.set_title("Group mean returns (combined)")
        ax.legend()
        plt.show()

    # Cross-sectional regression (test model)
    e.cross_sectional_regression(
        horizon=P.horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=P.cs_add_intercept,
        cov_type=P.cs_cov_type,
        white_type=P.cs_white_type,
    )
    res["cs_premia"] = e.factor_premia
    res["cs_premia_t"] = e.factor_premia_t
    res["cs_r2"] = e.factor_r2

    # Incremental tests vs baseline factors
    incremental = {}
    if P.incremental_baseline_factors:
        # Build baseline evaluator (regression with baseline factors only)
        base_dfs = [
            align_like_price(load_factor_df(fp))
            for fp in P.incremental_baseline_factors
        ]
        e_base = Evaluator(factor=base_dfs, price=price)
        e_base.cross_sectional_regression(
            horizon=P.horizon,
            feasible=feasible,
            weight=weight,
            add_intercept=P.cs_add_intercept,
            cov_type=P.cs_cov_type,
            white_type=P.cs_white_type,
        )

        # Correlation and stability: only defined for single test factor vs each baseline
        # For multi-factor tests, correlations are computed for each factor separately.
        corr_rows = []
        for test_factor_name, test_df in e._factors.items():
            for base_fp, base_df in zip(P.incremental_baseline_factors, base_dfs):
                stats_corr = pairwise_corr_and_stability(
                    test_factor=test_df,
                    base_factor=base_df,
                    feasible_mask=feasible,
                    roll_window=P.incremental_rolling_corr_window,
                )
                corr_rows.append(
                    {
                        "test_factor": test_factor_name,
                        "baseline_factor": base_fp,
                        **stats_corr,
                    }
                )
        incremental["corr_vs_baseline"] = pd.DataFrame(corr_rows)

        # R2 uplift (test vs baseline)
        incremental["r2_uplift"] = cs_r2_uplift(e_test=e, e_base=e_base)
    else:
        incremental["corr_vs_baseline"] = pd.DataFrame(
            columns=[
                "test_factor",
                "baseline_factor",
                "corr_mean",
                "corr_std",
                "corr_roll_mean_std",
            ]
        )
        incremental["r2_uplift"] = {
            "cs_r2_mean": (
                safe_float(
                    pd.Series(
                        e.factor_r2.select_dtypes(include=[np.number]).iloc[:, 0]
                    ).mean()
                )
                if isinstance(e.factor_r2, pd.DataFrame)
                else np.nan
            ),
            "cs_r2_base_mean": np.nan,
            "cs_r2_uplift": np.nan,
        }

    res["incremental"] = incremental

    return res


# %%
# Runner: execute all tests and display key tables for each test
tests_cfg = normalize_tests(P.factors, treat_list_str_as=P.treat_list_str_as)
if not tests_cfg:
    raise ValueError("No tests configured.")

results: Dict[str, Dict[str, Any]] = {}

for i, test_cfg in enumerate(tests_cfg, 1):
    test_key = pretty_test_key(test_cfg)
    display_markdown(Markdown(f"# Test {i}/{len(tests_cfg)}: `{test_key}`"))

    results[test_key] = run_one_test(test_cfg)

    display_markdown(Markdown("## Coverage"))
    display(results[test_key]["coverage"].to_frame())

    display_markdown(Markdown("## IC Summary (enhanced)"))
    display(results[test_key]["ic_summary"])

    display_markdown(Markdown("## IC Stability/Drift"))
    display(results[test_key]["ic_stability"])

    display_markdown(Markdown("## Group Return Summary"))
    display(results[test_key]["group_return_summary"])

    display_markdown(Markdown("## Monotonicity Tests"))
    display(results[test_key]["monotonicity"])

    display_markdown(Markdown("## Cross-Sectional Regression (R2 snapshot)"))
    display(results[test_key]["cs_r2"])

    display_markdown(Markdown("## Incremental: Corr vs Baseline (if configured)"))
    display(results[test_key]["incremental"]["corr_vs_baseline"])

    display_markdown(Markdown("## Incremental: R2 Uplift (if configured)"))
    display(pd.DataFrame([results[test_key]["incremental"]["r2_uplift"]]))


# %%
# Summary builder: generate one-row-per-test summary for Excel persistence
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


def build_test_matrix(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    blocks: List[pd.DataFrame] = []

    for test_key, res in results.items():
        cov = res.get("coverage")  # often Series
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
            # 取第一行作为 test 级别值
            inc_row = inc_r2_df.iloc[0]
            for c in inc_row.index:
                big[c] = inc_row[c]

        big.insert(0, "test_key", test_key)

        big = big.reset_index().rename(columns={"index": "factor"})

        blocks.append(big)

    if not blocks:
        return pd.DataFrame()

    out = pd.concat(blocks, axis=0, ignore_index=True)

    return out


summary_df = build_test_matrix(results)
display(summary_df)

# %%
# Excel persistence: write only summary sheet and (optional) baseline-corr details
output_path = Path(P.output_dir) / P.output_excel_name

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)

    # Optional: write correlation-vs-baseline as an auxiliary sheet (still "summary-like")
    corr_all = []
    for test_key, res in results.items():
        corr_df = res["incremental"]["corr_vs_baseline"]
        if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
            tmp = corr_df.copy()
            tmp.insert(0, "test_key", test_key)
            corr_all.append(tmp)
    if corr_all:
        pd.concat(corr_all, axis=0, ignore_index=True).to_excel(
            writer, sheet_name="corr_vs_baseline", index=False
        )

display_markdown(Markdown(f"Saved Excel summary to: `{str(output_path)}`"))

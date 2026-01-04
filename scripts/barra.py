# %%
# Barra Pure Factor Model Builder: compose sub-factors, build industry dummies, and run cross-sectional WLS to estimate factor returns
# Helpers and Utilities as follows
import os
import dotenv
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

import statsmodels.api as sm

from factool import DuckPQSource
from parquool import setup_logger
from tools import load_factor, mcap_neutralize

FactorPath = str
CorrMethod = Literal["pearson", "spearman"]
WeightMode = Literal["signed", "abs_with_sign_flip"]
ComposeMethod = Literal["equal_weight", "ic_weight_1y"]

dotenv.load_dotenv()


# Parameters: configure time range, returns horizon, style groups (sub-factors), and composition method
@dataclass
class BarraParams:
    # REQUIRED: Style-factor path
    style_factors: Dict[str, List[FactorPath]]

    begin: str = "2024-01-01"
    end: str = "2025-06-30"

    ptype: str = "open_post"
    horizon: int = 5
    skip_horizon: bool = True

    min_list_days: int = 90
    mcap_factor_path: FactorPath = "barra_size/mcap_float_a"

    # Sub-factor composition
    compose_method: ComposeMethod = "equal_weight"
    ic_method: Literal["spearman", "pearson"] = "spearman"
    ic_lookback_days: int = 252
    ic_min_obs: int = 60
    ic_clip_weights: float = 3.0  # clip z-scored IC weights to [-k, k]

    # Output
    output_path: str = "barra_model_outputs.parquet"


# Data loading: price, tradable mask, and forward returns (date x code)
def load_future_returns(
    source: DuckPQSource,
    begin: str,
    end: str,
    horizon: int,
    *,
    quotes_table: str = "quotes_day",
    ptype: str = "open_post",
    lookback: int = 252,
    min_list_days: int = 90,
    instruments_table: str = "instruments_info",
    extend_for_horizon: bool = True,
):
    # 1) Form trading calendar
    estimate_begin = pd.to_datetime(begin) - pd.Timedelta(days=2 * lookback + 7)
    estimate_end = pd.to_datetime(end) + pd.Timedelta(days=2 * horizon + 7)
    cal_sql = f"""
    SELECT DISTINCT date
    FROM {quotes_table}
    WHERE date >= '{estimate_begin.date()}' AND date <= '{estimate_end.date()}'
    ORDER BY date
    """
    cal = source.query(cal_sql)
    if cal.empty:
        idx = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([], name="date"), pd.Index([], name="code")]
        )
        return pd.DataFrame({"returns": pd.Series(dtype="float64")}, index=idx)

    cal["date"] = pd.to_datetime(cal["date"])
    trading_days = cal["date"].sort_values().reset_index(drop=True)

    # 2) Extending window to acquire future return
    begin_dt = pd.to_datetime(begin)
    end_dt = pd.to_datetime(end)

    adjusted_begin_dt = begin_dt
    adjusted_end_dt = end_dt
    if extend_for_horizon:
        # end_dt + 1 + horizon for future return on end_dt
        need_ahead = 1 + int(horizon)
        need_back = lookback - 1

        # Find where end_dt is
        begin_pos = int(trading_days.searchsorted(begin_dt, side="right"))
        end_pos = int(trading_days.searchsorted(end_dt, side="left"))
        if end_pos < len(trading_days) and trading_days.iloc[end_pos] == end_dt:
            end_pos = end_pos
        else:
            # end_dt not in the calender, use latest date
            end_pos = end_pos - 1
        if begin_pos > 0 and trading_days.iloc[begin_pos] == begin_dt:
            begin_pos = begin_pos
        else:
            begin_pos = begin_pos + 1

        if end_pos >= 0:
            target_pos = end_pos + need_ahead
            if target_pos < len(trading_days):
                adjusted_end_dt = trading_days.iloc[target_pos]
            else:
                adjusted_end_dt = trading_days.iloc[-1]
        if begin_pos >= 0:
            target_pos = begin_pos - need_back
            if target_pos > 0:
                adjusted_begin_dt = trading_days.iloc[target_pos]
            else:
                adjusted_begin_dt = trading_days.iloc[0]

    # 3) SQL query for quotes and tradable mask
    sql = f"""
    SELECT
        q.date AS date,
        q.code AS code,
        q.{ptype} AS price,
        (
            q.high > q.limit_down
            AND q.low  < q.limit_up
            AND COALESCE(q.st, false) = false
            AND COALESCE(q.suspended, false) = false
            AND datediff('day', i.listed_date, q.date) > {min_list_days}
        ) AS tradable_mask
    FROM {quotes_table} AS q
    JOIN {instruments_table} AS i
        ON q.code = i.code
    WHERE q.date >= '{adjusted_begin_dt.date()}' AND q.date <= '{adjusted_end_dt.date()}'
    """
    ds_data = source.query(sql)
    ds_data["date"] = pd.to_datetime(ds_data["date"])

    price = ds_data[["date", "code", "price"]].copy()
    price["price"] = price["price"].where(ds_data["tradable_mask"])
    price = price.set_index(["date", "code"]).sort_index()

    # 4) For future return
    g = price.groupby("code")["price"]
    returns = g.shift(-1 - int(horizon)) / g.shift(-1) - 1
    ret = returns.to_frame("returns")

    # 5) Clip to user defined interval
    if lookback > 0:
        return ret.loc[(slice(adjusted_begin_dt, end_dt), slice(None)), :]
    ret = ret.loc[(slice(begin_dt, end_dt), slice(None)), :]
    return ret


# Industry dummies: transform long table industry_mapping into date x code x industry dummy design matrix
def load_industry_dummies(source: DuckPQSource, begin: str, end: str) -> pd.DataFrame:
    sql_ind = f"""
    SELECT
        date,
        code,
        first_industry_code
    FROM industry_mapping
    WHERE date >= '{begin}' AND date <= '{end}'
    """
    ind = source.query(sql_ind)
    ind["date"] = pd.to_datetime(ind["date"])
    ind = ind.dropna(subset=["first_industry_code"])
    ind["first_industry_code"] = ind["first_industry_code"].astype(str)

    # One-hot per (date, code)
    dummies = pd.get_dummies(ind["first_industry_code"], prefix="ind", dtype=float)
    out = pd.concat([ind[["date", "code"]], dummies], axis=1)
    return out.set_index(["date", "code"]).sort_index()


# Style composition: equal-weight or rolling 1y IC-weight within each style group
def ic_weighted_combine(
    factors: pd.DataFrame,
    forward_ret: pd.DataFrame | pd.Series,
    *,
    style_name: str = "combo",
    window: int = 60,
    min_periods: Optional[int] = None,
    corr: CorrMethod = "spearman",
    weight_mode: WeightMode = "abs_with_sign_flip",
    weight_clip: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combines factors using rolling IC-mean weights (no groupby-apply)."""
    if not isinstance(factors.index, pd.MultiIndex) or list(factors.index.names) != [
        "date",
        "code",
    ]:
        raise ValueError("factors must have MultiIndex names ['date','code']")

    if isinstance(forward_ret, pd.DataFrame):
        if forward_ret.shape[1] != 1:
            raise ValueError("fwd_ret DataFrame must have exactly one column")
        y = forward_ret.iloc[:, 0]
    else:
        y = forward_ret
    y.name = "returns"

    if not isinstance(y.index, pd.MultiIndex) or list(y.index.names) != [
        "date",
        "code",
    ]:
        raise ValueError("fwd_ret must have MultiIndex names ['date','code']")

    ic_data = pd.concat([factors, y], axis=1).sort_index()
    ic = ic_data.groupby("date").corr(method=corr)
    ic = ic.loc(axis=0)[:, "returns"].droplevel(1).iloc[:, :-1]

    if min_periods is None:
        min_periods = window
    w = ic.rolling(window=window, min_periods=min_periods).mean()
    w = w.dropna(axis=0, how="all")
    factors = factors.loc[w.index]

    if weight_clip is not None:
        w = w.clip(-float(weight_clip), float(weight_clip))

    # Broadcast weights back to row-level (N x K) by date
    d = factors.index.get_level_values("date")
    W = w.reindex(d).to_numpy(dtype=float)  # (N, K)
    X = factors.to_numpy(dtype=float)  # (N, K)

    if weight_mode == "abs_with_sign_flip":
        sgn = np.sign(W)
        sgn[~np.isfinite(sgn)] = 0.0
        X = X * sgn
        W_use = np.abs(W)
    elif weight_mode == "signed":
        W_use = W
    else:
        raise ValueError("weight_mode must be 'abs_with_sign_flip' or 'signed'")

    valid = np.isfinite(X) & np.isfinite(W_use)
    W_eff = np.where(valid, W_use, 0.0)
    X_eff = np.where(valid, X, 0.0)

    num = (W_eff * X_eff).sum(axis=1)
    den = W_eff.sum(axis=1)
    combo = np.where(den != 0.0, num / den, np.nan)

    combo_df = pd.DataFrame({style_name: combo}, index=factors.index)
    return combo_df


# Cross-sectional regression: per date WLS with country (implicit), industry dummies, and style exposures
def barra_cs_regression(
    returns: pd.DataFrame,
    style_expo: pd.DataFrame,
    industry: pd.DataFrame,
    mcap: pd.DataFrame,
    *,
    min_style_coverage: float = 0.5,
    drop_industry_rule: str = "max_cap",  # {"max_cap", "first"}
    progress: bool = True,
    progress_every: int = 20,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    """
    Cross-sectional Barra-like regression (per date) using WLS with:
      - explicit intercept ("const") as the country/market factor
      - industry dummy exposures
      - style exposures

    Identification approach:
      - Drop ONE industry dummy per date (baseline industry) to avoid perfect multicollinearity
        with the intercept ("dummy variable trap").
      - Additionally, drop any all-zero industry columns on that date (industry absent in sample).

    Model per date (after dropping baseline industry):
      y = const + sum_{k != base} ind_k * r_ind_k + Styles * r_style + eps

    Recovering the dropped (baseline) industry factor return:
      If you want the FULL set of industry returns r_ind_full (including the dropped one),
      you need one extra normalization. Here we impose a common Barra convention:
        cap-weighted sum of industry factor returns equals 0:
          sum_k w_k * r_k = 0
      where w_k is the within-sample cap weight of industry k on that date.

      Given estimated returns for non-baseline industries, we back out:
        r_base = -(sum_{k != base} w_k * r_k) / w_base

    Notes:
      - This avoids solving a singular KKT system.
      - If w_base is ~0 (baseline industry has ~0 cap), r_base is set to NaN.

    Parameters
    ----------
    drop_industry_rule:
        "max_cap": choose the baseline industry as the one with the largest within-sample cap
                   (recommended for stability).
        "first": choose the first industry column (deterministic but may be unstable).

    Returns
    -------
    factor_returns_df:
        DataFrame indexed by date with columns:
          - "const"
          - all kept industry columns (excluding baseline) + styles
          - plus recovered baseline industry return column (so you can see it too)
        (If you prefer, you can separate it out; kept together here for convenience.)
    specific_returns:
        Series indexed like returns (MultiIndex date, code) with regression residuals.
    design_mats:
        Dict[date -> design matrix used in regression (after dropping baseline and empty cols)].
    """
    if logger is None:
        logger = setup_logger("barra_cs_regression")

    factor_returns: List[pd.Series] = []
    specific_returns = pd.Series(index=returns.index, dtype=float)
    design_mats: Dict[pd.Timestamp, pd.DataFrame] = {}

    # concat: [returns, industry, style, mcap]
    reg_df = pd.concat([returns, industry, style_expo, mcap], axis=1, join="inner")
    dates = reg_df.index.get_level_values("date").unique().sort_values()
    n_dates = len(dates)

    for di, dt in enumerate(dates, start=1):
        rdf = reg_df.loc[dt]

        if progress and (di == 1 or di % progress_every == 0 or di == n_dates):
            logger.info("barra_cs_regression %s (%d/%d)", dt, di, n_dates)

        # y
        y = rdf.iloc[:, 0].astype(float)
        codes = y.index
        if len(codes) < 100:
            if progress:
                logger.info("  skip %s: too few stocks (%d)", dt, len(codes))
            factor_returns.append(pd.Series(name=dt, dtype=float))
            continue

        # Build X_full (industry + style), without mcap column
        X_full = rdf.iloc[:, 1:-1].copy()

        # Identify industry dummies
        ind_cols_all = [c for c in X_full.columns if str(c).startswith("ind_")]

        # --- 1) Per-day coverage filter for style factors (<min_style_coverage drop) ---
        keep_style_cols = []
        for c in X_full.columns:
            if c in ind_cols_all:
                continue
            coverage = X_full[c].notna().sum() / float(len(codes))
            if coverage >= min_style_coverage:
                keep_style_cols.append(c)

        # Start from all industry cols + kept styles
        X = X_full[ind_cols_all + keep_style_cols].copy()

        # Add intercept
        X = sm.add_constant(X, has_constant="add")

        # --- 2) valid rows: require y, all X, and mcap notna/positive ---
        valid = y.notna()
        valid &= X.notna().all(axis=1)

        mcap_col = rdf.columns[-1]
        mcap_today = rdf[mcap_col].astype(float)
        valid &= mcap_today.notna()
        valid &= mcap_today > 0

        # Apply row filter
        yv = y.loc[valid].astype(float)
        Xv = X.loc[valid].astype(float)
        mcap_v = mcap_today.loc[valid].astype(float)

        # Basic sufficiency check
        if yv.shape[0] < 20:
            if progress:
                logger.info("  skip %s: too few valid rows (%d)", dt, int(yv.shape[0]))
            factor_returns.append(pd.Series(name=dt, dtype=float))
            continue

        # WLS weights (here: mcap)
        wv = mcap_v.values

        # --- 3) Clean industry columns for this date ---
        # Drop all-zero (or constant) industry columns in the valid sample.
        ind_cols_present = []
        for c in ind_cols_all:
            if c not in Xv.columns:
                continue
            # For dummies, std==0 means either all 0 or all 1; both are non-informative.
            if float(Xv[c].std(ddof=0)) > 0.0:
                ind_cols_present.append(c)

        # If no usable industry dummies, just run const + styles
        if len(ind_cols_present) == 0:
            Xreg = Xv[["const"] + keep_style_cols].copy()
            n_params = Xreg.shape[1]
            if Xreg.shape[0] < max(20, n_params + 5):
                if progress:
                    logger.info("  skip %s: insufficient rows vs params", dt)
                factor_returns.append(pd.Series(name=dt, dtype=float))
                continue

            fit = sm.WLS(yv.values, Xreg.values, weights=wv).fit()
            params = pd.Series(fit.params, index=Xreg.columns, name=dt)
            resid = fit.resid

            factor_returns.append(params)
            specific_returns.loc[(dt, yv.index)] = resid
            design_mats[dt] = Xreg
            continue

        # --- 4) Choose baseline industry to drop (identification) ---
        ind_mat = Xv[ind_cols_present]  # (n, K)
        cap_by_ind = (ind_mat.mul(mcap_v, axis=0)).sum(axis=0)  # (K,)
        total_cap = float(cap_by_ind.sum())

        if drop_industry_rule == "max_cap":
            base_ind = cap_by_ind.idxmax()
        elif drop_industry_rule == "first":
            base_ind = ind_cols_present[0]
        else:
            raise ValueError(f"Unknown drop_industry_rule={drop_industry_rule}")

        # Regression uses all columns except the baseline industry dummy
        ind_cols_reg = [c for c in ind_cols_present if c != base_ind]
        Xreg_cols = ["const"] + ind_cols_reg + keep_style_cols
        Xreg = Xv[Xreg_cols].copy()

        n_params = Xreg.shape[1]
        if Xreg.shape[0] < max(20, n_params + 5):
            if progress:
                logger.info(
                    "  skip %s: insufficient valid rows (%d) vs params (%d)",
                    dt,
                    int(Xreg.shape[0]),
                    int(n_params),
                )
            factor_returns.append(pd.Series(name=dt, dtype=float))
            continue

        # --- 5) Fit WLS (full rank by construction) ---
        try:
            fit = sm.WLS(yv.values, Xreg.values, weights=wv).fit()
        except Exception as e:
            if progress:
                logger.exception("  regression failed %s: %s", dt, e)
            factor_returns.append(pd.Series(name=dt, dtype=float))
            continue

        params = pd.Series(fit.params, index=Xreg.columns, name=dt)
        resid = fit.resid

        # --- 6) Map industry coefficients to a consistent "sum-to-zero" level ---
        # In a drop-one dummy regression with intercept, the estimated industry params for
        # non-baseline industries are *relative* returns:
        #   g_k = r_k - r_base
        # We want "absolute" industry returns r_k under the normalization:
        #   sum_k w_k * r_k = 0
        # where w_k are within-sample cap weights by industry (sum w_k = 1).
        #
        # Let r_k = g_k + r_base for all k, and g_base = 0. Then:
        #   0 = sum_k w_k * (g_k + r_base) = sum_k w_k*g_k + r_base
        # => r_base = - sum_k w_k*g_k = - sum_{k!=base} w_k*g_k
        #
        # To keep fitted values unchanged, the intercept must be shifted oppositely:
        #   const_abs = const_rel - r_base

        w_by_ind = (cap_by_ind / total_cap) if total_cap > 0 else None

        # Build g_k for all present industries (relative to baseline)
        g_by_ind = {c: 0.0 for c in ind_cols_present}
        for c in ind_cols_reg:
            g_by_ind[c] = float(params.get(c, 0.0))

        if w_by_ind is None:
            r_base = np.nan
        else:
            r_base = 0.0
            for c in ind_cols_present:
                r_base -= float(w_by_ind.get(c, 0.0)) * float(g_by_ind[c])

        # Absolute industry returns under sum-to-zero normalization
        r_by_ind = {c: (float(g_by_ind[c]) + float(r_base)) for c in ind_cols_present}

        # Shift intercept so predictions stay identical after re-leveling industries
        params_full = params.copy()
        if np.isfinite(r_base):
            params_full["const"] = float(params_full["const"]) - float(r_base)

        for c, r_c in r_by_ind.items():
            params_full[c] = r_c

        # Optional stable order
        ordered_cols = ["const"] + sorted(ind_cols_present) + keep_style_cols
        params_full = params_full.reindex(ordered_cols)

        factor_returns.append(params_full)

        # Save residuals and design matrix actually used
        specific_returns.loc[(dt, yv.index)] = resid
        design_mats[dt] = Xreg

    factor_returns_df = pd.DataFrame(factor_returns).sort_index()
    return factor_returns_df, specific_returns, design_mats


# Pure factor portfolio weights (optional): omega = (X' W X)^(-1) X' W ; each row corresponds to a factor portfolio
def compute_pure_factor_weights_for_date(
    X: pd.DataFrame,
    w: pd.Series,
) -> pd.DataFrame:
    # X: n_stocks x k_factors (including const + industries + styles)
    # w: n_stocks weights (non-negative)
    Xv = X.copy()
    ww = w.reindex(Xv.index).fillna(0.0).values
    W = np.diag(ww)

    XtW = Xv.values.T @ W
    XtWX = XtW @ Xv.values
    try:
        inv = np.linalg.pinv(XtWX)
    except Exception:
        inv = np.linalg.pinv(XtWX + 1e-8 * np.eye(XtWX.shape[0]))
    omega = inv @ XtW  # k x n

    return pd.DataFrame(omega, index=Xv.columns, columns=Xv.index)


# %%
# Basic Variables and parameter settings
P = BarraParams(
    begin="2024-01-01",
    end="2025-06-30",
    style_factors={
        "size": [
            "barra_size/mcap_float_a",
            "barra_size/ln_mcap_float_a",
            "barra_size/ln_mcap_float_a_cu",
            "barra_size/non_linear_size",
        ],
        "value": [
            "barra_value/total_equity_mrq_to_mktcap",
            "barra_value/total_equity_ttm_to_mktcap",
            "barra_value/operating_revenue_mrq_to_mktcap",
            "barra_value/operating_revenue_ttm_to_mktcap",
        ],
        "liquidity": [
            "barra_liquidity/barra_liquidity_turnover_63d",
            "barra_liquidity/barra_liquidity_amount_turnover",
            "barra_liquidity/barra_liquidity_amihud_illiquidity",
        ],
        "leverage": [
            "barra_leverage/barra_book_leverage",
            "barra_leverage/barra_market_leverage",
        ],
        "volatility": [
            "barra_volatility/barra_vol_std_252",
            "barra_volatility/barra_beta_252",
            "barra_volatility/barra_residvol_252",
        ],
        "momentum": [
            "barra_momentum/barra_mom_st_63d",
            "barra_momentum/barra_mom_lt_126d_ex21d",
            "barra_momentum/barra_rev_st_21d",
        ],
        "profitability": [
            "barra_profitability/ROA",
            "barra_profitability/ROE",
            "barra_profitability/asset_turnover",
            "barra_profitability/profit_margin",
        ],
        "growth": [
            "barra_growth/net_profit_qoq_gr_mean_20",
            "barra_growth/net_profit_qoq_gr_std_20",
            "barra_growth/net_profit_yoy_gr_mean_20",
            "barra_growth/net_profit_yoy_gr_std_20",
            "barra_growth/net_profit_accel_mean_20",
        ],
    },
    compose_method="ic_weight_1y",
    mcap_factor_path="barra_size/mcap_float_a",
)


DATASET_PATH = os.getenv("DATASET_PATH")
FACTOR_DATA_PATH = os.getenv("FACTOR_DATA_PATH")
if not DATASET_PATH or not FACTOR_DATA_PATH:
    raise EnvironmentError(
        "Missing env vars: DATASET_PATH and/or FACTOR_DATA_PATH. "
        "Please set them (e.g. in .env) before running."
    )

ds = DuckPQSource(Path(DATASET_PATH))
ds.register("quotes_day")
ds.register("instruments_info")
ds.register("industry_mapping")

fs = DuckPQSource(Path(FACTOR_DATA_PATH))


ret = load_future_returns(
    ds, begin=P.begin, end=P.end, horizon=P.horizon, lookback=P.ic_lookback_days
)

industry_data = load_industry_dummies(source=ds, begin=P.begin, end=P.end)

# Market cap weights: load float market cap matrix for WLS and style standardization
mcap = load_factor(
    fs, "barra_size/mcap_float_a", P.begin, P.end, lookback=P.ic_lookback_days
)
mcap = mcap.where(mcap > 0)

# %%
# Build style exposures (8 columns) as date x code matrices and optionally standardize
style_exposures: Dict[str, pd.DataFrame] = {}
for style_name, paths in P.style_factors.items():
    raw = load_factor(
        source=fs,
        factor_paths=paths,
        begin=P.begin,
        end=P.end,
        lookback=P.ic_lookback_days,
    )
    neutralized = mcap_neutralize(
        factors=raw,
        mcap=mcap,
    )
    ic_combined = ic_weighted_combine(
        factors=neutralized,
        forward_ret=ret,
        window=P.ic_lookback_days,
        style_name=style_name,
    )
    final = mcap_neutralize(
        ic_combined,
        mcap=mcap,
    )
    style_exposures[style_name] = final
style_exposures = pd.concat(list(style_exposures.values()), axis=1)

# %%
# Run Regression
factor_returns, specific_returns, design_mats = barra_cs_regression(
    returns=ret,
    style_expo=style_exposures,
    industry=industry_data,
    mcap=mcap,
    progress_every=1,
)

# %%
# Example: compute omega for one date (can be slow if done for all dates)
OMEGA_SAMPLE_DATE = factor_returns.dropna(how="all").index.min()
omega_sample = None
if OMEGA_SAMPLE_DATE is not None and OMEGA_SAMPLE_DATE in design_mats:
    Xv = design_mats[OMEGA_SAMPLE_DATE]
    w = (
        (
            mcap.loc[OMEGA_SAMPLE_DATE].squeeze()
            / mcap.loc[OMEGA_SAMPLE_DATE].squeeze().sum()
        )
        .reindex(Xv.index)
        .fillna(0.0)
    )
    omega_sample = compute_pure_factor_weights_for_date(Xv, w)

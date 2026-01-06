import os
import dotenv
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Tuple, Literal, Sequence

import numpy as np
import pandas as pd

import statsmodels.api as sm

from factool import DuckPQSource, ComposerConfig, Composer
from parquool import setup_logger


FactorPath = str
CorrMethod = Literal["pearson", "spearman"]
WeightMode = Literal["abs_with_sign_flip", "signed"]

dotenv.load_dotenv()


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


@dataclass
class ICComposerConfig(ComposerConfig):
    factor_paths: Dict[str, List[FactorPath]]
    begin: str
    end: str
    target_path: FactorPath = "target/open"
    weight_path: FactorPath = "barra_size/mcap_float_a"
    window: int = 252
    horizon: int = 5
    method: Literal["spearman", "pearson"] = "spearman"
    weight_mode: WeightMode = "abs_with_sign_flip"
    min_obs: int = 60
    clip_weights: float = 3.0  # clip z-scored IC weights to [-k, k]


class ICComposer(Composer):
    def __init__(
        self,
        source: DuckPQSource,
        config: ICComposerConfig,
    ):
        super().__init__(
            source=source,
            config=config,
        )
        self.mcap = source.load(
            config.weight_path,
            begin=config.begin,
            end=config.end,
            pad_begin=config.window - 1,
        )

    def preprocess(self, X: pd.DataFrame, y: pd.DataFrame):
        if isinstance(self.mcap, pd.DataFrame):
            if self.mcap.shape[1] != 1:
                raise ValueError("mcap DataFrame must have exactly one column")
            w = self.mcap.iloc[:, 0]
        else:
            w = self.mcap

        if not isinstance(w, pd.Series):
            raise TypeError("mcap must be a Series or a single-column DataFrame")

        X = X.apply(pd.to_numeric, errors="coerce")
        w = pd.to_numeric(w, errors="coerce")

        # Align indices
        tmp = X.join(w.rename("w"), how="left")
        w = tmp["w"]

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
        std = var.pow(0.5).replace(0.0, np.nan)
        # Broadcast mean/std back to row level by date using transform
        mean_row = X.groupby(level="date").transform(lambda s: mean[s.name])
        std_row = X.groupby(level="date").transform(lambda s: std[s.name])
        Z = (X - mean_row) / std_row
        return Z, y

    def postprocess(self, factor):
        factor, _ = self.preprocess(factor, None)
        return factor

    def compose(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        cfg = self.config

        # 1) per-date IC: corr across codes between each factor column and returns
        ic_data = pd.concat([X, y], axis=1).sort_index()
        ic = ic_data.groupby("date").corr(method=cfg.method)

        # pick correlation with "returns" row in each date block
        ic = ic.loc(axis=0)[:, "returns"].droplevel(1).iloc[:, :-1]  # (date x factors)

        # 2) rolling mean weight, then shift(1) to avoid peeking at same-day returns
        w = ic.rolling(window=cfg.window, min_periods=cfg.min_obs).mean()
        w = w.dropna(axis=0, how="all")

        # Align X to weight dates
        X2 = X.loc[X.index.get_level_values("date").isin(w.index)]
        if X2.empty:
            return pd.DataFrame(
                {cfg.composed_name: pd.Series(dtype="float64")},
                index=pd.MultiIndex.from_arrays(
                    [pd.DatetimeIndex([], name="date"), pd.Index([], name="code")]
                ),
            )

        # 3) broadcast w by date to row-level
        d = X2.index.get_level_values("date")
        W = w.reindex(d).to_numpy(dtype=float)  # (N,K)
        XX = X2.to_numpy(dtype=float)  # (N,K)

        if cfg.weight_mode == "abs_with_sign_flip":
            sgn = np.sign(W)
            sgn[~np.isfinite(sgn)] = 0.0
            XX = XX * sgn
            W_use = np.abs(W)
        elif cfg.weight_mode == "signed":
            W_use = W
        else:
            raise ValueError("weight_mode must be 'abs_with_sign_flip' or 'signed'")

        valid = np.isfinite(XX) & np.isfinite(W_use)
        W_eff = np.where(valid, W_use, 0.0)
        X_eff = np.where(valid, XX, 0.0)

        num = (W_eff * X_eff).sum(axis=1)
        den = W_eff.sum(axis=1)
        combo = np.where(den != 0.0, num / den, np.nan)

        out = pd.DataFrame({"target": combo}, index=X2.index)

        # light diagnostics
        self.info = {
            **self.info,
            "ic_mean_tail": ic.tail(3),
            "weight_mean_tail": w.tail(3),
            "window": cfg.window,
            "min_periods": cfg.min_obs,
            "corr": cfg.method,
            "weight_mode": cfg.weight_mode,
        }
        return out


@dataclass
class BarraConfig(ComposerConfig):
    factor_paths: Dict[str, ICComposerConfig]
    begin: str
    end: str
    target_path: FactorPath = "target/open"
    weight_path: FactorPath = "barra_size/mcap_float_a"
    window: int = 252
    horizon: int = 5


class BarraComposer(Composer):

    def __init__(
        self,
        source: DuckPQSource,
        config: BarraConfig,
        industry: pd.DataFrame,
    ):
        super().__init__(source, config)
        self.mcap = source.load(
            config.weight_path,
            begin=config.begin,
            end=config.end,
            pad_begin=config.window - 1,
        )
        self.industry = industry
        self.logger = setup_logger("BarraComposer")

    def load_X(self):
        style_exposures = []
        for name, cfg in self.config.factor_paths.items():
            iccomp = ICComposer(source=self.source, config=cfg).run()
            self.logger.info(
                f"Loaded {name} factor, from {iccomp.index.levels[0].min()} to {iccomp.index.levels[0].min()}"
            )
            iccomp.columns = [name]
            style_exposures.append(iccomp)
        style_exposures = pd.concat(style_exposures, axis=1)
        return style_exposures

    def compose(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        min_style_coverage: float = 0.5,
        drop_industry_rule: Literal["max_cap", "first"] = "max_cap",
        progress: bool = True,
        progress_every: int = 20,
    ):
        factor_returns: List[pd.Series] = []
        specific_returns = pd.Series(index=y.index, dtype=float)

        # concat: [returns, industry, style, mcap]
        reg_df = pd.concat([y, self.industry, X, self.mcap], axis=1, join="inner")
        dates = reg_df.index.get_level_values("date").unique().sort_values()
        n_dates = len(dates)

        for di, dt in enumerate(dates, start=1):
            rdf = reg_df.loc[dt]

            if progress and (di == 1 or di % progress_every == 0 or di == n_dates):
                self.logger.info("barra_cs_regression %s (%d/%d)", dt, di, n_dates)

            # y
            y = rdf.iloc[:, 0].astype(float)
            codes = y.index
            if len(codes) < 100:
                if progress:
                    self.logger.info("  skip %s: too few stocks (%d)", dt, len(codes))
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
                    self.logger.info(
                        "  skip %s: too few valid rows (%d)", dt, int(yv.shape[0])
                    )
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
                        self.logger.info("  skip %s: insufficient rows vs params", dt)
                    factor_returns.append(pd.Series(name=dt, dtype=float))
                    continue

                fit = sm.WLS(yv.values, Xreg.values, weights=wv).fit()
                params = pd.Series(fit.params, index=Xreg.columns, name=dt)
                resid = fit.resid

                factor_returns.append(params)
                specific_returns.loc[(dt, yv.index)] = resid
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
                    self.logger.info(
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
                    self.logger.exception("  regression failed %s: %s", dt, e)
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
            r_by_ind = {
                c: (float(g_by_ind[c]) + float(r_base)) for c in ind_cols_present
            }

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

        factor_returns_df = pd.DataFrame(factor_returns).sort_index()
        return factor_returns_df, specific_returns

    def run(
        self,
        min_style_coverage: float = 0.5,
        drop_industry_rule: Literal["max_cap", "first"] = "max_cap",
        progress: bool = True,
        progress_every: int = 20,
    ):
        X = self.load_X()
        y = self.load_y()
        X, y = self.align_Xy(X, y)
        X, y = self.preprocess(X, y)
        out = self.compose(
            X,
            y,
            min_style_coverage=min_style_coverage,
            drop_industry_rule=drop_industry_rule,
            progress=progress,
            progress_every=progress_every,
        )
        out = self.postprocess(out)
        self.style_exposure = X
        self.future_return = y
        self.info = {
            "begin": self.config.begin,
            "end": self.config.end,
            "horizon": self.config.horizon,
            "ptype": self.config.target_path,
            "factor_paths": list(self.config.factor_paths),
            "n_rows": len(out),
        }
        return out


# Basic Variables and parameter settings
BEGIN = "2023-01-01"
END = "2024-03-19"
HORIZON = 5
WINDOW = 252
TARGET_PATH = "target/open"
WEIGHT_PATH = "barra_size/mcap_float_a"
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

size_config = ICComposerConfig(
    factor_paths=[
        "barra_size/mcap_float_a",
        "barra_size/ln_mcap_float_a",
        "barra_size/ln_mcap_float_a_cu",
        "barra_size/non_linear_size",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
value_config = ICComposerConfig(
    factor_paths=[
        "barra_value/total_equity_mrq_to_mktcap",
        "barra_value/total_equity_ttm_to_mktcap",
        "barra_value/operating_revenue_mrq_to_mktcap",
        "barra_value/operating_revenue_ttm_to_mktcap",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
liquidity_config = ICComposerConfig(
    factor_paths=[
        "barra_liquidity/barra_liquidity_turnover_63d",
        "barra_liquidity/barra_liquidity_amount_turnover",
        "barra_liquidity/barra_liquidity_amihud_illiquidity",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
leverage_config = ICComposerConfig(
    factor_paths=[
        "barra_leverage/barra_book_leverage",
        "barra_leverage/barra_market_leverage",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
volatility_config = ICComposerConfig(
    factor_paths=[
        "barra_volatility/barra_vol_std_252",
        "barra_volatility/barra_beta_252",
        "barra_volatility/barra_residvol_252",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
momentum_config = ICComposerConfig(
    factor_paths=[
        "barra_momentum/barra_mom_st_63d",
        "barra_momentum/barra_mom_lt_126d_ex21d",
        "barra_momentum/barra_rev_st_21d",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
profitability_config = ICComposerConfig(
    factor_paths=[
        "barra_profitability/ROA",
        "barra_profitability/ROE",
        "barra_profitability/asset_turnover",
        "barra_profitability/profit_margin",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
growth_config = ICComposerConfig(
    factor_paths=[
        "barra_growth/net_profit_qoq_gr_mean_20",
        "barra_growth/net_profit_qoq_gr_std_20",
        "barra_growth/net_profit_yoy_gr_mean_20",
        "barra_growth/net_profit_yoy_gr_std_20",
        "barra_growth/net_profit_accel_mean_20",
    ],
    begin=BEGIN,
    end=END,
    horizon=HORIZON,
    window=WINDOW,
    weight_path=WEIGHT_PATH,
)
barra_config = BarraConfig(
    begin=BEGIN,
    end=END,
    factor_paths={
        "size": size_config,
        "value": value_config,
        "liquidity": liquidity_config,
        "leverage": leverage_config,
        "volatility": volatility_config,
        "momentum": momentum_config,
        "profitability": profitability_config,
        "growth": growth_config,
    },
    horizon=HORIZON,
    window=WINDOW,
    target_path=TARGET_PATH,
    weight_path=WEIGHT_PATH,
)


industry = load_industry_dummies(source=ds, begin=BEGIN, end=END)
composer = BarraComposer(fs, barra_config, industry=industry)
factor_returns = composer.run(
    min_style_coverage=0.5,
    drop_industry_rule="max_cap",
    progress=True,
    progress_every=20,
    logger=None,
)
style_exposure = composer.style_exposure
style_exposure["date"] = style_exposure["date"].dt.strftime("%Y-%m-%d")
fs.upsert("barra", style_exposure, keys=["date", "code"], partition_by=["date"])

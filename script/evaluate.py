import numpy as np
import pandas as pd
from factool import Evaluator
from typing import Dict, List, Optional, Union, Tuple


def run_single_factor_pipeline(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    feasible: Optional[pd.DataFrame] = None,
    weight: Optional[pd.DataFrame] = None,
    # Grouping/HL params
    n_groups: int = 10,
    horizon: int = 1,
    bucketing_mode: str = "conditional",
    # IC params
    ic_method: str = "spearman",
    # Rolling TS exposure params
    ts_window: int = 252,
    ts_min_obs: int = 60,
    ts_intercept: bool = True,
    ts_n_jobs: int = -1,
    # Cross-sectional regression params
    cs_add_intercept: bool = True,
    cs_cov_type: str = "white",
    cs_white_type: str = "HC1",
    # Fama-MacBeth params
    fmb_nw_lag: int = 3,
    # GMM pricing params
    run_gmm: bool = True,
) -> Dict[str, object]:
    """Run a full single-factor testing pipeline.

    Steps:
      1) Initialize Evaluator and align data.
      2) Compute Information Coefficient (IC) and determine direction.
      3) Construct grouped portfolios and HL factor return.
      4) Rolling time-series exposure of each asset to the HL factor.
      5) Cross-sectional regressions of future returns on the factor.
      6) Fama-MacBeth regression (premia + NW t-stats).
      7) GRS joint alpha test.
      8) GMM linear pricing for HL factor (optional).

    Args:
      factor: DataFrame of factor exposures (dates x assets).
      price: DataFrame of asset prices (dates x assets).
      feasible: Optional eligibility mask (dates x assets).
      weight: Optional within-group weights (dates x assets).
      n_groups: Number of quantile groups for HL construction.
      horizon: Return horizon for dependent variables.
      bucketing_mode: 'conditional' or 'independent' bucketing.
      cell_weight: 'equal' or 'count' aggregation across buckets.
      hl_mode: HL derivation: 'first_last' or 'extreme'.
      ic_freq: IC horizon in periods.
      ic_method: Correlation method ('spearman' recommended).
      ts_window: Rolling window length for exposure.
      ts_min_obs: Minimum observations per rolling window.
      ts_intercept: Include intercept in rolling exposure.
      ts_standardize_hl: Standardize HL before rolling exposure.
      ts_n_jobs: Parallel jobs for rolling exposure.
      benchmark: Optional benchmark series for backtest.
      commission: Transaction cost per trade.
      cs_add_intercept: Include intercept in cross-sectional regressions.
      cs_cov_type: 'none' or 'white' covariance in cross-sectional regressions.
      cs_white_type: White estimator type ('HC0' or 'HC1').
      fmb_nw_lag: Newey-West lag for FMB t-stats.
      run_gmm: Whether to run GMM linear pricing for HL factor.

    Returns:
      Dict of outputs with keys:
        - evaluator: the Evaluator instance with all computed attributes
        - ic: daily IC series
        - hl_return: HL factor return series
        - group_returns: grouped portfolio returns dataframe (G1..Gn)
        - ts_exposure_beta / ts_exposure_t / ts_exposure_alpha
        - cs_betas / cs_tstats / cs_r2
        - fmb_premia / fmb_tstats
        - ts_alpha / ts_alpha_t / ts_beta / ts_beta_t
        - grs_stat / grs_pval
        - alpha_test_result
        - gmm_result (if run_gmm=True)
    """
    e = Evaluator(factor=factor, price=price)

    # Step 1: IC and direction
    e.get_info_coef(horizon=horizon, method=ic_method)
    ic = e.ic
    ic_tstat = e.ic.mean() / (e.ic.std() / np.sqrt(e.ic.shape[0]))

    # Step 2: Grouped portfolios and HL
    e.get_group_returns(
        n=n_groups,
        horizon=horizon,
        mode=bucketing_mode,
        feasible=feasible,
        weight=weight,
    )
    factor_return = e.sorted_factor_return
    group_returns = pd.concat(
        [gr.groupby(level=0).mean() for gr in e.group_returns.values()],
        axis=1,
    )
    group_value = pd.concat(
        [
            1 + group_returns.shift(1 + horizon).fillna(0),
            1 + factor_return.shift(1 + horizon).fillna(0),
        ],
        axis=1,
    ).cumprod()

    # Step 3: Rolling TS exposure vs HL
    e.get_factor_exposure(
        horizon=horizon,
        feasible=feasible,
        window=ts_window,
        min_obs=ts_min_obs,
        intercept=ts_intercept,
        n_jobs=ts_n_jobs,
    )
    factor_exposure_mean = e.factor_exposure.groupby(level=0).mean()
    factor_exposure_mean_t = e.factor_exposure_t.groupby(level=0).mean()

    # Step 4: Cross-sectional regression on the single factor
    e.cross_sectional_regression(
        horizon=horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=cs_add_intercept,
        cov_type=cs_cov_type,
        white_type=cs_white_type,
    )
    factor_premia = e.factor_premia
    factor_premia_t = e.factor_premia_t
    factor_r2 = e.factor_r2

    # Step 5: Fama-MacBeth regression
    e.fama_macbeth(nw_lag=fmb_nw_lag)
    fmb_premia = e.fmb_premia
    fmb_premia_t = e.fmb_tstats

    # Step 6: GRS test (joint alpha = 0)
    e.grs_test(horizon=horizon, add_intercept=True)
    grs_stat, grs_pval = e.grs_stat, e.grs_pval

    # Step 7: GMM pricing (optional) with HL factor
    gmm_result = None
    if run_gmm:
        e.gmm_linear_pricing(horizon=horizon, two_step=True)
        gmm_result = e.gmm_result

    return {
        "ic": ic,
        "ic_tstat": ic_tstat,
        "factor_return": factor_return,
        "group_returns": group_returns,
        "group_value": group_value,
        "factor_exposure_mean": factor_exposure_mean,
        "factor_exposure_mean_t": factor_exposure_mean_t,
        "factor_premia": factor_premia,
        "factor_premia_t": factor_premia_t,
        "r2": factor_r2,
        "fmb_premia": fmb_premia,
        "fmb_premia_t": fmb_premia_t,
        "grs_stat": grs_stat,
        "grs_pval": grs_pval,
        "gmm_result": gmm_result,
    }


if __name__ == "__main__":
    import factool
    import os
    import dotenv

    dotenv.load_dotenv()

    output_path = "out/test.xlsx"
    factor_name = "naive_market_size"
    n_groups = 3
    bucketing_mode = "independent"
    ts_n_jobs = -1

    dps = factool.DuckParquetSource(f"data/{factor_name}")
    source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    df = dps.get_factor("log_market_size", begin="2025-01-01", end="2025-06-30")
    price = source.get_factor("close_post", begin="2025-01-01", end="2025-06-30")

    results = run_single_factor_pipeline(
        [df],
        price,
        bucketing_mode=bucketing_mode,
        n_groups=n_groups,
        ts_n_jobs=ts_n_jobs,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results["ic"].reset_index().to_excel(writer, index=False, sheet_name="IC")
        results["factor_return"].reset_index().to_excel(
            writer, index=False, sheet_name="Factor Return"
        )
        results["group_returns"].reset_index().to_excel(
            writer, index=False, sheet_name="Group Return"
        )
        results["group_value"].reset_index().to_excel(
            writer, index=False, sheet_name="Group Value"
        )

        pd.concat(
            [
                results["factor_exposure_mean"],
                results["factor_exposure_mean_t"],
            ],
            axis=1,
        ).reset_index(names=["code"]).to_excel(
            writer, index=False, sheet_name="Time Series Regression"
        )

        pd.concat(
            [
                results["factor_premia"],
                results["factor_premia_t"],
                results["r2"].to_frame(),
            ],
            axis=1,
        ).reset_index(names=["date"]).to_excel(
            writer, index=False, sheet_name="Cross Sectional Regression"
        )

        pd.concat(
            [
                results["ic_tstat"].add_suffix("-ic-t"),
                pd.Series(
                    {
                        "grs_stat": results["grs_stat"],
                        "grs_pval": results["grs_pval"],
                    },
                ),
                results["fmb_premia"].add_suffix("-fmb_premia"),
                results["fmb_premia_t"].add_suffix("-fmb_premia-t"),
                pd.Series(
                    {
                        "gmm_J": results["gmm_result"]["J"],
                        "gmm_pval": results["gmm_result"]["pval"],
                    }
                ),
            ]
        ).to_frame("Stats").reset_index(names=["stat_name"]).to_excel(
            writer, index=False, sheet_name="Stats"
        )

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
from factool import Evaluator


def run_single_factor_pipeline(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    feasible: Optional[pd.DataFrame] = None,
    weight: Optional[pd.DataFrame] = None,
    # Grouping/HL params
    n_groups: int = 10,
    horizon: int = 1,
    bucketing_mode: str = "conditional",
    cell_weight: str = "equal",
    hl_mode: str = "first_last",
    # IC params
    ic_freq: int = 1,
    ic_method: str = "spearman",
    # Rolling TS exposure params
    ts_window: int = 252,
    ts_min_obs: int = 60,
    ts_intercept: bool = True,
    ts_standardize_hl: bool = False,
    ts_n_jobs: int = -1,
    # Cross-sectional regression params
    cs_add_intercept: bool = True,
    cs_cov_type: str = "white",
    cs_white_type: str = "HC1",
    # Fama-MacBeth params
    fmb_add_intercept: bool = True,
    fmb_nw_lag: int = 3,
    # Time-series regression params
    ts_cov_type: str = "nw",
    ts_nw_lag: int = 3,
    ts_hc_type: str = "HC1",
    # GMM pricing params
    run_gmm: bool = True,
) -> Dict[str, object]:
    """Run a full single-factor testing pipeline.

    Steps:
      1) Initialize Evaluator and align data.
      2) Compute Information Coefficient (IC) and determine direction.
      3) Construct grouped portfolios and HL factor return.
      4) Rolling time-series exposure of each asset to the HL factor.
      5) Evaluate a top-k strategy based on the factor ranks.
      6) Cross-sectional regressions of future returns on the factor.
      7) Fama-MacBeth regression (premia + NW t-stats).
      8) Time-series regressions of assets on HL; alpha tests.
      9) GRS joint alpha test.
     10) GMM linear pricing for HL factor (optional).

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
      fmb_add_intercept: Include intercept in step-1 cross-sectional regressions.
      fmb_nw_lag: Newey-West lag for FMB t-stats.
      ts_cov_type: Covariance type in time-series regressions ('none', 'white', 'nw').
      ts_nw_lag: Newey-West lag for time-series regressions.
      ts_hc_type: White estimator type for time-series regressions.
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
    e.get_info_coef(freq=ic_freq, method=ic_method)

    # Step 2: Grouped portfolios and HL
    e.get_group_returns(
        n=n_groups,
        horizon=horizon,
        other_factors=None,  # Single-factor study
        mode=bucketing_mode,
        feasible=feasible,
        weight=weight,
        cell_weight=cell_weight,
        hl_mode=hl_mode,
    )
    hl_return = e.sorted_factor_return
    group_returns = e.group_returns

    # Step 3: Rolling TS exposure vs HL
    e.get_factor_exposure(
        n=n_groups,
        horizon=horizon,
        other_factors=None,
        mode=bucketing_mode,
        feasible=feasible,
        weight=weight,
        cell_weight=cell_weight,
        window=ts_window,
        min_obs=ts_min_obs,
        intercept=ts_intercept,
        standardize_factor=ts_standardize_hl,
        n_jobs=ts_n_jobs,
        hl_mode=hl_mode,
    )

    # Step 4: Cross-sectional regression on the single factor
    e.cross_sectional_regression(
        horizon=horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=cs_add_intercept,
        cov_type=cs_cov_type,
        white_type=cs_white_type,
        orthogonalize=False,
        other_factors=None,  # Single-factor
    )

    # Step 5: Fama-MacBeth regression
    e.fama_macbeth(
        horizon=horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=fmb_add_intercept,
        nw_lag=fmb_nw_lag,
        orthogonalize=False,
        other_factors=None,
    )

    # Step 6: Time-series regressions of assets on HL factor
    e.ts_regression(
        asset_returns=None,  # defaults to next-period returns from prices
        factor_returns=None,  # defaults to HL
        add_intercept=True,
        cov_type=ts_cov_type,
        nw_lag=ts_nw_lag,
        hc_type=ts_hc_type,
    )

    # Step 7: Alpha tests (individual)
    alpha_tests = e.alpha_tests(
        asset_returns=None,
        factor_returns=None,
        cov_type=ts_cov_type,
        nw_lag=ts_nw_lag,
        hc_type=ts_hc_type,
    )

    # Step 8: GRS test (joint alpha = 0)
    grs_stat, grs_pval = e.grs_test(
        asset_returns=None, factor_returns=None, add_intercept=True
    )

    # Step 9: GMM pricing (optional) with HL factor
    gmm_result = None
    if run_gmm:
        asset_ret = e._price.pct_change(fill_method=None).shift(-1).dropna(how="all")
        factor_ret = e.sorted_factor_return.to_frame("HL").loc[asset_ret.index]
        gmm_result = e.gmm_linear_pricing(
            asset_returns=asset_ret, factor_returns=factor_ret, two_step=True
        )

    return {
        "ic": e.ic,
        "hl_return": hl_return,
        "group_returns": group_returns,
        "ts_exposure_beta": e.ts_exposure_beta,
        "ts_exposure_t": e.ts_exposure_t,
        "ts_exposure_alpha": e.ts_exposure_alpha,
        "cs_betas": e.cs_betas,
        "cs_tstats": e.cs_tstats,
        "cs_r2": e.cs_r2,
        "fmb_premia": e.fmb_premia,
        "fmb_tstats": e.fmb_tstats,
        "ts_alpha": e.ts_alpha,
        "ts_alpha_t": e.ts_alpha_t,
        "ts_beta": e.ts_beta,
        "ts_beta_t": e.ts_beta_t,
        "grs_stat": grs_stat,
        "grs_pval": grs_pval,
        "alpha_test_result": alpha_tests,
        "gmm_result": gmm_result,
    }


def run_multi_factor_pipeline(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    other_factors_for_grouping: Optional[
        Union[List[pd.DataFrame], Tuple[pd.DataFrame, ...]]
    ] = None,
    other_factors_for_cs: Optional[Dict[str, pd.DataFrame]] = None,
    feasible: Optional[pd.DataFrame] = None,
    weight: Optional[pd.DataFrame] = None,
    # Grouping/HL params
    n_groups: int = 10,
    horizon: int = 1,
    bucketing_mode: str = "conditional",
    cell_weight: str = "equal",
    hl_mode: str = "first_last",
    # IC params (single-factor IC for primary factor)
    ic_freq: int = 1,
    ic_method: str = "spearman",
    # Cross-sectional regression params (multi-factor)
    cs_add_intercept: bool = True,
    cs_cov_type: str = "white",
    cs_white_type: str = "HC1",
    orthogonalize_cs: bool = False,
    # Fama-MacBeth params
    fmb_add_intercept: bool = True,
    fmb_nw_lag: int = 3,
    # Time-series regression params (if you have factor return series beyond HL)
    ts_asset_returns: Optional[pd.DataFrame] = None,
    ts_factor_returns: Optional[pd.DataFrame] = None,
    ts_add_intercept: bool = True,
    ts_cov_type: str = "nw",
    ts_nw_lag: int = 3,
    ts_hc_type: str = "HC1",
    # GMM pricing params
    run_gmm: bool = False,
) -> Dict[str, object]:
    """Run a full multi-factor testing pipeline.

    Key differences vs single-factor:
      - Grouping for HL can be controlled using `other_factors_for_grouping` (list/tuple).
      - Cross-sectional regressions include multiple regressors via `other_factors_for_cs` (dict).
      - Fama-MacBeth estimation provides premia and NW t-stats for all included regressors.
      - Time-series regressions can use user-supplied factor returns DataFrame with multiple columns (K factors).
      - GMM pricing typically needs a multi-column factor_returns with K factors.

    Args:
      factor: Primary factor DataFrame (dates x assets).
      price: Asset prices (dates x assets).
      other_factors_for_grouping: Control factors (list/tuple of DataFrames) for bucketing HL.
      other_factors_for_cs: Dict name -> DataFrame for cross-sectional multi-factor regressions.
      feasible: Optional eligibility mask.
      weight: Optional within-group weights.
      n_groups: Number of quantile groups for HL construction.
      horizon: Return horizon in periods.
      bucketing_mode: 'conditional' or 'independent' for grouping.
      cell_weight: 'equal' or 'count' across buckets.
      hl_mode: HL derivation: 'first_last' or 'extreme'.
      ic_freq: IC horizon; computed for primary factor only.
      ic_method: Correlation method.
      cs_add_intercept: Include intercept in cross-sectional regressions.
      cs_cov_type: 'none' or 'white'.
      cs_white_type: White type.
      orthogonalize_cs: Whether to residualize regressors using other_factors_for_cs.
      fmb_add_intercept: Include intercept in CS step for FMB.
      fmb_nw_lag: Newey-West lag for FMB.
      ts_asset_returns: Optional asset returns; defaults to next-period returns from prices.
      ts_factor_returns: Optional factor returns (dates x K). If None, HL is used (single-factor TS only).
      ts_add_intercept: Include intercept in time-series regressions.
      ts_cov_type: 'none', 'white', or 'nw'.
      ts_nw_lag: Newey-West lag.
      ts_hc_type: White estimator type.
      benchmark: Optional benchmark index series.
      commission: Transaction cost per trade.
      run_gmm: Whether to run GMM pricing using ts_factor_returns.

    Returns:
      Dict of outputs similar to run_single_factor_pipeline, plus:
        - cs_betas / cs_tstats include multiple regressors
        - ts_regression uses provided factor return matrix if available
        - gmm_result (if run_gmm=True and ts_factor_returns provided)
    """
    e = Evaluator(factor=factor, price=price)

    # IC for the primary factor
    e.get_info_coef(freq=ic_freq, method=ic_method)

    # Grouped portfolios and HL (controlled by other_factors_for_grouping)
    e.get_group_returns(
        n=n_groups,
        horizon=horizon,
        other_factors=other_factors_for_grouping,
        mode=bucketing_mode,
        feasible=feasible,
        weight=weight,
        cell_weight=cell_weight,
        hl_mode=hl_mode,
    )
    hl_return = e.sorted_factor_return
    group_returns = e.group_returns

    # Cross-sectional multi-factor regression
    e.cross_sectional_regression(
        horizon=horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=cs_add_intercept,
        cov_type=cs_cov_type,
        white_type=cs_white_type,
        orthogonalize=orthogonalize_cs,
        other_factors=other_factors_for_cs,  # Multi-factor regressors
    )

    # Fama-MacBeth (multi-factor premia & NW t-stats)
    e.fama_macbeth(
        horizon=horizon,
        feasible=feasible,
        weight=weight,
        add_intercept=fmb_add_intercept,
        nw_lag=fmb_nw_lag,
        orthogonalize=orthogonalize_cs,
        other_factors=other_factors_for_cs,
    )

    # Time-series regressions
    # If ts_factor_returns is None, defaults to HL factor only (single-factor TS).
    e.ts_regression(
        asset_returns=ts_asset_returns,  # if None -> next-period returns from prices
        factor_returns=ts_factor_returns,  # provide multi-factor returns (dates x K) if available
        add_intercept=ts_add_intercept,
        cov_type=ts_cov_type,
        nw_lag=ts_nw_lag,
        hc_type=ts_hc_type,
    )

    # Alpha tests
    alpha_tests = e.alpha_tests(
        asset_returns=ts_asset_returns,
        factor_returns=(
            ts_factor_returns
            if ts_factor_returns is not None
            else e.sorted_factor_return.to_frame("HL")
        ),
        cov_type=ts_cov_type,
        nw_lag=ts_nw_lag,
        hc_type=ts_hc_type,
    )

    # GRS test
    grs_stat, grs_pval = e.grs_test(
        asset_returns=ts_asset_returns,
        factor_returns=(
            ts_factor_returns
            if ts_factor_returns is not None
            else e.sorted_factor_return.to_frame("HL")
        ),
        add_intercept=True,
    )

    # Optional GMM pricing for multi-factor returns
    gmm_result = None
    if run_gmm and ts_factor_returns is not None:
        asset_ret = (
            ts_asset_returns
            if ts_asset_returns is not None
            else e._price.pct_change().shift(-1).dropna(how="all")
        )
        # Align indices
        common_idx = asset_ret.index.intersection(ts_factor_returns.index)
        gmm_result = e.gmm_linear_pricing(
            asset_returns=asset_ret.loc[common_idx],
            factor_returns=ts_factor_returns.loc[common_idx],
            two_step=True,
        )

    return {
        "ic": e.ic,
        "hl_return": hl_return,
        "group_returns": group_returns,
        "cs_betas": e.cs_betas,
        "cs_tstats": e.cs_tstats,
        "cs_r2": e.cs_r2,
        "fmb_premia": e.fmb_premia,
        "fmb_tstats": e.fmb_tstats,
        "ts_alpha": e.ts_alpha,
        "ts_alpha_t": e.ts_alpha_t,
        "ts_beta": e.ts_beta,
        "ts_beta_t": e.ts_beta_t,
        "grs_stat": grs_stat,
        "grs_pval": grs_pval,
        "alpha_test_result": alpha_tests,
        "gmm_result": gmm_result,
    }


if __name__ == "__main__":
    import factool
    import os
    import dotenv

    dotenv.load_dotenv()
    dps = factool.DuckParquetSource("data/naive_market_size")
    source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    df = dps.get_factor("log_market_size", begin="2025-01-01", end="2025-06-30")
    price = source.get_factor("close_post", begin="2025-01-01", end="2025-06-30")

    print(
        run_single_factor_pipeline(
            df,
            price,
        )
    )

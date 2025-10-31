import numpy as np
import pandas as pd
from factool import Evaluator
from typing import Dict, Optional

import quool


def run_factor_pipeline(
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
    e = Evaluator(factor=factor, price=price)

    # Step 1: IC and direction
    e.get_info_coef(horizon=horizon, method=ic_method)
    ic = e.ic
    ic_mean = e.ic.mean()
    ic_tstat = ic_mean / (e.ic.std() / np.sqrt(e.ic.shape[0]))

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
        [gr.groupby(level=0).mean() for gr in e.group_returns.values()] + [factor_return],
        axis=1,
    )
    group_value = (1 + group_returns.shift(1 + horizon).fillna(0)).cumprod()
    group_eval = group_value.apply(quool.Evaluator.evaluate)

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
        "ic_mean": ic_mean,
        "ic_tstat": ic_tstat,
        "factor_return": factor_return,
        "group_returns": group_returns,
        "group_value": group_value,
        "group_eval": group_eval,
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

    begin = "2015-01-01"
    end = "2025-06-30"
    output_path = "out/log_market_size_h1_n10.xlsx"
    n_groups = 10
    bucketing_mode = "single"
    ts_n_jobs = -1

    dps = factool.DuckParquetSource(f"data/barra_sizes")
    df = dps.get_factor("log_market_size", begin=begin, end=end)
    source = factool.DuckParquetSource(os.getenv("QUOTESDAY_PATH"))
    price = source.get_factor("close_post", begin=begin, end=end)

    results = run_factor_pipeline(
        factor=[df],
        price=price,
        bucketing_mode=bucketing_mode,
        n_groups=n_groups,
        ts_n_jobs=ts_n_jobs,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results["ic"].reset_index().to_excel(writer, index=False, sheet_name="IC")
        results["group_returns"].reset_index().to_excel(
            writer, index=False, sheet_name="Group Return"
        )
        results["group_value"].reset_index().to_excel(
            writer, index=False, sheet_name="Group Value"
        )
        results["group_eval"].reset_index().to_excel(
            writer, index=False, sheet_name="Group Eval"
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
                results["ic_mean"].add_suffix("-ic-mean"),
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

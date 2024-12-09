import pandas as pd
import factorlab
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec


def load_factor_data(factor_name, ptype, data_path, period, ngroup, njobs, begin, end):
    """Load factor data and return performance results."""
    factor = factorlab.FactorManager(data_path)
    test_results = factor.performance(
        factor_name,
        ptype=ptype,
        period=period,
        begin=begin,
        end=end,
        ngroup=ngroup,
        njobs=njobs,
    )
    return test_results


def prepare_inforcoef_df(inforcoef):
    """Process the inforcoef data and compute rolling mean and cumulative values."""
    inforcoef_df = pd.DataFrame({
        "inforcoef": inforcoef,
        "rolling_mean": inforcoef.rolling(window=5).mean(),
        "cumulative": inforcoef.cumsum()
    })
    return inforcoef_df


def create_plots(crosssection, inforcoef_df, grouping, topk, out_path):
    """Create multiple plots and save them as PNG file."""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig)

    # Plot factor histogram
    ax_hist_factor = fig.add_subplot(gs[0, 0])
    ax_hist_factor.hist(crosssection["factor"], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax_hist_factor.set_title("Factor Histogram")
    ax_hist_factor.set_xlabel("Factor Value")
    ax_hist_factor.set_ylabel("Frequency")

    # Plot factor vs returns scatter plot
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_scatter.scatter(crosssection["factor"], crosssection["returns"], alpha=0.6, color='green')
    ax_scatter.set_title("Factor vs Returns")
    ax_scatter.set_xlabel("Factor Value")
    ax_scatter.set_ylabel("Returns")

    # Plot returns histogram
    ax_hist_returns = fig.add_subplot(gs[1, 0])
    ax_hist_returns.hist(crosssection["returns"], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax_hist_returns.set_title("Returns Histogram")
    ax_hist_returns.set_xlabel("Returns")
    ax_hist_returns.set_ylabel("Frequency")

    # Empty subplot (for future expansion)
    ax_blank = fig.add_subplot(gs[1, 1])
    ax_blank.axis('off')

    # Plot inforcoef with rolling mean and cumulative
    ax_inforcoef = fig.add_subplot(gs[0:2, 2:])
    ax_inforcoef.plot(inforcoef_df["inforcoef"], label="Inforcoef")
    ax_inforcoef.plot(inforcoef_df["rolling_mean"], label="5D Rolling Mean", linestyle="--")
    ax_cumulative = ax_inforcoef.twinx()
    ax_cumulative.plot(inforcoef_df["cumulative"], label="Cumulative Inforcoef", color="orange", linestyle=":")
    ax_inforcoef.set_title("Inforcoef with Rolling Mean & Cumulative")
    ax_inforcoef.legend(loc="upper left")
    ax_cumulative.legend(loc="upper right")

    # Plot net value for groups, TopK, and LongShort
    ax_combined = fig.add_subplot(gs[2, :])
    grouping["value"].plot(ax=ax_combined, label=["group1", "group2", "group3", "group4", "group5"])
    topk["value"].plot(ax=ax_combined, label="TopK Net Value", color="blue", linestyle="--")
    longshort_value = grouping["value"].iloc[:, -1]
    longshort_value.plot(ax=ax_combined, label="LongShort Net Value", color="red", linestyle=":")
    ax_combined.set_title("Net Value for Groups, TopK, and LongShort")
    ax_combined.legend(loc="best")
    ax_combined.set_xlabel("Date")
    ax_combined.set_ylabel("Net Value")

    # Adjust layout and save figure
    plt.tight_layout()
    fig.savefig(out_path)


def generate_factor_performance_report(factor_name, ptype, out_path, data_path, period, ngroup, njobs, begin, end):
    """Main function to load factor data, generate plots, and save the report."""
    test_results = load_factor_data(factor_name, ptype, data_path, period, ngroup, njobs, begin, end)

    crosssection = test_results["crosssection"]
    inforcoef = test_results["inforcoef"]
    grouping = test_results["grouping"]
    topk = test_results["topk"]

    inforcoef_df = prepare_inforcoef_df(inforcoef)

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Create plots and save the report
    create_plots(crosssection, inforcoef_df, grouping, topk, out_path)


if __name__ == "__main__":
    # User-configurable parameters for the report generation

    factor_name = "volume_weighted_price"  # The factor you want to analyze
    ptype = "volume_weighted_price"
    out_path = "out/report_volume_weighted_price.png"  # Output file path for the report
    data_path = "data/price_volume"  # Path to the data folder (default is "data/price_volume")

    # Performance settings
    period = -5  # Period for the performance calculation (e.g., -5 for 5 periods)
    ngroup = 5  # Number of groups for factor sorting
    njobs = -1  # Number of jobs for parallel execution, -1 means use all CPUs

    # Date range for the analysis
    begin = "2024-01-01"  # Start date
    end = "2024-12-02"  # End date

    # Generate the report with the specified parameters
    generate_factor_performance_report(
        factor_name=factor_name,
        ptype=ptype,
        out_path=out_path,
        data_path=data_path,
        period=period,
        ngroup=ngroup,
        njobs=njobs,
        begin=begin,
        end=end
    )

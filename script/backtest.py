import pandas as pd
import factorlab
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

# 测试用的参数设置
name = "price_volume_corr"
ptype = "volume_weighted_price"
factor = factorlab.FactorManager("data/price_volume")
out_path = Path(f"out/report_{name}.png")

# 获取测试结果
test_results = factor.performance(
    name,
    ptype=ptype,
    period=-5,
    begin="2015-01-01",
    end="2024-12-02",
    ngroup=5,
    n_jobs=1,
)

crosssection = test_results["crosssection"]  # 包含两列：'factor' 和 'returns'
inforcoef = test_results["inforcoef"]
grouping = test_results["grouping"]
topk = test_results["topk"]

inforcoef_df = pd.DataFrame({
    "inforcoef": inforcoef,
    "rolling_mean": inforcoef.rolling(window=5).mean(),
    "cumulative": inforcoef.cumsum()
})
out_path.parent.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 4, figure=fig)

# 第 1 行第 1 列：factor 的直方图
ax_hist_factor = fig.add_subplot(gs[0, 0])
ax_hist_factor.hist(crosssection["factor"], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax_hist_factor.set_title("Factor Histogram")
ax_hist_factor.set_xlabel("Factor Value")
ax_hist_factor.set_ylabel("Frequency")

# 第 1 行第 2 列：散点图（factor vs returns）
ax_scatter = fig.add_subplot(gs[0, 1])
ax_scatter.scatter(crosssection["factor"], crosssection["returns"], alpha=0.6, color='green')
ax_scatter.set_title("Factor vs Returns")
ax_scatter.set_xlabel("Factor Value")
ax_scatter.set_ylabel("Returns")

# 第 2 行第 1 列：returns 的直方图
ax_hist_returns = fig.add_subplot(gs[1, 0])
ax_hist_returns.hist(crosssection["returns"], bins=30, alpha=0.7, color='orange', edgecolor='black')
ax_hist_returns.set_title("Returns Histogram")
ax_hist_returns.set_xlabel("Returns")
ax_hist_returns.set_ylabel("Frequency")

# 占据第 2 行第 2 列：留白或可扩展区域（空子图）
ax_blank = fig.add_subplot(gs[1, 1])
ax_blank.axis('off')  # 留空

# 右上角：inforcoef的时间序列图
ax_inforcoef = fig.add_subplot(gs[0:2, 2:])
ax_inforcoef.plot(inforcoef_df["inforcoef"], label="Inforcoef")
ax_inforcoef.plot(inforcoef_df["rolling_mean"], label="5D Rolling Mean", linestyle="--")
ax_cumulative = ax_inforcoef.twinx()
ax_cumulative.plot(
    inforcoef_df["cumulative"], label="Cumulative Inforcoef", color="orange", linestyle=":"
)
ax_inforcoef.set_title("Inforcoef with Rolling Mean & Cumulative")
ax_inforcoef.legend(loc="upper left")
ax_cumulative.legend(loc="upper right")

# 左下角：分组的回测与 TopK/LongShort 的净值曲线合并图
ax_combined = fig.add_subplot(gs[2, :])
grouping["value"].plot(ax=ax_combined, label=["group1", "group2", "group3", "group4", "group5"])
topk["value"].plot(ax=ax_combined, label="TopK Net Value", color="blue", linestyle="--")
longshort_value = grouping["value"].iloc[:, -1]
longshort_value.plot(ax=ax_combined, label="LongShort Net Value", color="red", linestyle=":")
ax_combined.set_title("Net Value for Groups, TopK, and LongShort")
ax_combined.legend(loc="best")
ax_combined.set_xlabel("Date")
ax_combined.set_ylabel("Net Value")

# 调整整体布局并保存图像
plt.tight_layout()
fig.savefig(out_path)

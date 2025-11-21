## 资本利得突出量（Capital Gain Overhang, CGO）

### 定义

**资本利得突出量（CGO）** 旨在刻画投资者的平均持仓盈亏状态，基于“参考价格（Reference Price, RP）”构造：

- **参考价格 RP** 代表当前市场的“平均成本价”，通过历史周度换手率对过去价格进行前权重（front-weight）累积与归一化得到；
- **CGO** 衡量当前价格相对参考价格的偏离，数值越大表示投资者整体处于更高的平均浮盈，潜在的获利了结抛压越强，未来收益（通常）越高。

### 数学形式

设：

- $V_t$：第 $t$ 周的换手率（基于流通股本的周度换手率，取值区间 $[0,1]$）；
- $P_t$：第 $t$ 周末的后复权收盘价；
- $T$：回溯期（周数），常用 $T=260$（约 5 年）；
- $k$：对价格前权重进行归一化的系数。

**参考价格（Reference Price）：**

$$
RP_t = \frac{1}{k}\sum_{n=1}^{T}\left( V_{t-n}\prod_{\tau=1}^{n-1}(1 - V_{t-n+\tau}) \right) P_{t-n}
$$

其中 $\prod_{\tau=1}^{0}(\cdot)\equiv 1$。
直观上，权重：

$$
w_{t,n}=V_{t-n}\prod_{\tau=1}^{n-1}(1 - V_{t-n+\tau})
$$

等同于“在 $t-n$ 周买入、此后 $n-1$ 周未被卖出的存量比例”。

**归一化系数：**

$$
k=\sum_{n=1}^{T}w_{t,n}=\sum_{n=1}^{T}V_{t-n}\prod_{\tau=1}^{n-1}(1 - V_{t-n+\tau})
$$

保证所有权重之和为 1（若使用上式 $RP_t$ 的 $\tfrac{1}{k}$ 形式）。

**资本利得突出量（CGO）：**

$$
CGO_t = \frac{P_{t-1} - RP_t}{P_{t-1}}
$$

> **解释：** 当 $CGO_t$ 较大时，代表平均成本 $RP_t$ 远低于当前价格 $P_{t-1}$，投资者整体浮盈更高；
> 在 A 股等市场的经验中，CGO 与后续收益常呈**正相关**（即 $CGO$ 高 $\Rightarrow$ 未来收益更高），
> 可理解为高浮盈导致的短期价格压制与估值低估在后续得到修复。

### 数据与字段说明

该因子基于 `quotes_day` 数据表计算。主要字段如下：

| 变量                     | 对应字段          | 含义         | 备注             |
| ------------------------ | ----------------- | ------------ | ---------------- |
| 股票代码                 | `code`          | 股票唯一标识 | 用于分组计算     |
| 日期                     | `date`          | 交易日期     | 建议设为索引     |
| 收盘价（复权）           | `close_post`    | 后复权收盘价 | 用于价格$P_t$  |
| 成交量（用于计算换手率） | `volume`        | 当日成交量   | 用于$V_t$      |
| 流通股本                 | `circulation_a` | A 股流通股数 | 用于$V_t$      |
| 停牌标志                 | `suspended`     | 是否停牌     | 停牌日应跳过计算 |
| ST标志                   | `st`            | 是否ST       | ST应跳过计算     |

### 数据预处理与聚合

1. **日频换算为周频：**

   - 每周取周内每个交易日的 `close_post`平均 作为 $P_t$；
   - 计算当周换手率 $V_t$ 为：

     $$
     V_t = \frac{ \sum_{i \in \text{week }t}V_i}{\sum_{i \in \text{week }t} Circulation\_A_i }
     $$

     即“本周成交的总量 / 本周流通股本之和”
2. **处理异常值：**

   - 若停牌（`suspended=True`）或ST（st=True），则该日换手设为 0。
3. **计算流程：**

   - 筛选对应时间窗口的日线数据；
   - 对每支股票（`code`）按周分组；
   - 按时间顺序计算 $RP_t$ 与 $CGO_t$；

### 边界与实现细节

- **换手截断：** 若 $V_t$ 异常（>1 或 NaN），建议 Winsorize 到 $[0,1]$ 并缺失值前向填充。
- **回溯长度：** $T$ 越长，参考价格越平滑；极端低换手可减小 $T$ 或设置 $\epsilon$ 约束。
- **新股或停牌：** 样本不足 $T$ 时使用可得历史重算 $k$。
- **频率：** 本定义以周频为主；同时请计算日频的因子值，可将 $V_t$ 替换为日换手并令 $T$ 为日数。

### 参考文献

- Huijun W., Jinghua Y., Jianfeng Y. (2016). *Reference‑dependent preferences and the risk‑return trade-off.* Journal of Financial Economics, 123(2): 395–414.
- Grinblatt, M., & Han, B. (2005). *Prospect theory, mental accounting, and momentum.* Journal of Financial Economics, 78(2), 311–339.

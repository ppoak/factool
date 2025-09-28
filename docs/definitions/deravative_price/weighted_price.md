## 衍生价格因子

### 成交量加权价格因子（volume_weighted）

**定义**：

成交量加权价格因子是指按照成交量对同一时点的价格加权平均，反映主流资金交易价格水平。

**计算步骤**：

$$
\text{volume\_weighted} = \frac{\sum_{t} {P_t \times V_t}}{\sum_{t} V_t}
$$

其中 $P_t$ 为某时点价格（close_post），$V_t$ 为同期成交量（volume）。

### 时间加权价格因子（time_weighted）

**定义**：

时间加权价格因子表示在指定时间区间内的价格算术平均，仅考虑时间覆盖，不加任何成交量权重。

**计算步骤**：

$$
\text{time\_weighted} = \frac{1}{n} \sum_{t} P_t
$$

其中 $n$ 为样本数量，$P_t$ 为close_post。

### 尾盘成交量加权价格因子（tail_weighted）

**定义**：

尾盘成交量加权价格因子是指在14:30-15:00时段内，依据成交量对价格加权平均，反映尾盘的主流成交价格。

**计算步骤**：

1. 选取14:30-15:00时段的数据。
2. 计算区间内成交量加权平均价：

$$
\text{tail\_weighted} = \frac{\sum_{t \in \text{tail}} {P_t \times V_t}}{\sum_{t \in \text{tail}} V_t}
$$

### 盘初成交量加权价格因子（head_weighted）

**定义**：

盘初成交量加权价格因子为9:30-10:00时段内的成交量加权价格，衡量集合竞价和盘初第一波主导资金价格水平。

**计算步骤**：

1. 选取9:30-10:00时段的数据。
2. 计算区间内成交量加权平均价：

$$
\text{head\_weighted} = \frac{\sum_{t \in \text{head}} {P_t \times V_t}}{\sum_{t \in \text{head}} V_t}
$$

## 流动性因子

### 月度换手率（monthly_turnover）

**定义**：

月度换手率定义为一个月内每日交易量的总和与每日的流通股数的比值的对数。

**计算步骤**：

1. 提取计算时点t之前的周期T=21天对应时点，记为t+T。
2. 提取数据表quotes_day中t时刻到t+T时刻对应的volume数据矩阵和circulation_a矩阵，t时刻的st和suspended矩阵。
3. 计算每t+T时间内的所有成交量总和与每日流通股数总和：
   $$
   turnover = \frac {\sum_{i=t}^{t+T} volume_i} {\sum_{i=t}^{t+T} circulation\_a_i}
   $$
4. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的turnover数据，将他们设置为np.nan

### 季度换手率（quartly_turnover）

**定义**：

月度换手率定义为一个月内每日交易量的总和与每日的流通股数的比值的对数。

**计算步骤**：

1. 提取计算时点t之前的周期T=63天对应时点，记为t+T。
2. 提取数据表quotes_day中t时刻到t+T时刻对应的volume数据矩阵和circulation_a矩阵，t时刻的st和suspended矩阵。
3. 计算每t+T时间内的所有成交量总和与每日流通股数总和：

   $$
   turnover = \frac {\sum_{i=t}^{t+T} volume_i} {\sum_{i=t}^{t+T} circulation\_a_i}
   $$
4. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的turnover数据，将他们设置为np.nan

### 年度换手率（annually_turnover）

**定义**：

月度换手率定义为一个月内每日交易量的总和与每日的流通股数的比值的对数。

**计算步骤**：

1. 提取计算时点t之前的周期T=252天对应时点，记为t+T。
2. 提取数据表quotes_day中t时刻到t+T时刻对应的volume数据矩阵和circulation_a矩阵，t时刻的st和suspended矩阵。
3. 计算每t+T时间内的所有成交量总和与每日流通股数总和：

   $$
   turnover = \frac {\sum_{i=t}^{t+T} volume_i} {\sum_{i=t}^{t+T} circulation\_a_i}
   $$
4. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的turnover数据，将他们设置为np.nan

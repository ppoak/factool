## 波动率因子

### 一年内超额收益波动率因子（excessive_return_volatility）

**定义**：

一年内超额收益波动率因子定义为是过去252个交易日日超额收益率波动率，按照指数加权权重，半衰期为42个交易日。

**计算步骤**：

1. 提取计算时点t之前的周期T=252天对应时点，记为t+T。
2. close_post列、st列、suspended列从t+T到t的数据矩阵。
3. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的个股close_post数据。
4. 计算后复权收盘价对应的超额收益率，其中 $\bar{r _t}$表示t期所有收益的平均值，$\bar{r_t} = \frac{\sum_{i=1}^{N}r_{i, t}}{N} $。

$$
excessive\_return = r_t - \bar{r_t}
$$

5. 计算T期的超额收益平方的指数加权平均，其中$w_t$为指数加权平均的第t期系数，半衰期为两个月（42个交易日）。

$$
excessive\_return\_volatility = \frac{1}{T} \sum_{t=1}^T w_t (r_t - \bar{r_t}) ^2
$$

### 年度超额收益离差（yearly_excessive_deviation）

**定义**：

年度超额收益离差指的是过去十二个月每个月超额收益时间序列的离差

**计算步骤**：

1. 提取计算时点t之前的周期T=252天对应时点，记为t+T。
2. close_post列、st列、suspended列从t+T到t的数据矩阵。
3. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的个股close_post数据。
4. 通过对数收益率，构建12个月度收益率的时间序列$Z(t)$，其中$M=21$，$r_t = P_t / P_{t-1} - 1$。

   $$
   Z(T) = \sum_{t=1}^{M} ln(1 + r_t), T = 1, 2, ..., 12
   $$
5. 计算$Z(t)$序列的离差，记为因子值。

   $$
   yearly\_excessive\_deviation = Z(T)_{max} - Z(T)_{min}
   $$

### 残差波动率因子（residual_volatility）

**定义**：

残差波动率因子为通过市场收益及β解释股票收益率时的残差在时间序列上的波动率。

**计算步骤**：

1. 提取计算时点t之前的周期T=252天对应时点，记为t+T。
2. 提取close_post矩阵、st矩阵、suspended矩阵从t+T到t的数据。提取data/barra_beta数据表中的barra_beta的数矩阵为$\vec \beta$。
3. 提取quotes_day表中的close_post矩阵、st矩阵、suspended矩阵从t+T到t的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算股票的日收益率矩阵$R$。
4. 提取quotes_day表中的close矩阵和circulation_a矩阵，计算流通市值矩阵$M = close \times circulation\_a$。
5. 使用流通市值矩阵和日收益率矩阵$R$计算出市值加权的收益率矩阵得到市值加权的收益率作为市场收益率$R_M$。
6. 计算收益率矩阵后，计算残差波动率， 其中$r_t$为第t期的资产收益率，beta为t期资产收益率向量，$R_{M, t}$表示t期市场收益率：

$$
residual\_volatility = std(r_t - beta_t \times R_{M, t})
$$

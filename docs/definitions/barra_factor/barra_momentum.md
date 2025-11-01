## 动量因子

### 剔除最近期动量因子（nonrecent_momentum）

**定义**：

剔除最近期动量因子的定义为过去两年（504个交易日）内每日的对数收益率以半衰期为半年（126个交易日）的指数加权平均值，减去最近一个月（21个交易日）每日的对数收益率以半衰期为半年的指数加权平均的差值。


**计算步骤**：

1. 提取计算时点t之前的周期T=504天对应时点，记为t+T。
2. 提取数据表quotes_day中对应的close_post列、st列、suspended列从t+T到t的数据矩阵
3. 使用 ~st且 ~suspended数据作为掩码，过滤掉st或suspended的close_post数据
4. 按照如下公式计算因子，其中$ w_t$为t时刻对应的指数加权平均系数。

$$
\text{nonrecent\_momentum} = \Sigma_{t=L}^{T+L} w_{t} ln(1 + r_t) - \Sigma_{t=0}^{L} w_{t} ln(1 + r_t)
$$

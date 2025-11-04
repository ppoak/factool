## 市值因子

### 对数市值因子（log_market_size）

**定义**：

对数市值因子定义为股票流通股本与收盘价乘积的对数。

**计算步骤**：

提取计算时点t的quotes_day表中的close矩阵、st矩阵、suspended矩阵的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算流通市值矩阵$M=close \times circulatio\_a$，取对数$log(M)$作为因子，注意M为矩阵，返回因子时候需要转成Series。

### 非线性市值因子（nonlinear_market_size）

**定义**：

非线性市值因子是对数市值因子的三次项（立方）对其一次项线性回归后的残差。

**计算步骤**：

1. 提取计算时点t的quotes_day表中的close矩阵、st矩阵、suspended矩阵的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算流通市值矩阵$M=close \times circulatio\_a$，取对数$LM =log(M)$。
2. 以 $ LM^3 $ 为因变量，$LM $ 加常数项为自变量做线性回归：

   $$
   LM^3 = \beta_0 + \beta_1 LM + \epsilon
   $$

3. 用残差 $ \epsilon $ 作为非线性市值因子：

   $$
   nonlinear\_market\_size = \epsilon
   $$

## 市值因子

### 对数市值因子（log_market_size）

**定义**：  

对数市值因子定义为股票流通股本与收盘价乘积的对数。

**计算步骤**：

$$
\text{log\_market\_size} = \log(\text{circulation}_a \times \text{close\_post})
$$

其中 circulation_a 表示流通股本，close_post 表示收盘价。

对数市值因子用于衡量传统市值规模；  

### 非线性市值因子（nonlinear_market_size）

**定义**：  
非线性市值因子是对数市值因子的三次项（立方）对其一次项线性回归后的残差。

**计算步骤**：  

1. 首先计算对数市值： $ x = \log(\text{circulation}_a \times \text{close\_post}) $

2. 以 $ x^3 $ 为因变量，$ x $ 加常数项为自变量做线性回归：  
    $$
    x^3 = \beta_0 + \beta_1 x + \epsilon
    $$

3. 用残差 $ \epsilon $ 作为非线性市值因子：
    $$
    \text{nonlinear\_market\_size} = \epsilon
    $$

非线性市值因子捕捉市值效应中的非线性部分，可揭示超越线性规模效应的定价信息。
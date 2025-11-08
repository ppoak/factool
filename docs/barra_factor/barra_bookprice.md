## 账面市值比因子

### 账面市值比因子（barra_btop）

**定义**：

上个季报公司普通股权账面价值（净资产）除以公司当前的市值。

**计算步骤**：

提取计算时点t的quotes_day表中的close矩阵、st矩阵、suspended矩阵的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算流通市值矩阵$M=close \times circulatio\_a$；读取财务报告表financial_report中的total_equity矩阵的数据，计算total_equity与M的比值，作为因子。

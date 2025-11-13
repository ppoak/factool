## 盈利因子

### 盈利市值比因子（barra_earningprice）

**定义**：

公司营业收入（ttm）除以公司当前的市值。

**计算步骤**：

提取计算时点t的quotes_day表中的close矩阵、st矩阵、suspended矩阵的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算流通市值矩阵$M=close \times circulation\_a$；读取财务报告表financial_report中的operating_revenue矩阵的数据，计算operating_revenue与M的比值，作为因子。

### 现金流量市值比因子（barra_cashflowprice）

**定义**：

公司净现金流量（ttm）除以公司当前的市值。

**计算步骤**：

提取计算时点t的quotes_day表中的close矩阵、st矩阵、suspended矩阵的数据。使用 `~st & ~suspended`作为掩码，筛选掉停牌和ST的股票后，计算流通市值矩阵$M=close \times circulation\_a$；读取财务报告表financial_report中的net_inc_cash_and_equivalents矩阵的数据，计算net_inc_cash_and_equivalents与M的比值，作为因子。

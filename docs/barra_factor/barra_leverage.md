## 杠杆因子（Leverage Factors）

### 市场杠杆因子（market_leverage）

**定义**：

市场杠杆因子衡量公司市场层面的资本结构状况，表示普通股市值与**优先股**及**长期负债**账面价值之和相对于普通股市值的比值。

$$
market\_leverage = \frac{me + pe + ld}{me}
$$

其中：

- `me` 为普通股市值（market equity）；
- `pe` 为优先股账面价值；
- `ld` 为长期负债账面价值。

**计算步骤**：

1. 从 `quotes_day` 提取收盘价 `close` 与流通股数 `circulation_a`，计算：
   $$
   me = close \times circulation\_a
   $$
2. 从 `financial_report` 按报告期（ttm）提取：
   - 优先股账面价值 `preference_shares`（缺失按 0 处理）；
   - 长期负债账面价值 `non_current_liabilities`。
3. 将财报字段按股票代码与报告期与日频市值数据对齐（披露日后向前填充至下一次披露）。
4. 计算：
   $$
   market\_leverage = \frac{me + preferred\_shares + non\_current\_liabilities}{me}
   $$
5. 使用 ~st 且 ~suspended 作为掩码滤除 ST 与停牌样本。

### 账面杠杆因子（book_leverage）

**定义**：

账面杠杆因子反映公司账面层面的资本结构状况，表示普通股账面价值、优先股账面价值及长期负债账面价值之和相对于普通股账面价值的比值。

$$
book\_leverage = \frac{be + pe + ld}{be}
$$

其中：

- `be` 为普通股账面价值；
- `pe` 为优先股账面价值；
- `ld` 为长期负债账面价值。

**计算步骤：**

1. 从 `financial_report` 按报告期（ttm）提取：
   - 普通股账面价值 `total_equity`
   - 优先股账面价值 `preference_shares`
   - 长期负债账面价值 `non_current_liabilities`
2. 将财报字段按股票代码与报告期与日频市值数据对齐（披露日后向前填充至下一次披露）。
3. 计算：
   $$
   book\_leverage = \frac{total\_equity + preferred\_stock + non\_current\_liabilities}{total\_equity}
   $$
4. 使用 ~st 且 ~suspended 作为掩码滤除 ST 与停牌样本。

### 资产负债率因子（debt_to_asset）

**定义**：

资产负债率衡量公司资产中由负债融资的比例。

$$
debt\_to\_asset = \frac{td}{ta}
$$

其中：

- `td` 为总负债账面价值；
- `ta` 为总资产账面价值。

**计算步骤**：

1. 从 `financial_report` 按报告期（ttm）提取：
   - 总资产账面价值 `total_assets`
   - 总负债账面价值 `total_liabilities`
2. 将财报字段按股票代码与报告期与日频市值数据对齐（披露日后向前填充至下一次披露）。
3. 计算：
   $$
   debt\_to\_asset = \frac{total\_liabilities}{total\_assets}
   $$
4. 使用 ~st 且 ~suspended 作为掩码滤除 ST 与停牌样本。

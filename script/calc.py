# %% [markdown]
# # Factor Calculation
import contrib
import factorlab

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
source = factorlab.ParquetFactorSource("D:/Documents/DataBase/quotes_day")
calculator = contrib.MarketSize(source)

# %% [markdown]
# ## 2. Factor Calculation
factor_name = "market_sizes"
data = calculator.calc(factor_name, "2015-01-01", "now", n_jobs=-1)

# %% [markdown]
# ## 3. Save Factor
dumper = factorlab.ParquetFactorSource(f"data/{factor_name}")
dumper.save(data)

# %% [markdown]
# ## 4. Load Factor
dumper.get_factor("log_market_size_processed", begin="2025-01-01", end="now")

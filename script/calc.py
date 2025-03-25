# %% [markdown]
# # Factor Calculation
import contrib
import factorlab

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
source = factorlab.XtFactorSource(None, period="1d")
calculator = contrib.DerivativePrice(source)

# %% [markdown]
# ## 2. Factor Calculation
data = calculator.calc("weighted_price", "2024-03-24", "now", n_jobs=-1)

# %% [markdown]
# ## 3. Save Factor
dumper = factorlab.ParquetFactorSource("data/price_volume")
dumper.save("weighted_price", data, partition_col="month", partitioner=data.index.get_level_values(0).strftime(r"%Y-%m"))

# %%
# ## 4. Load Factor
# While using subclass for XtFactroDuckDB, we can load factor from duckdb file using DuckDBFactorSource
dumper.get_factor("weighted_price", "volume_weighted")

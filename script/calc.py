# %% [markdown]
# # Factor Calculation
import contrib

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
duckdb = "data/price_volume.db"
factor = contrib.DerivativePrice(None, duckdb)

# %% [markdown]
# ## 2. Factor Calculation
factor.calc("head_weighted_price", "2025-03-01", "now", n_jobs=-1)

# %%
# ## 3. Save Factor
factor.save()

# %%
# ## 4. Load Factor
# While using subclass for XtFactroDuckDB, we can load factor from duckdb file using DuckDBFactorSource
from factorlab import DuckDBFactorSource

name = "DerivativePrice"
source = DuckDBFactorSource(duckdb, name)
source.get_factor("head_weighted_price")

# %% [markdown]
# # Factor Calculation
import contrib

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
duckdb_path = "data/price_volume.db"
factor = contrib.DerivativePrice(None, duckdb_path)

# %% [markdown]
# ## 2. Factor Calculation
factor.calc("volume_weighted_price", "2024-03-13", "now", n_jobs=-1)

# %% [markdown]
# ## 3. Save Factor
factor.save()

# %%
# ## 4. Load Factor
# While using subclass for XtFactroDuckDB, we can load factor from duckdb file using DuckDBFactorSource
from factorlab import DuckDBFactorSource

name = "DerivativePrice"
source = DuckDBFactorSource(duckdb_path)
source.get_factor("volume_weighted_price")

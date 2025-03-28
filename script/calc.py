# %% [markdown]
# # Factor Calculation
import contrib
import factorlab

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
source = factorlab.ParquetFactorSource("D:/Documents/DataBase/quotes_day")
calculator = contrib.PriceVolumeDay(source)

# %% [markdown]
# ## 2. Factor Calculation
data = calculator.calc("compound_volume", "2018-01-01", "now", n_jobs=-1)

# %% [markdown]
# ## 3. Save Factor
dumper = factorlab.ParquetFactorSource("data/coumpound_volume")
dumper.save(
    data,
    partitioner=data.index.get_level_values(0).strftime(r"%Y-%m"),
)

# %% [markdown]
# ## 4. Load Factor
dumper.get_factor("corr_pos_pos", begin="2025-01-01", end="now")

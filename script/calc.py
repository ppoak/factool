# %% [markdown]
# # Factor Calculation
import contrib
import factorlab
import pandas as pd

# %% [markdown]
# ## 1. Factor Definition
# Currently, all factor definition should be written in contrib
day = factorlab.ParquetFactorSource("D:/Documents/DataBase/quotes_day")
minute = factorlab.ParquetFactorSource("D:/Documents/DataBase/quotes_min")
calculator = contrib.MomentumReverse([day, minute])

# %% [markdown]
# ## 2. Factor Calculation
factor_name = "decomposed_momentum"
data = calculator.calc(factor_name, day.get_times("2015-01-01", "now"), n_jobs=-1)

# %% [markdown]
# ## 3. Save Factor
dumper = factorlab.ParquetFactorSource(
    f"data/{factor_name}",
    grouper=pd.Grouper(key="time", freq="ME"),
)
dumper.save(data)

# %% [markdown]
# ## 4. Load Factor
dumper.get_factor("nonrecent_weekly_return_processed", begin="2025-01-01", end="now")

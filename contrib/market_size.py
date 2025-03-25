import numpy as np
import pandas as pd
import statsmodels.api as sm
from .factor import BaseFactor


class MarketSize(BaseFactor):

    def calc_market_sizes(self, time: str | pd.Timestamp) -> pd.Series:
        shares = self.source.get_factor("circulation_a", begin=time, end=time)
        price = self.source.get_factor("close", begin=time, end=time)
        adjfactor = self.source.get_factor("adjfactor", begin=time, end=time)
        log = np.log(shares * price * adjfactor).squeeze()
        model = sm.OLS((log**3).dropna(), sm.add_constant(log).dropna()).fit()
        nonlinear = model.resid
        return pd.concat(
            [log, nonlinear], axis=1, keys=["log_market_size", "nonlinear_market_size"]
        )

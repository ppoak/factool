from volatile import VolatileFactor
from marketsize import MarketSizeFactor
import pandas as pd
import numpy as np

name = 'capm_std_3m'
start = '20140104'
stop = '20140109'

factor = VolatileFactor("./data/factor")
data = factor.get(name, start=start, n_jobs= 1)
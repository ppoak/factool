from .base import FactorManager
from .base import (
    quotes_day,
    quotes_min,
    index_weights,
)


from .marketsize import MarketSizeFactor
from .retdist import RetDistFactor
from .price_volume import DeraPriceFactor, PriceVolumeCorr
from .voldist import VolDistFactor
from .volatile import VolatileFactor
from .capflow import CapFlowFactor
from .evaluation import EvaluationFactor
from .revmom import MomentumFactor
from .liquidity import LiquidityFactor

from .operators import (
    zscore, minmax,
    add, sub, mul, div,
    madoutlier, stdoutlier, iqroutlier,
    shift, corr, rank, group, where, mean,
    weightify, diff, absolute,
    rsum, rmean,
    sum, cumsum, cumprod,
    fillna, log, sqrt,
)


class Factor(
    MarketSizeFactor, 
    RetDistFactor, 
    DeraPriceFactor, 
    PriceVolumeCorr,
    VolDistFactor, 
    VolatileFactor, 
    CapFlowFactor, 
    EvaluationFactor, 
    MomentumFactor,
    LiquidityFactor
):
    pass

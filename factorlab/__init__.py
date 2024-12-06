from .datasource import (
    quotes_day, quotes_min
)

from .factor import Factor
from .marketsize import MarketSizeFactor
from .retdist import RetDistFactor
from .pricevolume import DeraPriceFactor, PriceVolumeCorr
from .voldist import VolDistFactor
from .volatile import VolatileFactor
from .capflow import CapFlowFactor
from .evaluation import EvaluationFactor
from .revmom import MomentumFactor
from .liquidity import LiquidityFactor

from .processors import (
    zscore, minmax,
    madoutlier, stdoutlier, iqroutlier,
    fillna,
    log, sqrt, tsmean,
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


__version__ = "0.2.0"

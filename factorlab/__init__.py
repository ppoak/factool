from .base import (
    BaseFactor, 
    quotes_day, quotes_min,
    stock_connect, financial, industry_info, 
    index_quotes_day, index_quotes_min, index_weights, 
    filter, prices, industry_returns, 
    barra_rq, 
    zscore, minmax,
    madoutlier, stdoutlier, iqroutlier,
    fillna, log, tsmean, sqrt, neutralization, 
)

from .marketsize import MarketSizeFactor
from .retdist import RetDistFactor
from .pricevolume import DeraPriceFactor, PriceVolumeCorr
from .voldist import VolDistFactor
from .volatile import VolatileFactor
from .capflow import CapFlowFactor
from .evaluation import EvaluationFactor
from .revmom import MomentumFactor
from .liquidity import LiquidityFactor



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

factor = Factor("./data/factor", code_level="order_book_id", date_level="date")
alpha = Factor("./data/alpha", code_level="order_book_id", date_level="date")
barra = Factor("./data/barra", code_level="order_book_id", date_level="date")
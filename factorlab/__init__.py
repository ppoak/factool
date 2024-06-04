from .base import (
    BaseFactor, 
    quotes_day, quotes_min,
    stock_connect, financial, 
    index_quotes_day, index_weights,
    zscore, minmax,
    madoutlier, stdoutlier, iqroutlier,
    fillna, log, tsmean,
)

from .marketsize import MarketSizeFactor
from .retdist import RetDistFactor
from .deraprice import DeraPriceFactor
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
    VolDistFactor, 
    VolatileFactor, 
    CapFlowFactor, 
    EvaluationFactor, 
    MomentumFactor,
    LiquidityFactor
):
    pass
    
factor = Factor("./data/factor", code_level="order_book_id", date_level="date")

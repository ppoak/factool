import pandas as pd
import numpy as np
import quool as q
from joblib import Parallel, delayed
from tqdm import  tqdm
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore")


fidxwgt = q.Factor("/home/wiikai/factor/data/index-weights", code_level="order_book_id", date_level="date")
factor = q.Factor("/home/wiikai/factor/data/factor", code_level="order_book_id", date_level="date")
fqtd = q.Factor("/home/wiikai/factor/data/quotes-day", code_level="order_book_id", date_level="date")
industry = q.Factor("/home/wiikai/factor/data/industry-info", code_level="order_book_id", date_level="date")
fqtd = q.Factor("/home/wiikai/factor/data/quotes-day", code_level="order_book_id", date_level="date")


class TestFactor(q.Factor):
    def get_barra_return(self,date: str):
        rollback = fqtd.get_trading_days_rollback(date, 1)
        price = fqtd.read('close', start=rollback, stop=date)
        _adj = fqtd.read('adjfactor', start=rollback, stop=date)
        shares = fqtd.read("circulation_a", start=date, stop=date)

        # 个股市值权重
        size = (shares * price * _adj).loc[date]
        weight = size.apply(lambda x: x / size.sum())
        weight.name = 'weight'

        # 回归权重矩阵
        weight_mat = size.apply(lambda x: np.sqrt(x) / np.sqrt(size.sum()))
        weight_mat.name = 'weight_mat'

        # 不考虑rf的超额收益
        ret = (price * _adj).pct_change(fill_method=None).loc[date]
        ret.name = 'ret'

        ind = industry.read('first_industry_name',start=date, stop=date).loc[date]
        ind = pd.get_dummies(ind, drop_first=True)
        ind = ind.select_dtypes(include=[bool]).astype(int) 
        industry_columns = ind.columns # 获取行业名字
        ind['国家'] = 1

        df = factor.read('barra_log_marketcap,beta,rstr,residual_volatility,barra_nonlinear_size,book_to_price,liquidity,barra_leverage',start=date, stop=date)
        df = df.reset_index(level='date', drop=True)
        df = df.apply(lambda x: (x - np.sum(x * weight) )/ x.std()) # 因子截面标准化
        df = pd.concat([df,ind,ret], axis=1).dropna()

        # 匹配长度
        industry_columns = industry_columns.intersection(df.columns)
        weight_mat = weight_mat.loc[df.index]

        # 计算因子收益
        X = df.drop(['ret'], axis=1)
        y = df['ret']

        # 设置变量
        N, K = X.shape
        beta = cp.Variable(K)
        residuals = y.values - X.values @ beta

        # 加权最小化残差平方和
        weighted_residuals = cp.multiply(weight_mat, residuals)
        objective = cp.Minimize(cp.sum_squares(weighted_residuals))

        # 添加约束条件
        constraints = 0
        for col in industry_columns:
            industry_weight = (weight * df[col]).sum()
            industry_index = X.columns.get_loc(col)
            constraints += beta[industry_index] * industry_weight

        problem = cp.Problem(objective, [constraints == 0])
        problem.solve()

        # 获取回归系数
        res = pd.Series(beta.value,index = df.drop(['ret'], axis=1).columns)
        res.name = date
        return res   
    
    def get(self, name: str, start: str = None, stop: str = None, n_jobs: int = -1):
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index().loc[start:stop]
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]
        

testfactor = TestFactor("/home/wiikai/factor/data/barra-returns")

data = testfactor.get("barra_return", start='20231222', stop="20240101", n_jobs= 1)
data.index.name = 'date'
print(data)

# barra = q.DatetimeTable ('/home/data/barra-returns')
# barra.update(data)
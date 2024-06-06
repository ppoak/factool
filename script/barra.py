import quool
import cvxpy as cp
import numpy as np
import pandas as pd
import factorlab as lab
import dataforge as forge
from tqdm import  tqdm
from joblib import Parallel, delayed


class BarraReturn(quool.DatetimeTable, lab.Factor):

    def get_barra_return(self, date: str):

        rollback = lab.quotes_day.get_trading_days_rollback(date, 1)
        price = lab.quotes_day.read('close', start=rollback, stop=date)
        _adj = lab.quotes_day.read('adjfactor', start=rollback, stop=date)
        shares = lab.quotes_day.read("circulation_a", start=date, stop=date)

        # 个股市值权重
        size = (shares * price * _adj).loc[date]
        weight = size.apply(lambda x: x / size.sum())
        weight.name = 'weight'

        # 回归权重矩阵
        reg_weight = size.apply(lambda x: np.sqrt(x) / np.sqrt(size.sum()))
        reg_weight.name = 'reg_weight'
        # V = np.diag(reg_weight)

        # 不考虑rf的超额收益
        ret = (price * _adj).pct_change(fill_method=None).loc[date]
        ret.name = 'ret'

        ind = forge.industry_info.read('first_industry_name',start=date, stop=date)
        ind = ind.reset_index(level='date', drop=True)
        ind = pd.get_dummies(ind, prefix='', prefix_sep='')
        ind = ind.select_dtypes(include=[bool]).astype(int) 
        industry_columns = ind.columns # 获取行业名字
        ind['国家'] = 1

        # 因子数据
        df = lab.factor.read(
            'log_marketcap, market_beta, nonrecent_momentum, residual_volatility,'
            'nonlinear_size, book_to_price, compound_turnover, compound_leverage',
            start=rollback, stop=rollback
        )
        df = df.reset_index(level='date', drop=True)
        df = df.apply(lambda x: (x - np.sum(x * weight))/ x.std()) # 因子截面标准化
        df = pd.concat([df,ind,ret], axis=1).dropna()

        # 匹配长度
        industry_columns = industry_columns.intersection(df.columns)
        reg_weight = reg_weight.loc[df.index]
        
        # 计算因子收益
        X = df.drop(['ret'], axis=1)
        y = df['ret']

        # 设置变量
        N, K = X.shape
        f = cp.Variable(K)
        w = cp.Variable((K, N))
        residuals = y.values - X.values @ f

        # 加权最小化残差平方和
        weighted_residuals = cp.multiply(reg_weight, residuals)
        objective = cp.Minimize(cp.sum_squares(weighted_residuals))

        # r = f * X 
        # where f = w * r
        # r = (w * r) * X
        # constraint：when ∑(w(i) * X) = 1 ， ∑(w(other) * X) = 0
        # 添加约束条件
        industry_constraints = 0
        for col in industry_columns:
            industry_weight = (weight * df[col]).sum()
            industry_index = X.columns.get_loc(col)
            industry_constraints += f[industry_index] * industry_weight

        f_constraint = [f == w @ y.values]

        exposure_constraints = [(w @ X.values)== np.eye(K)]

        problem = cp.Problem(objective, [industry_constraints == 0] + f_constraint + exposure_constraints)
        problem.solve()

        # 获取回归系数
        res = pd.Series(f.value, index=df.drop(['ret'], axis=1).columns)
        res.name = date
        return res   

barrareturn = BarraReturn("./data/barra-returns")

data = barrareturn.get("barra_return", start='20140104', stop="20240101", n_jobs= 1)
# data.index.name = 'date'
# barrareturn.update(data)
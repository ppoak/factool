import quool
import cvxpy as cp
import numpy as np
import pandas as pd
import factorlab as lab
import dataforge as forge
from tqdm import  tqdm
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

class BarraReturn(quool.DatetimeTable, lab.Factor):

    def get_barra_return(self, date: str):
        # r = f * X 
        # where f = w * r
        # r = (w * r) * X
        # constraint：when ∑(w(i) * X) = 1 ， ∑(w(other) * X) = 0

        rollback = lab.quotes_day.get_trading_days_rollback(date, 1)
        price = lab.quotes_day.read('close', start=rollback, stop=date)
        _adj = lab.quotes_day.read('adjfactor', start=rollback, stop=date)
        shares = lab.quotes_day.read("circulation_a", start=date, stop=date)
        size = (shares * price * _adj).loc[date]

        # 行业因子
        ind = forge.industry_info.read('first_industry_name',start=date, stop=date)
        ind = ind.reset_index(level='date', drop=True)
        ind = pd.get_dummies(ind, prefix='', prefix_sep='')
        ind = ind.select_dtypes(include=[bool]).astype(int) 

        # 风格因子
        style = lab.factor.read(
            'log_marketcap, market_beta, nonrecent_momentum, residual_volatility,'
            'nonlinear_size, book_to_price, compound_turnover, compound_leverage',
            start=rollback, stop=rollback
        )
        style = style.reset_index(level='date', drop=True)
        scaler = StandardScaler()
        scaled_style = pd.DataFrame(scaler.fit_transform(style), index=style.index, columns=style.columns)

         # 国家因子
        country = pd.Series(np.ones(len(style.index)), index=style.index, name='country').to_frame()
        X = pd.concat([country, ind, scaled_style], axis=1).dropna()

        # 回归权重矩阵
        reg_weight = np.sqrt(size) / np.sqrt(size).sum()
        reg_weight = reg_weight.reindex(X.index)
        reg_weight.index.name = ''
        V = pd.DataFrame(np.diag(reg_weight), index=reg_weight.index, columns=reg_weight.index)

        # 约束矩阵 K*K-1
        N, K = X.shape
        R = np.diag(np.ones(K))
        R = np.delete(R, len(ind.columns), axis=1)

        ind_weight = ind.mul(reg_weight, axis=0).sum()
        ind_weight_adj = -ind_weight.div(ind_weight.iloc[-1]).iloc[:-1]
        R[len(ind.columns),1:len(ind_weight_adj)+1] = ind_weight_adj.values
        R = pd.DataFrame(R, index=X.columns, columns=X.columns.drop(ind.columns[-1]))

        # 带约束，且考虑异方差的股票权重矩阵 omega = R @ (R.T @ X.T @ V @ X @ R)^-1 @ R.T @ X.T @ V
        W = np.dot(R.dot(np.linalg.inv(R.T.dot(X.T).dot(V).dot(X).dot(R))),R.T).dot(X.T).dot(V)
        W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，但考虑异方差的股票权重矩阵，可以直接WLS。omega = (X.T @ V @ X)^-1 @ X.T @ V
        # W = np.linalg.inv(X.T.dot(V).dot(X)).dot(X.T).dot(V)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，无异方差的股票权重矩阵，可以直接OLS。omega = (X.T @ X)^-1 @ X.T
        # W = np.linalg.inv(X.T.dot(X)).dot(X.T)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)
        
        # 因子收益率, r=不考虑rf的超额收益
        r = (price * _adj).pct_change(fill_method=None).loc[date]
        r = r.reindex(X.index)
        r.name = 'ret'

        f = W.dot(r)
        f.name = date
        return f   

barrareturn = BarraReturn("./data/barra-returns")

data = barrareturn.get("barra_return", start='20140104', stop="20240101", n_jobs= 1)
# data.index.name = 'date'
# barrareturn.update(data)
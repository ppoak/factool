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
        # r = f * X 
        # where f = w * r
        # r = (w * r) * X
        # constraint：when ∑(w(i) * X) = 1 ， ∑(w(other) * X) = 0
        rollback = lab.quotes_day.get_trading_days_rollback(date, 2)
        price = lab.quotes_day.read('close', start=rollback, stop=date)
        _adj = lab.quotes_day.read('adjfactor', start=rollback, stop=date)
        shares = lab.quotes_day.read("circulation_a", start=rollback, stop=date)
        size = (shares * price * _adj).loc[rollback]
        

        # 行业因子
        ind_columns = ['交通运输', '传媒', '农林牧渔', '医药', '商贸零售', '国防军工', '基础化工', '家电', '建材', '建筑',
                       '房地产', '有色金属', '机械', '汽车', '消费者服务', '煤炭', '电力及公用事业', '电力设备及新能源', '电子',
                       '石油石化', '纺织服装', '综合', '计算机', '轻工制造', '通信', '钢铁', '银行', '非银行金融', '食品饮料']
        ind = forge.industry_info.read('first_industry_name',start=rollback, stop=rollback)
        ind = ind.reset_index(level='date', drop=True)
        ind = pd.get_dummies(ind, prefix='', prefix_sep='').loc[:,ind_columns]
        ind = ind.select_dtypes(include=[bool]).astype(int) 

        # 风格因子
        style_columns = ['log_marketcap', 'market_beta', 'nonrecent_momentum', 'residual_volatility','nonlinear_size', 
                        'book_to_price', 'compound_turnover', 'compound_leverage']
        style = pd.DataFrame()
        for col in style_columns:
            data = lab.barra_factor.read(col, start=rollback, stop=rollback)
            data = lab.zscore(data)
            data = data.unstack()
            data.name = col
            if style.empty:
                style = data
            else:
                style = pd.concat([data,style] ,axis=1)
        style = style.reset_index(level='date', drop=True)

         # 国家因子
        country = pd.Series(np.ones(len(style.index)), index=style.index, name='country').to_frame()
        X = pd.concat([country, ind, style], axis=1).dropna()

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

        # 带约束，且考虑异方差的股票权重矩阵。 omega = R @ (R.T @ X.T @ V @ X @ R)^-1 @ R.T @ X.T @ V
        W = np.dot(R.dot(np.linalg.inv(R.T.dot(X.T).dot(V).dot(X).dot(R))),R.T).dot(X.T).dot(V)
        W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，但考虑异方差的股票权重矩阵，可以直接WLS。omega = (X.T @ V @ X)^-1 @ X.T @ V
        # W = np.linalg.inv(X.T.dot(V).dot(X)).dot(X.T).dot(V)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，无异方差的股票权重矩阵，可以直接OLS。omega = (X.T @ X)^-1 @ X.T
        # W = np.linalg.inv(X.T.dot(X)).dot(X.T)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)
        
        # 因子收益率, r=不考虑rf的超额收益
        # r = lab.barra_factor.get_future(start=date, stop=date) #用vwap计算收益率
        r = (price * _adj).pct_change(fill_method=None).loc[date]
        r = r.reindex(X.index).dropna()
        r.name = 'ret'

        W = W.loc[:,r.index]
        f = W.dot(r)
        # f.name = date
        f.name = rollback
        return f   

barrareturn = BarraReturn("./data/barra-returns")
data = barrareturn.get("barra_return", start='20140201', stop="20240101", n_jobs= 10)
# print(data)
# data.index.name = 'date'
# barrareturn.update(data)
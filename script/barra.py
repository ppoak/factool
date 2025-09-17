import quool
import numpy as np
import pandas as pd
import factool as lab
import dataforge as forge
class BarraReturn(quool.DatetimeTable, lab.Factor):

    def get_barra_return(self, date: str):
        # r = f * X 
        # where f = w * r
        # r = (w * r) * X
        # constraint：when ∑(w(i) * X) = 1 ， ∑(w(other) * X) = 0
        rollback = lab.quotes_day.get_trading_days_rollback(date, 1)
        price = lab.quotes_day.read('close', start=rollback, stop=date)
        _adj = lab.quotes_day.read('adjfactor', start=rollback, stop=date)
        shares = lab.quotes_day.read("circulation_a", start=rollback, stop=date)
        size = (shares * price * _adj).loc[rollback]
        
        # 行业因子
        ind = forge.industry_info.read('first_industry_name',start=rollback, stop=rollback)
        ind = ind.reset_index(level='date', drop=True)
        ind = pd.get_dummies(ind, prefix='', prefix_sep='')
        ind = ind.select_dtypes(include=[bool]).astype(int) 

        # 风格因子
        barra_factor = ['log_marketcap', 'market_beta', 'nonrecent_momentum', 'residual_volatility','nonlinear_size', 
                        'book_to_price', 'compound_turnover', 'compound_leverage']
        data = lab.barra.read(barra_factor, start=rollback, stop=rollback)
        nonrealizable = lab.filter.read('cn',start=rollback, stop=rollback).squeeze()
        data = data[~data.index.get_level_values('order_book_id').isin(nonrealizable[nonrealizable].index)]
        data = lab.zscore(lab.madoutlier(data, dev=5, drop=False))
        data = data.reset_index(level='date', drop=True)

         # 国家因子
        country = pd.Series(np.ones(len(data.index)), index=data.index, name='country').to_frame()
        
        # 合并
        common_index = ind.index.intersection(data.index)
        country = country.loc[common_index]
        data = data.loc[common_index]
        ind = ind.loc[common_index]
        X = pd.concat([country, ind, data], axis=1).fillna(0)

        # 回归权重矩阵，每个资产特有风险的不同
        size = size.reindex(X.index)
        reg_weight = (np.sqrt(size) / np.sqrt(size).sum()).fillna(0)
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
        
        # LinearRegression 假设同方差性，没有V矩阵
        # 带约束，且考虑异方差的股票权重矩阵。 omega = R @ (R.T @ X.T @ V @ X @ R)^-1 @ R.T @ X.T @ V
        try:
            W = np.dot(R.dot(np.linalg.inv(R.T.dot(X.T).dot(V).dot(X).dot(R))),R.T).dot(X.T).dot(V)
        except:
            W = np.dot(R.dot(np.linalg.pinv(R.T.dot(X.T).dot(V).dot(X).dot(R))),R.T).dot(X.T).dot(V)
        W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，但考虑异方差的股票权重矩阵。omega = (X.T @ V @ X)^-1 @ X.T @ V
        # W = np.linalg.inv(X.T.dot(V).dot(X)).dot(X.T).dot(V)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)

        # 无约束，无异方差的股票权重矩阵。omega = (X.T @ X)^-1 @ X.T
        # W = np.linalg.inv(X.T.dot(X)).dot(X.T)
        # W = pd.DataFrame(W, index=X.columns, columns=X.index)
        
        # 因子收益率, r=不考虑rf的超额收益
        # r = lab.barra_factor.get_future(start=date, stop=date) #用vwap计算收益率
        # W.dot(X).round(4) 验证
        r = (price * _adj).pct_change(fill_method=None).loc[date]
        r = r.reindex(X.index).fillna(0)

        W = W.loc[:,r.index]
        f = W.dot(r)
        f.name = rollback
        return f   

barrareturn = BarraReturn("./data/barra-returns")
data = barrareturn.get("barra_return", start='20160105', stop="20240101", n_jobs= 5)
# print(data)
# data.index.name = 'date'
# barrareturn.update(data)
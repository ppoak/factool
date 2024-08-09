import os
import sys
sys.path.append(os.chdir('/home/rice/huangweikai/CvxpyLayer'))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import TimeSliceDataset, cov_matrix_sqrt_svd, zscore, madoutlier
from model import RiskBudgetModel, train_model
import quool
import random

cvlayer = quool.Factor("./data/cv-layer-factor", code_level="order_book_id", date_level="date")
cvret = quool.Factor("./data/cv-layer-ret", code_level="order_book_id", date_level="date")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def evaluate_model():
    model = RiskBudgetModel(input_dim=55, hidden_dim=10, output_dim=5, lower=0.05, upper=0.3)
    model.load_state_dict(torch.load('FactorModel.pth', weights_only=True))
    model.eval()

    input_data = cvlayer.read('1d_ret, 2d_ret, 3d_ret, 4d_ret, 5d_ret, 10d_ret, 10d_std, 20d_ret, 20d_std, 30d_ret, 30d_std', start='20230101', stop='20240601').swaplevel('date', 'order_book_id').sort_index().drop(columns='20d_future_ret')
    future = cvlayer.read('20d_future_ret',start='20230601', stop='20240601')
    cov_matrix = input_data['1d_ret'].unstack().rolling(30).cov().dropna(how='all')
    cov_matrix.columns.name = ''
    input_data = pd.concat([input_data, cov_matrix], axis=1).loc['20230601':]
    
    cov_columns = cov_matrix.columns
    results = {}
    for date, group in input_data.groupby(level='date'):
        data = group.drop(columns=cov_columns).values
        Q_sqrt = cov_matrix_sqrt_svd(group[cov_columns].values)
        Q_sqrt_tensor = torch.tensor(Q_sqrt, dtype=torch.float32).unsqueeze(0)
        data_tensor = torch.tensor(data, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            predictions = model(data_tensor, Q_sqrt_tensor)
        results[date] = predictions.numpy()

    weights = pd.DataFrame(
    {date: values.flatten() for date, values in results.items()}, index =future.columns).T
    net_val = (weights * future).sum(axis=1)
    return weights, net_val

def main():
    set_seed(0)
    processor= [(madoutlier,{'dev': 5, 'drop': False}), zscore]
    data = cvlayer.read('1d_ret, 2d_ret, 3d_ret, 4d_ret, 5d_ret, 10d_ret, 10d_std, 20d_ret, 20d_std, 30d_ret, 30d_std',stop='20230501', processor=processor).swaplevel('date', 'order_book_id').sort_index()
    future = cvlayer.read('20d_future_ret',stop='20230501').stack().to_frame(name='future')
    cov_matrix = data['1d_ret'].unstack().rolling(30).cov().dropna(how='all')
    cov_matrix.columns.name = ''
    data = pd.concat([data, cov_matrix, future], axis=1).loc['20170601':]

    dataset = TimeSliceDataset(data)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # for features, Q_sqrt, label in dataloader:    
    #     print("features Shape:", features.shape)
    #     print("cov_matrix Shape:", Q_sqrt.shape)
    #     print("label Shape:", label.shape)
        
    #     print("features Element:", features[0])
    #     print("cov_matrix Element:", Q_sqrt[0])
    #     print("label Element:", label[0])
    #     break

    model = RiskBudgetModel(input_dim=55, hidden_dim=10, output_dim=5, lower=0.05, upper=0.25)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_model(dataloader, model, optimizer)

if __name__ == '__main__':
    main()
    # weights, net_val = evaluate_model()
import os
import sys
sys.path.append(os.chdir('/home/rice/huangweikai/CvxpyLayer'))

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TimeSliceDataset, RiskBudgetModel, cov_matrix_sqrt_svd 
import quool

cvlayer = quool.Factor("./data/cv-layer-factor", code_level="order_book_id", date_level="date")
cvret = quool.Factor("./data/cv-layer-ret", code_level="order_book_id", date_level="date")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)

def main():
    set_seed(0)

    data = cvlayer.read(stop='20230501').swaplevel('date', 'order_book_id').sort_index()
    cov_matrix = data['1d_ret'].unstack().rolling(30).cov().dropna(how='all')
    cov_matrix.columns.name = ''
    data = pd.concat([data, cov_matrix], axis=1).loc['20170601':]

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

    epochs = 50
    early_stopping = 10
    patience_counter = 0
    best_loss = np.inf

    for epoch in range(epochs):
        epoch_loss = 0.0
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if idx % 5 == 0:
            #     print('Current epoch: %d, Current batch: %d, Loss is %.3f' %(epoch+1,idx+1,loss.item()))

        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'FactorModel2.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print("Early stopping")
            break

def eval():
    model = RiskBudgetModel(input_dim=55, hidden_dim=10, output_dim=5, lower=0.05, upper=0.3)
    model.load_state_dict(torch.load('FactorModel2.pth', weights_only=True))
    model.eval()

    input_data = cvlayer.read(start='20230101', stop='20240601').swaplevel('date', 'order_book_id').sort_index().drop(columns='20d_future_ret')
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

    future = cvlayer.read('20d_future_ret',start='20230601', stop='20240601')
    weights = pd.DataFrame(
    {date: values.flatten() for date, values in results.items()}, index =future.columns).T
    weights = weights.div(weights.sum(axis=1), axis=0)
    net = (weights * future).sum(axis=1)
    return weights, net

if __name__ == '__main__':
    # main()
    weights, net = eval()



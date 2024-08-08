import numpy as np
import cvxpy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cvxpylayers.torch import CvxpyLayer


def cov_matrix_sqrt_svd(cov_matrix):
    U, S, Vt = np.linalg.svd(cov_matrix)
    return U @ np.diag(np.sqrt(S)) @ Vt

def minmax(tensor):
    min_vals = tensor.min(dim=1, keepdim=True)[0]
    max_vals = tensor.max(dim=1, keepdim=True)[0]
    return (tensor - min_vals) / (max_vals - min_vals)

class TimeSliceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.time_slices = data.index.get_level_values(0).unique()
        self.cov_matrix = data.index.get_level_values(1).unique()
        self.label = '20d_future_ret'

    def __len__(self):
        return len(self.time_slices)

    def __getitem__(self, idx):
        time_point = self.time_slices[idx]
        slice_data = self.data.loc[time_point]

        features = slice_data.drop(columns=[self.label] + list(self.cov_matrix)).values
        label = slice_data[self.label].values

        try:
            cov_matrix = slice_data[self.cov_matrix].values
            Q_sqrt = cov_matrix_sqrt_svd(cov_matrix)
            Q_sqrt_tensor = torch.tensor(Q_sqrt, dtype=torch.float32)
        except:
            Q_sqrt_tensor = None

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return features_tensor, Q_sqrt_tensor, label_tensor

class RiskBudgetModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, lower, upper):
        super(RiskBudgetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=-0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hardtanh = nn.Hardtanh(min_val=lower, max_val=upper)

        n = 5
        c = 0.1 # 最低风险预算
        self.b = cp.Parameter(n, nonneg=True) # 风险预算，前向传播
        self.Q_sqrt = cp.Parameter((n, n)) # 斜方差矩阵的平方根

        self.w = cp.Variable(n)   

        self.obj = cp.Minimize(cp.sum_squares(self.Q_sqrt @ self.w)) # 最小化组合的方差，控制总风险

        self.cons = [
            self.w >= 0,  
            self.b.T @ cp.log(self.w) >= c, # 每个资产的权重 w 满足特定的风险分配，对数函数线性化一些非线性关系具有凸优化的特性
        ]

        self.prob = cp.Problem(self.obj, self.cons)
        self.cvxpy_layer = CvxpyLayer(
            self.prob, 
            parameters=[self.b, self.Q_sqrt], 
            variables=[self.w]
        )

    def forward(self, x, Q_sqrt):
        # x 的形状应为 [batch_size, 5, 11]
        b = self.fc1(x.view(x.size(0), -1)) # [batch_size, 5*11]
        b = self.leaky_relu(b)
        b = self.fc2(b)
        b = self.softmax(b)
        b = self.hardtanh(b)
        b = minmax(b)

        # Loop over batch dimension
        weights = []
        for i in range(b.shape[0]):  
            w, = self.cvxpy_layer(b[i], Q_sqrt[i])
            weights.append(w)
        return torch.stack(weights)

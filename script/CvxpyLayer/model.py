import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from preprocess import minmax
class RiskBudgetModel(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, lower, upper):
        super(RiskBudgetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=-0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.hardtanh = nn.Hardtanh(min_val=lower, max_val=upper)

        n = 5
        self.c = nn.Parameter(torch.tensor(0.1)) # 较大的 c 倾向于更分散的权重分布
        b = cp.Parameter(n, nonneg=True) # 风险预算，前向传播
        Q_sqrt = cp.Parameter((n, n)) # 斜方差矩阵的平方根
        y = cp.Variable(n)   

        obj = cp.Minimize(cp.sum_squares(Q_sqrt @ y)) # 最小化组合的方差，控制总风险

        cons = [
            y >= 0, 
            b.T @ cp.log(y) >= self.c.detach().numpy(), # 每个资产满足特定的风险分配，对数函数线性化一些非线性关系具有凸优化的特性
        ]

        prob = cp.Problem(obj, cons)
        self.cvxpy_layer = CvxpyLayer(
            prob, 
            parameters=[b, Q_sqrt], 
            variables=[y]
        )

    def forward(self, x, Q_sqrt):
        # x 的形状应为 [batch_size, 5, 11]
        b = self.fc1(x.view(x.size(0), -1)) # [batch_size, 5*11]
        b = self.leaky_relu(b)
        b = self.fc2(b)
        b = self.softmax(b)
        b = self.hardtanh(b)
        b = minmax(b)

        y, = self.cvxpy_layer(b, Q_sqrt)
        w = y / y.sum(dim=1, keepdim=True)
        return w
    
        # b = self.fc1(x)
        # b = self.leaky_relu(b)
        # b = self.fc2(b)
        # b = self.softmax(b)
        # b = self.hardtanh(b)
        # b = minmax(b)
        # b = b.view(b.size(0), -1)

        # y, = self.cvxpy_layer(b, Q_sqrt)
        # w = y / y.sum(dim=1, keepdim=True)
        # return w
    
def train_model(train_dataloader,test_dataloader, model, optimizer, epochs=50, early_stopping=10):
    patience_counter = 0
    best_loss = np.inf
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # if idx % 5 == 0:
            #     print('Current epoch: %d, Current batch: %d, Loss is %.3f' %(epoch+1,idx+1,loss.item()))

        epoch_loss /= len(train_dataloader)
        train_losses.append(epoch_loss)

        model.eval()
        test_loss = test_model(test_dataloader, model)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss}, Test Loss: {test_loss}')

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'FactorModel.pth')
        else:
            patience_counter += 1

        if patience_counter >= early_stopping:
            print("Early stopping")
            break

    return train_losses, test_losses

def test_model(dataloader, model):
    total_loss = 0.0
    with torch.no_grad():
        for idx, (batch_features, batch_Q_sqrt, batch_labels) in enumerate(dataloader):
            weights = model(batch_features, batch_Q_sqrt)
            ret = torch.mul(weights, batch_labels)
            loss = -torch.sum(ret)
            total_loss += loss.item()
    return total_loss / len(dataloader)
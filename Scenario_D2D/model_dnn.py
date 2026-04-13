# =============================================================================
# Filename: model_dnn.py
# Description: Deep Neural Network (DNN) for Wireless Power Control
#
# Deep Neural Network Model (DNN)
# Refer: 【说明2】 DNN from TensorFlow to PyTorch
# Attention: this dnn model was rewritten in PyTorch, and is almost completely
#   different from the original TensorFlow version (W1P1) in structure and code.
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DNN_T1(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers): 
        super(DNN_T1, self).__init__()
        self.layers = nn.ModuleList()
        curr_dim = n_input
        for h_dim in hidden_layers:
            self.layers.append(nn.Linear(curr_dim, h_dim))
            self.layers.append(nn.BatchNorm1d(h_dim)) # BatchNorm stabilizes training even with normalized inputs
            self.layers.append(nn.ReLU())
            curr_dim = h_dim
        self.out = nn.Linear(curr_dim, n_output)
        self.sigmoid = nn.Sigmoid()

   
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.sigmoid(self.out(x))

def train_dnn_torch(X_train, Y_train, epochs=100, hidden_layers=[400, 200, 100], batch_size=64, lr=1e-2, device='cuda', cfg=None):
    n_input = X_train.shape[1]
    n_output = Y_train.shape[1]
    
    # 1. 必须先创建模型（带上参数），且只保留这一行
    model = DNN_T1(n_input, n_output, hidden_layers).to(device)
    
    # 2. 必须在模型创建之后定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # ... 数据加载部分保持不变 ...
    X_tensor = torch.FloatTensor(X_train).to(device)
    Y_tensor = torch.FloatTensor(Y_train).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"DNN Epoch {epoch} Loss: {total_loss / len(loader):.5f}")
            print(f"Pred Mean: {pred.mean().item():.4f}, Max: {pred.max().item():.4f}")
            
    return model

def predict_dnn_torch(model, X_test, device='cuda', binary=False):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        pred = model(X_tensor)
        pred = pred.cpu().numpy()
        
    return pred

# ================================================================================
# Unsupervised DNN Training
# ================================================================================

def train_dnn_unsup(X_train, H_train, epochs=100, hidden_layers=[400, 200, 100], 
                    batch_size=64, lr=1e-3, device='cuda', cfg=None):
    """
    无监督训练函数直接最大化系统和速率Sum-Rate
    Args:
        X_train: 归一化后的输入特征 [Batch, K*K]
        H_train: 原始信道增益矩阵 [Batch, K, K]
        cfg: 包含 n_links, tx_power, output_noise_power 的配置对象
    """
    n_input = X_train.shape[1]
    n_output = cfg.n_links 
    K = cfg.n_links
    var_noise = cfg.output_noise_power / cfg.tx_power # 归一化噪声
    
    # 1. 初始化模型与优化器
    # 建议在 IMAC 复杂场景下使用较小的学习率（如 1e-3）以稳定收敛
    model = DNN_T1(n_input, n_output, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. 准备数据加载器
    # 注意：此时不再需要 WMMSE 标签 Y_train，而是传入原始信道 H_train
    X_tensor = torch.FloatTensor(X_train).to(device)
    H_tensor = torch.FloatTensor(H_train).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, H_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for bx, bh in loader:
            optimizer.zero_grad()
            
            # 3. 前向传播得到功率预测 p (经过 Sigmoid 限制在 0-1)
            p_pred = model(bx)
            
            # 4. 计算无监督 Sum-Rate Loss
            # 这里的计算逻辑必须与测试时的 UO.compute_rates 保持物理一致
            loss = sumrate_loss_dnn(p_pred, bh, K, var_noise)
            
            # 5. 反向传播与优化
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # 打印训练状态：Loss 越小（负值越大），代表系统总速率越高
        if epoch % 10 == 0:
            print(f"DNN Epoch {epoch} | Neg-SumRate Loss: {total_loss / len(loader):.5f}")
            print(f"Pred Power Mean: {p_pred.mean().item():.4f}, Max: {p_pred.max().item():.4f}")
            
    return model

def sumrate_loss_dnn(p_pred, absH2, K, var_noise):
    """
    针对 DNN 优化的无监督和速率损失
    p_pred: [Batch, K] - 模型预测的功率
    absH2: [Batch, K, K] - 原始信道增益矩阵
    """
    # 1. 调整功率维度以匹配信道矩阵 [Batch, K, 1]
    p = torch.reshape(p_pred, (-1, K, 1))
    
    # 2. 计算接收功率 (Batch Matrix Multiplication)
    # rx_power[b, i, j] = 信号从发射机 j 到接收机 i 的强度
    rx_power = torch.mul(absH2, p.transpose(1, 2)) 
    
    # 3. 提取信号与干扰
    mask = torch.eye(K, device=p_pred.device)
    signal = torch.sum(rx_power * mask, dim=2) # [Batch, K]
    
    # 全功率减去信号功率即为干扰
    total_power = torch.sum(rx_power, dim=2) # [Batch, K]
    interference = total_power - signal + var_noise
    
    # 4. 计算速率 (Shannon Capacity)
    rate = torch.log2(1 + signal / (interference + 1e-12))
    
    # 5. 返回负均值和速率以进行最小化优化
    return -torch.mean(torch.sum(rate, dim=1))
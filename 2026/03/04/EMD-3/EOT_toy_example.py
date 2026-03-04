import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用於 3D 繪圖
from sklearn.datasets import make_swiss_roll

# 引入 POT
import ot

# --- 1. 定義 Sinkhorn Loss (保持使用 POT 的 No-Loop 版本) ---
class SinkhornLossPOT(nn.Module):
    def __init__(self, reg=0.05, max_iter=100, method='sinkhorn_log'):
        super().__init__()
        self.reg = reg
        self.max_iter = max_iter
        self.method = method

    def forward(self, x, y):
        """
        x: (N, D) - Generated points
        y: (M, D) - Target points
        D 可以是 2, 3, 甚至 100，邏輯完全不用變。
        """
        # 1. 計算代價矩陣 (Squared Euclidean Distance)
        # x, y 形狀為 (Batch, 3)
        M_dist = torch.cdist(x, y, p=2) ** 2
        
        # 2. 準備權重
        n = x.shape[0]
        m = y.shape[0]
        a = torch.ones(n, device=x.device) / n
        b = torch.ones(m, device=y.device) / m
        
        # 3. 計算 Loss
        loss = ot.sinkhorn2(a, b, M_dist, self.reg, numItermax=self.max_iter, method=self.method)
        
        return loss

# --- 2. 準備 3D 數據 (Swiss Roll) ---
def get_target_dist(batch_size=512):
    # 生成 3D 瑞士捲
    data, _ = make_swiss_roll(n_samples=batch_size, noise=0.1)
    
    # [重要] 瑞士捲原始數值很大，必須歸一化到 N(0, 1) 附近
    # 否則 Sinkhorn 計算 exp(-C/eps) 時 C 會太大導致數值問題
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return torch.tensor(data, dtype=torch.float32).cuda()

def get_source_dist(batch_size=512):
    # 生成 3D 高斯雜訊 (球狀分佈)
    return torch.randn(batch_size, 3).cuda()

# --- 3. 定義 3D 生成器 ---
class PointGenerator3D(nn.Module):
    def __init__(self):
        super().__init__()
        # 輸入 3維 -> 輸出 3維
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 輸出 x, y, z
        )

    def forward(self, x):
        return self.net(x)

# --- 4. 訓練與 3D 視覺化 ---
def train_3d_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PointGenerator3D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 初始化 Loss, reg 設為 0.05 通常對歸一化後的 3D 數據很穩定
    loss_fn = SinkhornLossPOT(reg=0.01, max_iter=50, method='sinkhorn_log')

    max_iter = 600
    plot_steps = [0, 30, 100, 600]
    
    # 設置畫布大小
    fig = plt.figure(figsize=(16, 4))
    
    for step in range(max_iter+1):
        optimizer.zero_grad()
        
        # 獲取數據
        source = get_source_dist(batch_size=512).to(device)
        target = get_target_dist(batch_size=512).to(device)
        
        # 生成
        generated = model(source)
        
        # 計算 Sinkhorn Loss
        loss = loss_fn(generated, target)
        
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # --- 3D 繪圖 ---
        if step in plot_steps:
            idx = plot_steps.index(step)
            # 使用 projection='3d'
            ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
            
            gen_np = generated.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            
            # 畫目標 (藍色半透明)
            ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], 
                       c='blue', alpha=0.1, s=5, label='Target (Swiss Roll)')
            
            # 畫生成 (紅色)
            ax.scatter(gen_np[:, 0], gen_np[:, 1], gen_np[:, 2], 
                       c='red', s=10, alpha=0.8, label='Generated')
            
            ax.set_title(f"Step {step}")
            
            # 固定視角以便觀察變化
            ax.view_init(elev=0, azim=80) 
            
            # 移除座標軸刻度讓圖比較乾淨
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            if idx == 0: 
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_3d_demo()
    
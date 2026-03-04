import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp  # 使用统一的新版 AMP 接口
from Net import GradNet, DCTPoisson  # 确保这两个类已定义
import os
import random
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import multiprocessing

# -----------------------------
# 性能优化全局设置
# -----------------------------
# 针对固定输入尺寸(256x256)开启加速
torch.backends.cudnn.benchmark = True
# 允许 TF32 (如果显卡支持 Ampere 架构，如 3090/4090，可加速矩阵运算)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# 超参数与配置
# -----------------------------
NUM_WORKERS = 4  # 因为数据已全量加载到内存，这里可以设小一点或者维持原样
SEED = 2025
INPUT_DIR = r'D:\ls\包裹相位\实验数据\general_phi\train_in'   # 修改为实际路径
LABEL_DIR = r'D:\ls\包裹相位\数据集创建\data2_k\train_gt'  # 修改为实际路径
PHI_DIR = r'D:\ls\包裹相位\数据集创建\data2_k\gt_phi'
N_SAMPLES = 8000
BATCH_SIZE = 8
VAL_RATIO = 0.25
LR = 0.9e-3

NUM_EPOCHS = 350
BEST_MODEL_PATH = 'real_model(V6).pth'#V5 loss1.2,但是对于Zer,opel预测不错
w = {'grad': 7, 'curl': 90, 'phi': 0.1, 'tv': 0}
#w = {'grad': 3, 'sin': 0, 'curl': 0.03, 'phi': 0.002, 'tv': 0.2}
#w = {'grad': 1.0, 'sin': 0, 'curl': 0.00, 'phi': 0, 'tv': 0}
# 初始化全局变量，防止报错
min_loss_total = float('inf')
min_loss_grad = float('inf')

# -----------------------------
# 损失函数 (保持不变1
# -----------------------------
def mse(a, b): return F.l1_loss(a, b)


def curl_loss(pred_gx, pred_gy):
    dygx = pred_gx[:, :, 1:, :-1] - pred_gx[:, :, :-1, :-1]
    dxgy = pred_gy[:, :, :-1, 1:] - pred_gy[:, :, :-1, :-1]
    curl = dygx - dxgy
    return torch.mean(curl ** 2)

# def tv_loss(phi, reduction='mean', order=2):
#     """
#     phi: (B,C,H,W)
#     """
#     dxx = phi - 2*torch.roll(phi, -1, dims=3) + torch.roll(phi, -2, dims=3)
#     dyy = phi - 2*torch.roll(phi, -1, dims=2) + torch.roll(phi, -2, dims=2)
#
#     dxy = ( torch.roll(phi, (-1,-1), dims=(2,3))
#           - torch.roll(phi, (-1,0),  dims=(2,3))
#           - torch.roll(phi, (0,-1),  dims=(2,3))
#           + phi )
#
#     loss = torch.abs(dxx).mean() + torch.abs(dyy).mean() + 0.5*torch.abs(dxy).mean()
#     return loss

def tv_loss(phi, reduction='mean', order=1):
    if order == 1:
        # First-order (anisotropic) TV: sum |dx| + |dy|
        dx = phi - torch.roll(phi, -1, dims=3)   # forward diff in x (width) direction
        dy = phi - torch.roll(phi, -1, dims=2)   # forward diff in y (height) direction

        # per-element TV
        tv_map = torch.abs(dx) + torch.abs(dy)  # shape (B,C,H,W)

    else:
        # Second-order (kept from your original code) — curvature-like penalty
        dxx = phi - 2*torch.roll(phi, -1, dims=3) + torch.roll(phi, -2, dims=3)
        dyy = phi - 2*torch.roll(phi, -1, dims=2) + torch.roll(phi, -2, dims=2)

        dxy = ( torch.roll(phi, (-1,-1), dims=(2,3))
              - torch.roll(phi, (-1,0),  dims=(2,3))
              - torch.roll(phi, (0,-1),  dims=(2,3))
              + phi )

        tv_map = torch.abs(dxx) + torch.abs(dyy) + 0.5*torch.abs(dxy)

    # reduction
    if reduction == 'mean':
        return tv_map.mean()
    elif reduction == 'sum':
        return tv_map.sum()
    elif reduction == 'none':
        # return per-sample aggregated TV: sum over channels+H+W, keep batch dim
        B = phi.shape[0]
        per_sample = tv_map.view(B, -1).sum(dim=1)
        return per_sample
    else:
        raise ValueError("reduction must be one of 'mean','sum','none'")

# -----------------------------
# 数据加载 (核心优化部分)
# -----------------------------
class MatDatasetStrict(Dataset):
    def __init__(self, input_dir, label_dir, phi_dir, n_samples):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.phi_dir = phi_dir
        self.n = int(n_samples)
        
        # --- 优化：预加载所有数据到 RAM ---
        self.cache = []
        print(f"正在将 {self.n} 个样本加载到内存中 (加速IO)...")
        
        for idx in tqdm(range(self.n)):
            i = idx - 51
            # 读取数据
            x_np = self._load_mat_array(os.path.join(self.input_dir, f'I_{i}.mat'))
            y_np = self._load_mat_array(os.path.join(self.label_dir, f'O_{i}.mat'))
            z_np = self._load_mat_array(os.path.join(self.phi_dir, f'P_{i}.mat'))

            # 预处理：转 Tensor, 归一化, 维度调整
            # (H,W,C) -> (C,H,W)
            x = torch.from_numpy(x_np.transpose(2, 0, 1)).float()
            y = torch.from_numpy(y_np.transpose(2, 0, 1)).float()
            # x = (x - 1.0).clamp(-1.0, 1.0)
            # x = x.clamp(-1.0, 1.0)
            
            # z 处理
            z = torch.from_numpy(z_np).float()
            if z.ndim == 2:
                z = z.unsqueeze(0)
            elif z.ndim == 3 and z.shape[2] == 1:
                 # 兼容部分可能的维度情况 (H,W,1) -> (1,H,W)
                 z = z.permute(2, 0, 1)

            # 存入缓存字典
            self.cache.append({'inp': x, 'lbl': y, 'gtp': z})

    def __len__(self):
        return self.n

    def _load_mat_array(self, path):
        try:
            # verify_compressed_data_integrity=False 可以微小加速读取
            mat = sio.loadmat(path, verify_compressed_data_integrity=False)
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            return mat[key]
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise e

    def __getitem__(self, idx):
        # --- 优化：直接从内存返回，无需磁盘 IO 和 CPU 计算 ---
        return self.cache[idx]


def create_loaders(input_dir, label_dir, phi_dir, n_samples, batch_size, val_ratio, seed=SEED):
    # 数据集初始化现在比较慢（因为要加载到内存），但训练会飞快
    dataset = MatDatasetStrict(input_dir, label_dir, phi_dir, n_samples)
    indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(indices)
    n_val = int(len(dataset) * val_ratio)

    # 优化：persistent_workers=True 保持工作进程存活，pin_memory=True 加速 CPU->GPU
    train_loader = DataLoader(
        Subset(dataset, indices[n_val:]), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        Subset(dataset, indices[:n_val]), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    return train_loader, val_loader


# -----------------------------
# 训练与验证核心
# -----------------------------
def train_one_epoch(model, poisson, loader, optimizer, scaler, scheduler, device, w):
    model.train()
    meter = {k: 0.0 for k in ['total', 'grad', 'curl', 'phi', 'tv']}

    for b in tqdm(loader, desc="Training", leave=False):
        # 优化：non_blocking=True 允许异步传输
        inp = b['inp'].to(device, non_blocking=True)
        lbl = b['lbl'].to(device, non_blocking=True)
        gtp = b['gtp'].to(device, non_blocking=True)
        
        #I_x, I_y = inp[:, 0:1], inp[:, 1:2]
        gt_gx, gt_gy = lbl[:, 0:1], lbl[:, 1:2]

        optimizer.zero_grad(set_to_none=True)
        
        # 使用统一的新版 autocast API
        with torch.amp.autocast('cuda'):
            gx, gy = model(inp)
            phi = poisson(gx, gy)
            
            # 计算各项损失
            l_grad = mse(gx, gt_gx) + mse(gy, gt_gy)
            l_curl = curl_loss(gx, gy)
            l_phi = mse(phi, gtp - gtp.mean((2, 3), keepdim=True))
            l_tv = tv_loss(phi,'mean',order=1)
            
            losses = {
                'grad': l_grad,
                'curl': l_curl,
                'phi': l_phi,
                'tv': l_tv
            }
            total = sum(w[k] * losses[k] for k in losses)

        scaler.scale(total).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            scheduler.step()

        # 优化：减少 .item() 调用产生的同步开销，虽然此处影响不大但习惯要好
        with torch.no_grad():
            for k in meter:
                meter[k] += (losses[k].detach() if k != 'total' else total.detach())

    # 最后再转 CPU 计算平均值
    return {k: v.item() / len(loader) for k, v in meter.items()}


@torch.no_grad()
def validate_one_epoch(model, poisson, loader, device, w):
    model.eval()
    meter = {k: 0.0 for k in ['total', 'grad', 'curl', 'phi', 'tv']}

    for b in loader:
        inp = b['inp'].to(device, non_blocking=True)
        lbl = b['lbl'].to(device, non_blocking=True)
        gtp = b['gtp'].to(device, non_blocking=True)
        
        I_x, I_y = inp[:, 0:1], inp[:, 1:2]
        gt_gx, gt_gy = lbl[:, 0:1], lbl[:, 1:2]

        with torch.amp.autocast('cuda'):
            gx, gy = model(inp)
            phi = poisson(gx, gy)
            
            l_grad = mse(gx, gt_gx) + mse(gy, gt_gy)
            l_curl = curl_loss(gx, gy)
            l_phi = mse(phi, gtp - gtp.mean((2, 3), keepdim=True))
            l_tv = tv_loss(phi)

            losses = {
                'grad': l_grad,
                'curl': l_curl,
                'phi': l_phi,
                'tv': l_tv
            }
            total = sum(w[k] * losses[k] for k in losses)

        for k in meter:
            meter[k] += (losses[k].detach() if k != 'total' else total.detach())

    return {k: v.item() / len(loader) for k, v in meter.items()}

# -----------------------------
# 主训练循环
# -----------------------------
def train():
    global min_loss_total
    global min_loss_grad
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting training on {device}...")

    # 数据加载 (这一步会花一点时间加载到内存)
    train_loader, val_loader = create_loaders(INPUT_DIR, LABEL_DIR, PHI_DIR, N_SAMPLES, BATCH_SIZE, VAL_RATIO)

    model = GradNet().to(device)
    poisson = DCTPoisson(precompute_shape=(256, 256)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10, epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.15
    )

    min_val_loss_total = float('inf')
    # min_val_loss_grad = float('inf') # 如需启用梯度保存逻辑可取消注释

    log_list = []
    log_save_path = "training_log.csv"

    for ep in range(NUM_EPOCHS):
        w['phi'] = 0.05 if ep >= 120 else 0.0

        train_metrics = train_one_epoch(model, poisson, train_loader, optimizer, scaler, scheduler, device, w)
        val_metrics = validate_one_epoch(model, poisson, val_loader, device, w)

        val_total = val_metrics['total']
        val_grad = val_metrics['grad']
        curr_lr = optimizer.param_groups[0]['lr']

        epoch_log = {
            'epoch': ep,
            'lr': curr_lr,
            'train_total': train_metrics['total'],
            'train_grad': train_metrics['grad'],
            'train_curl': train_metrics['curl'],
            'train_tv': train_metrics['tv'],
            'train_phi': train_metrics['phi'],
            'val_total': val_metrics['total'],
            'val_grad': val_metrics['grad'],
            'val_curl': val_metrics['curl'],
            'val_tv': val_metrics['tv'],
            'val_phi': val_metrics['phi']
        }
        log_list.append(epoch_log)

        if ep % 5 == 0 or ep == NUM_EPOCHS - 1:
            pd.DataFrame(log_list).to_csv(log_save_path, index=False)

        save_str = ""
        # 逻辑修改：如果当前验证总损失更小，则保存
        # 注意：原代码逻辑是用 val_grad < min_val_loss_total 来更新，我保持了你的逻辑
        # 但通常建议用 val_total 比较，或者 val_grad 比较 min_val_loss_grad
        if val_grad < min_val_loss_total:
            min_val_loss_total = val_grad
            min_loss_total = min_val_loss_total # 更新全局变量
            
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_loss': val_grad,
            }, BEST_MODEL_PATH)
            save_str = f" --> Model Saved (Best Val Grad: {val_grad:.3e})"

        print(
            f"[{ep:03d}] T_Total: {train_metrics['total']:.3e} | T_grad: {train_metrics['grad']:.3e}| "
            f"V_Total: {val_total:.3e} | V_Grad: {val_metrics['grad']:.3e} | LR: {curr_lr:.1e}{save_str}"
        )

    return log_list

def plot_curves(h):
    if not h: return
    df = pd.DataFrame(h)
    fig, ax1 = plt.subplots()
    ax1.plot(df['epoch'], df['train_grad'], label='Train Grad')
    ax1.plot(df['epoch'], df['val_grad'], label='Val Grad')
    ax1.set_yscale('log')
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    # Windows下使用多进程DataLoader需要的保护
    # 确保目录存在
    for d in [INPUT_DIR, LABEL_DIR, PHI_DIR]:
        if not os.path.exists(d):
            print(f"Warning: Directory not found: {d}")
            # os.makedirs(d, exist_ok=True) # 建议手动确认路径，不自动创建以免路径错误

    hist = train()
    print(f"最小验证损失: {min_loss_total:.4e}")
    plot_curves(hist)
    plt.savefig('loss_curve.png')
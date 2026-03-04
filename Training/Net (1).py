"""
grad2phi_medium.py
Single-file medium U-Net style model: input (2-channel) -> output (2-channel grads) -> FFT-Poisson -> phi
Losses: grad, sin-consistency, curl, phi (after Poisson), TV
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import time
import copy
from thop import profile
import math
# ---------------------------
# Basic building blocks
# ---------------------------
class DWConv(nn.Module):
    """Depthwise conv + pointwise conv"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, expansion=1):
        super().__init__()
        mid = int(in_ch * expansion)
        self.use_pw1 = (expansion != 1)
        layers = []
        if self.use_pw1:
            layers.append(nn.Conv2d(in_ch, mid, 1, bias=False))
            layers.append(nn.GroupNorm(8, mid))
            layers.append(nn.SiLU())
        layers.extend([
            nn.Conv2d(mid if self.use_pw1 else in_ch, mid if self.use_pw1 else in_ch,
                      kernel, stride, padding, groups=mid if self.use_pw1 else (in_ch), bias=False),
            nn.GroupNorm(8, mid if self.use_pw1 else in_ch),
            nn.SiLU(),
            nn.Conv2d(mid if self.use_pw1 else in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class InvResBlock(nn.Module):
    """Inverted residual: expansion -> DW conv -> project"""
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid = int(in_ch * expansion)
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch)
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y

# ---------------------------
# ASPP (bottleneck)
# ---------------------------
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1,2,4)):
        super().__init__()
        branches = []
        for d in dilations:
            if d == 1:
                b = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU()
                )
            else:
                b = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU()
                )
            branches.append(b)
        self.branches = nn.ModuleList(branches)
        self.project = nn.Sequential(
            nn.Conv2d(len(branches)*out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        return self.project(torch.cat(outs, dim=1))

# ---------------------------
# U-Net style network (Medium)
# ---------------------------
# # 没有dropout
# class GradNet(nn.Module):
#     def __init__(self, in_ch=2):
#         super().__init__()
#         # 初始层：稍微加宽，捕捉更多初级纹理
#         self.init = nn.Sequential(
#             nn.Conv2d(in_ch, 48, 3, padding=1, bias=False),
#             nn.GroupNorm(8, 48),
#             nn.SiLU()
#         )
#
#         # Encoder: 更加渐进的通道增长
#         # H -> H/2 (48 -> 64)
#         self.e1 = InvResBlock(48, 64, stride=2, expansion=3)
#         # H/2 -> H/4 (64 -> 96)
#         self.e2 = nn.Sequential(
#             InvResBlock(64, 96, stride=2, expansion=3),
#             InvResBlock(96, 96, stride=1, expansion=3)
#         )
#         # H/4 -> H/8 (96 -> 160)
#         self.e3 = nn.Sequential(
#             InvResBlock(96, 160, stride=2, expansion=3),
#             InvResBlock(160, 160, stride=1, expansion=3),
#             InvResBlock(160, 160, stride=1, expansion=3)  # 保持深度以获得足够感受野
#         )
#         # H/8 -> H/16 (160 -> 256)
#         self.e4 = nn.Sequential(
#             InvResBlock(160, 256, stride=2, expansion=3),
#             InvResBlock(256, 256, stride=1, expansion=3)
#         )
#
#         # ASPP: 保持 256 不变，这是捕捉全局梯度的核心
#         self.aspp = ASPP(256, 256, dilations=(1, 6, 12))  # 增加空洞率，扩大感受野
#
#         # Decoder: 显著增强上采样路径的通道
#         # 目的是平滑地将深层语义(全局相位)引导至浅层(局部细节)
#         self.up1 = self._make_up_block(256 + 160, 160)  # H/8
#         self.up2 = self._make_up_block(160 + 96, 96)  # H/4
#         self.up3 = self._make_up_block(96 + 64, 64)  # H/2
#         self.up4 = self._make_up_block(64 + 48, 32)  # H
#
#         self.head = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1, bias=False),
#             nn.GroupNorm(8, 32),
#             nn.SiLU(),
#             nn.Conv2d(32, 2, 1)
#         )
#
#     def _make_up_block(self, in_ch, out_ch):
#         return nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 1, bias=False),
#             nn.GroupNorm(8, out_ch),
#             nn.SiLU(),
#             DWConv(out_ch, out_ch, kernel=3, expansion=2)  # 增加内部扩展比
#         )
#
#     def forward(self, x):
#         # ... forward 逻辑与之前一致，只需对应新的模块名 ...
#         x0 = self.init(x)
#         f1 = self.e1(x0)
#         f2 = self.e2(f1)
#         f3 = self.e3(f2)
#         f4 = self.e4(f3)
#         b = self.aspp(f4)
#
#         u1 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
#         u1 = self.up1(torch.cat([u1, f3], dim=1))
#
#         u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
#         u2 = self.up2(torch.cat([u2, f2], dim=1))
#
#         u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
#         u3 = self.up3(torch.cat([u3, f1], dim=1))
#
#         u4 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
#         u4 = self.up4(torch.cat([u4, x0], dim=1))
#
#         out = self.head(u4)
#         return out[:, 0:1, :, :], out[:, 1:2, :, :]


# 注意：该文件只包含修改后的 GradNet 类。请确保你的工程中已定义 InvResBlock, ASPP, DWConv。

class GradNet(nn.Module):
    def __init__(self, in_ch=2, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p

        # 初始层：稍微加宽，捕捉更多初级纹理
        self.init = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )

        # Encoder: 更加渐进的通道增长
        # H -> H/2 (48 -> 64)
        self.e1 = InvResBlock(48, 64, stride=2, expansion=3)
        # H/2 -> H/4 (64 -> 96)
        self.e2 = nn.Sequential(
            InvResBlock(64, 96, stride=2, expansion=3),
            InvResBlock(96, 96, stride=1, expansion=3)
        )
        # H/4 -> H/8 (96 -> 160)
        self.e3 = nn.Sequential(
            InvResBlock(96, 160, stride=2, expansion=3),
            InvResBlock(160, 160, stride=1, expansion=3),
            InvResBlock(160, 160, stride=1, expansion=3)  # 保持深度以获得足够感受野
        )
        # H/8 -> H/16 (160 -> 256)
        self.e4 = nn.Sequential(
            InvResBlock(160, 256, stride=2, expansion=3),
            InvResBlock(256, 256, stride=1, expansion=3)
        )

        # ASPP: 保持 256 不变，这是捕捉全局梯度的核心
        self.aspp = ASPP(256, 320, dilations=(1, 6, 12))  # 增加空洞率，扩大感受野
        # 在 ASPP 输出后加 dropout，减少过拟合
        #self.post_aspp_dropout = nn.Dropout2d(self.dropout_p)

        # Decoder: 显著增强上采样路径的通道
        # 目的是平滑地将深层语义(全局相位)引导至浅层(局部细节)
        self.up1 = self._make_up_block(320 + 160, 160)  # H/8
        self.up2 = self._make_up_block(160 + 96, 96)  # H/4
        self.up3 = self._make_up_block(96 + 64, 64)  # H/2
        self.up4 = self._make_up_block(64 + 48, 32)  # H

        self.head = nn.Sequential(
            nn.Conv2d(32, 40, 3, padding=1, bias=False),
            nn.GroupNorm(8, 40),
            nn.SiLU(),
            #nn.Dropout2d(self.dropout_p/2),
            nn.Conv2d(40, 2, 1)
        )

    def _make_up_block(self, in_ch, out_ch):
        # 使用 self.dropout_p，方便在 __init__ 时设定统一的 dropout 概率
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            DWConv(out_ch, out_ch, kernel=3, expansion=2),  # 增加内部扩展比
        )

    def forward(self, x):
        x0 = self.init(x)
        f1 = self.e1(x0)
        f2 = self.e2(f1)
        f3 = self.e3(f2)
        f4 = self.e4(f3)
        b = self.aspp(f4)
        #b = self.post_aspp_dropout(b)

        u1 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, f3], dim=1))

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, f2], dim=1))

        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u3, f1], dim=1))

        u4 = F.interpolate(u3, scale_factor=2, mode='bilinear', align_corners=False)
        u4 = self.up4(torch.cat([u4, x0], dim=1))

        out = self.head(u4)
        return out[:, 0:1, :, :], out[:, 1:2, :, :]



# ---------------------------
# Poisson layer: FFT-based (periodic BC)
# ---------------------------
# class FFTPoisson(nn.Module):
#     """Solve laplacian(phi) = div(g) with FFT. Returns phi with mean zero."""
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps
#
#     def forward(self, gx, gy):
#         # gx, gy shape: (B,1,H,W)
#         # compute divergence (finite differences)
#         # pad periodicity not needed for spectral derivative if we treat as periodic
#         # convert to complex via torch.fft
#         div = self._divergence(gx, gy)  # (B,1,H,W)
#         phi = self._solve_poisson(div)
#         return phi
#
#     def _divergence(self, gx, gy):
#         # simple spectral-friendly finite difference (periodic)
#         # forward difference with wrap
#         div_x = gx - torch.roll(gx, shifts=1, dims=3)  # ∂x gx approx
#         div_y = gy - torch.roll(gy, shifts=1, dims=2)  # ∂y gy approx
#         div = div_x + div_y
#         return div
#
#     def _solve_poisson(self, f):
#         # f: (B,1,H,W)
#         B, C, H, W = f.shape
#         # FFT
#         f_c = torch.fft.rfftn(f.float(), dim=(2,3))  # shape (B,1,H,W//2+1), complex
#         # build freq grid (device-aware)
#         device = f.device
#         ky = torch.fft.fftfreq(H, d=1.0, device=device).reshape(H,1) * (2*torch.pi)
#         kx = torch.fft.rfftfreq(W, d=1.0, device=device).reshape(1, W//2+1) * (2*torch.pi)
#         ksq = (ky**2) + (kx**2)  # (H, W//2+1)
#         ksq = ksq.unsqueeze(0).unsqueeze(0)  # (1,1,H,Wc)
#         denom = -ksq
#         denom[...,0,0] = -1.0  # avoid division by zero for k=(0,0); will set mean=0
#         phi_c = f_c / (denom + self.eps)
#         # set DC mode to zero (enforce zero-mean)
#         phi_c[...,0,0] = 0.0
#         # inverse fft
#         phi = torch.fft.irfftn(phi_c, s=(H,W), dim=(2,3))
#         return phi
#
# class DCTPoisson(nn.Module):
#     """
#     完全对齐 MATLAB 逻辑的高精度 DCT 泊松求解器
#     精度目标：float64 下达到 1e-13 以上误差
#     """
#     def __init__(self, precompute_shape: tuple, device=None):
#         super().__init__()
#         H, W = precompute_shape
#         # 核心：必须使用 float64 才能匹配 MATLAB 的双精度
#         self.dtype = torch.float16
#         self.H, self.W = H, W
#
#         # 预计算高精度变换矩阵
#         self.register_buffer('_Cx', self._dct_ii_matrix(H, dtype=self.dtype))
#         self.register_buffer('_Cy', self._dct_ii_matrix(W, dtype=self.dtype))
#
#         # 严格对齐 MATLAB 的特征值构造
#         kx = torch.arange(0, W, dtype=self.dtype).reshape(1, W)
#         ky = torch.arange(0, H, dtype=self.dtype).reshape(H, 1)
#         # 这里的公式必须与离散拉普拉斯算子严格对应
#         lam = 2.0 * (torch.cos(math.pi * kx / W) + torch.cos(math.pi * ky / H) - 2.0)
#         lam[0, 0] = 1.0  # 避免除以 0
#         self.register_buffer('_lambda', lam)
#
#     def forward(self, gx, gy):
#         # 1. 强制转换为 float64 进行计算
#         B, _, H, W = gx.shape
#         gx = gx.to(self.dtype).squeeze(1)
#         gy = gy.to(self.dtype).squeeze(1)
#
#         # 2. 严格对齐 MATLAB 的 Adjoint Divergence (伴随散度)
#         # 注意：这里必须要克隆，避免原地操作修改输入
#         div = torch.zeros((B, H, W), dtype=self.dtype, device=gx.device)
#
#         # X-part
#         div[:, :, 0] = gx[:, :, 0]
#         div[:, :, 1:W - 1] = gx[:, :, 1:W - 1] - gx[:, :, 0:W - 2]
#         div[:, :, W - 1] = -gx[:, :, W - 2]
#
#         # Y-part
#         div[:, 0, :] += gy[:, 0, :]
#         div[:, 1:H - 1, :] += gy[:, 1:H - 1, :] - gy[:, 0:H - 2, :]
#         div[:, H - 1, :] -= gy[:, H - 2, :]
#
#         # 3. 高精度 DCT2: C @ rhs @ C^T
#
#         rhs_dct = torch.matmul(self._Cx, torch.matmul(div, self._Cy.t()))
#
#         # 4. 频域除法
#         phi_dct = rhs_dct / self._lambda
#         phi_dct[:, 0, 0] = 0.0  # 强制 DC 分量为 0 (mean zero)
#
#         # 5. 高精度 IDCT2: Cx^T @ phi_dct @ Cy
#         phi = torch.matmul(self._Cx.t(), torch.matmul(phi_dct, self._Cy))
#
#         # 6. 均值归一化 (对齐 MATLAB 的常数项消除)
#         # phi = phi - phi.view(B, -1).mean(dim=1).view(B, 1, 1)
#
#         return phi.unsqueeze(1)  # 回到 (B, 1, H, W)
#
#     @staticmethod
#     def _dct_ii_matrix(N, dtype):
#         n = torch.arange(N, dtype=dtype).reshape(1, N)
#         k = torch.arange(N, dtype=dtype).reshape(N, 1)
#         ang = math.pi * k * (2.0 * n + 1.0) / (2.0 * N)
#         C = torch.cos(ang)
#         # 正交化缩放因子
#         alpha = torch.full((N, 1), math.sqrt(2.0 / N), dtype=dtype)
#         alpha[0, 0] = math.sqrt(1.0 / N)
#         result = alpha * C
#         return result


class DCTPoisson(nn.Module):
    """
    Solve laplacian(phi) = div(gx, gy) with DCT (Neumann BC).
    Matches the input/output signature of FFTPoisson.
    """

    def __init__(self, precompute_shape: tuple, eps=1e-6, device=None):
        super().__init__()
        H, W = precompute_shape
        self.eps = eps
        # 内部计算建议使用 float64 以保持 MATLAB 级别的精度
        self.internal_dtype = torch.float64

        # 预计算正交变换矩阵
        self.register_buffer('_Cx', self._dct_ii_matrix(H, dtype=self.internal_dtype))
        self.register_buffer('_Cy', self._dct_ii_matrix(W, dtype=self.internal_dtype))

        # 预计算特征值 (Neumann Eigenvalues)
        kx = torch.arange(0, W, dtype=self.internal_dtype).reshape(1, W)
        ky = torch.arange(0, H, dtype=self.internal_dtype).reshape(H, 1)
        lam = 2.0 * (torch.cos(math.pi * kx / W) + torch.cos(math.pi * ky / H) - 2.0)
        lam[0, 0] = 1.0  # 避免除以 0
        self.register_buffer('_lambda', lam)

    def forward(self, gx, gy):
        """
        Input: gx, gy tensors shape (B, 1, H, W)
        Output: phi tensor shape (B, 1, H, W)
        """
        # 保存原始精度，以便最后转换回去
        orig_dtype = gx.dtype
        B, C, H, W = gx.shape

        # 1. 准备数据: (B, 1, H, W) -> (B, H, W) 并提升精度
        gx_in = gx.squeeze(1).to(self.internal_dtype)
        gy_in = gy.squeeze(1).to(self.internal_dtype)

        # 2. 计算 Adjoint Divergence (Neumann)
        div = torch.zeros((B, H, W), dtype=self.internal_dtype, device=gx.device)
        # X-part
        div[:, :, 0] = gx_in[:, :, 0]
        div[:, :, 1:W - 1] = gx_in[:, :, 1:W - 1] - gx_in[:, :, 0:W - 2]
        div[:, :, W - 1] = -gx_in[:, :, W - 2]
        # Y-part
        div[:, 0, :] += gy_in[:, 0, :]
        div[:, 1:H - 1, :] += gy_in[:, 1:H - 1, :] - gy_in[:, 0:H - 2, :]
        div[:, H - 1, :] -= gy_in[:, H - 2, :]

        # 3. 2D DCT: phi_dct = Cx @ div @ Cy^T
        # 使用 matmul 自动处理 Batch 维度
        rhs_dct = torch.matmul(self._Cx, torch.matmul(div, self._Cy.t()))

        # 4. 频域除法 (泊松求解)
        phi_dct = rhs_dct / (self._lambda)
        phi_dct[:, 0, 0] = 0.0  # 强制均值为 0

        # 5. 2D IDCT: phi = Cx^T @ phi_dct @ Cy
        phi = torch.matmul(self._Cx.t(), torch.matmul(phi_dct, self._Cy))
        phi = phi - phi.mean()

        # 6. 恢复形状与精度: (B, H, W) -> (B, 1, H, W)
        return phi.unsqueeze(1).to(orig_dtype)

    @staticmethod
    def _dct_ii_matrix(N, dtype):
        n = torch.arange(N, dtype=dtype).reshape(1, N)
        k = torch.arange(N, dtype=dtype).reshape(N, 1)
        ang = math.pi * k * (2.0 * n + 1.0) / (2.0 * N)
        C = torch.cos(ang)
        alpha = torch.full((N, 1), math.sqrt(2.0 / N), dtype=dtype)
        alpha[0, 0] = math.sqrt(1.0 / N)
        return alpha * C

def benchmark_model(h=256, w=256, batch_size=1, device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    # 1. 初始化模型
    model = GradNet(in_ch=2).to(device)
    poisson = DCTPoisson(precompute_shape=(h,w)).to(device)
    model.eval()

    # 2. 准备伪数据
    # 输入通常是 (B, 2, H, W) 代表 I_x, I_y
    dummy_input = torch.randn(batch_size, 2, h, w).to(device)

    print(f"--- Performance Report ({h}x{w}, Batch={batch_size}) ---")

    # 3. 计算参数量和 FLOPs (仅针对 CNN 部分)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    print(f"Total Parameters: {params / 1e6:.2f} M")
    print(f"Total FLOPs:      {flops / 1e9:.2f} G (CNN only)")

    # # 4. 测量推理时间 (End-to-End)
    # # 预热 GPU
    with torch.no_grad():
        for _ in range(1):
            gx, gy = model(dummy_input)
            _ = poisson(gx, gy)
    #
    # torch.cuda.synchronize()
    start_time = time.time()

    iters = 1
    with torch.no_grad():
        for _ in range(iters):
            gx, gy = model(dummy_input)
            phi = poisson(gx, gy)


    # torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / iters * 1000  # ms
    fps = 1000 / avg_time

    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f"Throughput (FPS):   {fps:.2f}")

    # 5. 显存占用测试
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        # 模拟一次推理过程
        gx, gy = model(dummy_input)
        phi = poisson(gx, gy)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak Memory usage:  {peak_mem:.2f} MB")

    print("-" * 45)


if __name__ == "__main__":
    # 测试不同分辨率下的表现
    for res in [128, 512]:
        benchmark_model(h=res, w=res, batch_size=1)
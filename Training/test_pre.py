import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Net import GradNet,DCTPoisson
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import json
import pandas as pd
import torch.nn.functional as F

import numpy as np
from typing import List, Union
import pickle
import os
# ===============================


class MatDatasetStrict(Dataset):
    def __init__(self, input_dir, label_dir, phi_dir,n_samples):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.phi_dir = phi_dir
        self.n = int(n_samples)

    def __len__(self):
        return self.n

    def _load_mat_array(self, path):
        mat = sio.loadmat(path)
        key = [k for k in mat.keys() if not k.startswith('__')][0]
        return mat[key]

    def __getitem__(self, idx):
        i = idx + 8000
        x = self._load_mat_array(os.path.join(self.input_dir, f'I_{i}.mat'))
        y = self._load_mat_array(os.path.join(self.label_dir, f'O_{i}.mat'))
        z = self._load_mat_array(os.path.join(self.phi_dir, f'P_{i}.mat'))


        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy(y.transpose(2, 0, 1)).float()
        z = torch.from_numpy(z).unsqueeze(0).float()

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        z = z.unsqueeze(0)


        x = x.squeeze(0)
        y = y.squeeze(0)
        z = z.squeeze(0)

        return {'inp': x, 'lbl': y, 'gtp': z}


## 计算 RMSE
def test_model(test_input_dir,
               test_label_dir,
               phi_dir,
               n_test,
               model_path,
               device='cuda',
               batch_size=1,
               visualize_n=1,
               save_results=True,
               result_dir='./test_results'):

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建结果目录
    if save_results and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # dataset & loader
    test_set = MatDatasetStrict(test_input_dir, test_label_dir, phi_dir, n_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model
    model = GradNet().to(device)
    poisson = DCTPoisson(precompute_shape=(256, 256)).to(device)

    # 加载模型权重
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    # 处理checkpoint格式
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise RuntimeError('Unrecognized checkpoint format.')

    model.load_state_dict(state_dict)
    model.eval()

    print(f'✓ 模型已加载: {model_path}')
    print(f'✓ 测试样本数: {n_test}')
    print(f'✓ 批量大小: {batch_size}')
    print('-' * 50)

    # 初始化统计变量
    total_rmse_grad_x = 0.0
    total_rmse_grad_y = 0.0
    total_rmse_phi_abs = 0.0  # 绝对相位误差
    # 相对误差 = 绝对误差 / 参考值
    total_rmse_phi_rel = 0.0
    total_samples = 0

    # 保存每个样本的误差用于分析
    all_errors = {
        'grad_x': [],
        'grad_y': [],
        'phi_abs': [],
        'phi_rel':[],
        'SSIM':[]
    }
    ssim_value = []

    with torch.no_grad():
        # 使用tqdm显示进度
        pbar = tqdm(test_loader, desc='测试中', ncols=100)

        for idx, batch in enumerate(pbar):
            inp = batch['inp'].to(device)
            gt = batch['lbl'].to(device)
            gt_phi = batch['gtp'].to(device)

            # 前向传播
            pred_x, pred_y = model(inp)

            # 边界条件
            pred_x[:, :, :, 255] = 0.0
            pred_y[:, :, 255, :] = 0.0

            # Poisson重建
            re_phi = poisson(pred_x, pred_y)

            # 计算当前batch的误差
            batch_size_current = inp.size(0)
            total_samples += batch_size_current

            # 1. 梯度误差
            gt_grad_x = gt[:, 0:1, :, :]  # 假设gt的形状是[N, 2, H, W]
            gt_grad_y = gt[:, 1:2, :, :]

            err_x = pred_x - gt_grad_x
            err_y = pred_y - gt_grad_y

            batch_rmse_x = torch.sqrt(torch.mean(err_x ** 2)).item()
            batch_rmse_y = torch.sqrt(torch.mean(err_y ** 2)).item()

            total_rmse_grad_x += batch_rmse_x * batch_size_current
            total_rmse_grad_y += batch_rmse_y * batch_size_current

            # 2. 相位误差
            # 方法1: 绝对误差（保持原始偏移）
            gt_phi = gt_phi - gt_phi.mean()
            phi_err_abs = re_phi - gt_phi
            # 相对误差 = 绝对误差 / 参考值
            data_range = gt_phi.max() - gt_phi.min()
            rel_error = phi_err_abs / data_range

            batch_rmse_phi_abs = torch.sqrt(torch.mean(phi_err_abs ** 2)).item()
            total_rmse_phi_abs += batch_rmse_phi_abs * batch_size_current

            batch_rmse_phi_rel = torch.sqrt(torch.mean(rel_error ** 2)).item()
            total_rmse_phi_rel += batch_rmse_phi_rel * batch_size_current

            ##ssim计算
            gt_phi_np = gt_phi[0, 0].cpu().numpy()
            re_phi_np = re_phi[0, 0].cpu().numpy()
            ssim,ssim_map = calculate_ssim_skimage(gt_phi_np, re_phi_np, data_range=None)
            ssim_value.append(ssim)

            # 保存误差统计
            all_errors['grad_x'].append(batch_rmse_x)
            all_errors['grad_y'].append(batch_rmse_y)
            all_errors['phi_abs'].append(batch_rmse_phi_abs)
            all_errors['phi_rel'].append(batch_rmse_phi_rel)
            all_errors['SSIM'].append(ssim)

            # 更新进度条
            pbar.set_postfix({
                'RMSE_x': f'{batch_rmse_x:.4f}',
                'RMSE_y': f'{batch_rmse_y:.4f}',
                'RMSE_phi_abs': f'{batch_rmse_phi_abs:.4f}',
                'RMSE_phi_rel': f'{batch_rmse_phi_rel:.4f}',
            })

            # 可视化前几个样本
            # if idx < visualize_n:
            #     visualize_sample(inp, gt, gt_phi, pred_x, pred_y, re_phi,
            #                      idx, save_results, result_dir)
        plt.imshow(ssim_map)
        plt.xlabel('Sample Index')
        plt.ylabel('SSIM')
        plt.title('SSIM Values per Sample')
        plt.show()
        np.save('ssim_values.npy', np.array(ssim_value))
        np.save('error.npy', np.array(all_errors))

    # 计算平均RMSE
    avg_rmse_grad_x = total_rmse_grad_x / total_samples
    avg_rmse_grad_y = total_rmse_grad_y / total_samples
    avg_rmse_phi_abs = total_rmse_phi_abs / total_samples
    avg_rmse_phi_rel = total_rmse_phi_rel / total_samples

    all_errors['grad_x'] = np.array(all_errors['grad_x'])
    all_errors['grad_y'] = np.array(all_errors['grad_y'])
    all_errors['phi_abs'] = np.array(all_errors['phi_abs'])
    all_errors['phi_rel'] = np.array(all_errors['phi_rel'])

    # 打印结果
    print('\n' + '=' * 60)
    print('测试结果汇总')
    print('=' * 60)
    print(f'总测试样本数: {total_samples}')
    print(f'平均梯度X RMSE: {avg_rmse_grad_x:.6f}')
    print(f'平均梯度Y RMSE: {avg_rmse_grad_y:.6f}')
    print(f'平均梯度总 RMSE: {np.sqrt(0.5 * (avg_rmse_grad_x ** 2 + avg_rmse_grad_y ** 2)):.6f}')
    print(f'平均相位绝对RMSE: {avg_rmse_phi_abs:.6f}')

    print('=' * 60)

    # 保存结果
    if save_results:
        save_test_results(all_errors, result_dir, avg_rmse_grad_x, avg_rmse_grad_y,
                          avg_rmse_phi_abs,ssim)

    return {
        'avg_rmse_grad_x': avg_rmse_grad_x,
        'avg_rmse_grad_y': avg_rmse_grad_y,
        'avg_rmse_phi_abs': avg_rmse_phi_abs,
        'avg_rmse_phi_rel': avg_rmse_phi_rel,
        'all_errors': all_errors,
        'ssim_values': ssim_value
    }


def visualize_sample(inp, gt, gt_phi, pred_x, pred_y, re_phi, idx,
                     save_results=True, result_dir='./test_results'):


    # 转换为numpy
    inp_np = inp[0, 0].cpu().numpy()  # 假设是单通道
    gt_grad_x_np = gt[0, 0].cpu().numpy()
    gt_grad_y_np = gt[0, 1].cpu().numpy()
    gt_phi_np = gt_phi[0, 0].cpu().numpy()
    pred_x_np = pred_x[0, 0].cpu().numpy()
    pred_y_np = pred_y[0, 0].cpu().numpy()
    re_phi_np = re_phi[0, 0].cpu().numpy()

    # 计算误差
    gt_phi_np = gt_phi_np - gt_phi_np.mean()
    err_phi = re_phi_np - gt_phi_np

    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'sample {idx + 1} vis', fontsize=16)

    # 第一行：输入和梯度
    im0 = axes[0, 0].imshow(inp_np, cmap='viridis')
    axes[0, 0].set_title('Input')
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(gt_grad_x_np, cmap='coolwarm')
    axes[0, 1].set_title('gx')
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(pred_x_np, cmap='coolwarm')
    axes[0, 2].set_title('pre_gx')
    plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[0, 3].imshow(pred_x_np - gt_grad_x_np, cmap='RdBu')
    axes[0, 3].set_title('err gx')
    plt.colorbar(im3, ax=axes[0, 3])

    # 第二行：相位
    im4 = axes[1, 0].imshow(gt_phi_np, cmap='viridis')
    axes[1, 0].set_title('gt_phi')
    plt.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(re_phi_np, cmap='viridis')
    axes[1, 1].set_title('re_phi')
    plt.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(err_phi, cmap='RdBu')
    axes[1, 2].set_title('err_phi')
    plt.colorbar(im6, ax=axes[1, 2])


    plt.tight_layout()

    if save_results:
        plt.savefig(f'{result_dir}/sample_{idx + 1:04d}.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        plt.show()


def save_test_results(all_errors, result_dir, rmse_x, rmse_y, rmse_phi_abs,ssim):

    # 保存统计数据
    stats = {
        'avg_rmse_grad_x': float(rmse_x),
        'avg_rmse_grad_y': float(rmse_y),
        'avg_rmse_phi_abs': float(rmse_phi_abs),
        'num_samples': len(all_errors['phi_abs'])
    }

    with open(f'{result_dir}/test_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

    # 保存详细误差
    df = pd.DataFrame({
        'grad_x_rmse': all_errors['grad_x'],
        'grad_y_rmse': all_errors['grad_y'],
        'phi_abs_rmse': all_errors['phi_abs'],

    })
    df.to_csv(f'{result_dir}/detailed_errors.csv', index=False)


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_errors['grad_x'], bins=100, alpha=0.7, color='blue')
    axes[0].axvline(rmse_x, color='red', linestyle='--', label=f'Average: {rmse_x:.4f}')
    axes[0].set_title(r'$\partial_x\varphi \, RMSE$')
    axes[0].set_xlabel('RMSE')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(False)

    axes[1].hist(all_errors['grad_y'], bins=4, alpha=0.7, color='green')
    axes[1].axvline(rmse_y, color='red', linestyle='--', label=f'Average: {rmse_y:.4f}')
    axes[1].set_title(r'$\partial_y\varphi \, RMSE$')
    axes[1].set_xlabel('RMSE')
    axes[1].legend()
    axes[1].grid(False)

    axes[2].hist(all_errors['phi_abs'], bins=4, alpha=0.7, color='orange')
    axes[2].axvline(rmse_phi_abs, color='red', linestyle='--', label=f'Average: {rmse_phi_abs:.4f}')
    axes[2].set_title(r'$\varphi_{rec},\varphi \, RMSE$')
    axes[2].set_xlabel('RMSE')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(False)

    plt.suptitle('Error Distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{result_dir}/error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()


    print(f'✓ 结果已保存到: {result_dir}')



## 计算 SSIM
def calculate_ssim_skimage(img1, img2, win_size=None,data_range = None):


    img1 = np.array(img1)
    img2 = np.array(img2)

    if data_range is None:
        if img1.dtype == np.uint8:
            data_range = 255.0
        elif img1.max() <= 1.0 and img1.min() >= 0.0:
            data_range = 1.0
        else:
            data_range = img1.max() - img1.min()

    if len(img1.shape) == 2:  # 灰度图像
        ssim_value, ssim_map = ssim(img1, img2,
                                    data_range=data_range,
                                    win_size=win_size,
                                    full=True)
    else:  # 彩色图像
        print('你好，嘿嘿')

    return ssim_value, ssim_map


if __name__ == "__main__":
    results = test_model(
        test_input_dir = r'D:\ls\包裹相位\实验数据\general_phi\train_in',
        test_label_dir = r'D:\ls\包裹相位\数据集创建\data9_k\train_gt',
        phi_dir = r'D:\ls\包裹相位\数据集创建\data9_k\gt_phi',
        n_test = 1000,
        model_path='real_model(V6).pth',
        device='cuda',
        batch_size = 1,
        visualize_n = 1,
        save_results = True,
        result_dir = './test_results'
    )


    # 访问结果
    print(f"\n梯度X平均RMSE: {results['avg_rmse_grad_x']:.6f}")
    print(f"相位绝对平均RMSE: {results['avg_rmse_phi_abs']:.6f}")
    print(f"相位相对平均RMSE: {results['avg_rmse_phi_rel']:.6f}")

import os
import os
import re
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Net import GradNet,DCTPoisson
import torch.nn.functional as F

import numpy as np
from typing import List, Union
import pickle
import os
# ===============================
# 1. Dataset（与训练严格一致）
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
        i = idx + 1
        x = self._load_mat_array(os.path.join(self.input_dir, f'I_{i}.mat'))
        y = self._load_mat_array(os.path.join(self.label_dir, f'O_{i}.mat'))
        z = self._load_mat_array(os.path.join(self.phi_dir, f'P_{i}.mat'))

        # (H,W,C) -> (C,H,W)
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()
        y = torch.from_numpy(y.transpose(2, 0, 1)).float()

        #x = (x - 1.0).clamp(-1.0, 1.0)
        # 修正：之前这里 z 赋值成了 y
        z = torch.from_numpy(z).unsqueeze(0).float() #if z.ndim == 2 else torch.from_numpy(z.transpose(2, 0, 1)).float()

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        z = z.unsqueeze(0)

        # 下采样到 256×256
        # x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        # y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=False)
        # z = F.interpolate(z, size=(256, 256), mode='bilinear', align_corners=False)

        # 去掉 batch 维
        x = x.squeeze(0)
        y = y.squeeze(0)
        z = z.squeeze(0)

        return {'inp': x, 'lbl': y, 'gtp': z}


# ===============================
# 4. 主测试流程
# ===============================


def _to_numpy(x):
    """Convert tensor/array to squeezed numpy array (handles torch tensors)."""
    try:
        # handle torch tensor
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.array(x)
    return np.squeeze(x)

def _slugify(title):
    """Create filesystem-friendly filename part from title."""
    s = re.sub(r'[^0-9a-zA-Z_-]+', '_', title).strip('_').lower()
    return s if s else 'image'

def visualize(inp, gt_grad, gt_phi, pred_x, pred_y, re_phi, idx=1,
              save_dir='img_phi/vis6', prefix='exp1', dpi=400, cmap='plasma',
              vmin_vmax=None, save_each=True, save_montage=True):

    # convert to numpy and squeeze
    inp = _to_numpy(inp)
    gt_grad = _to_numpy(gt_grad)
    gt_phi = _to_numpy(gt_phi)
    pred_x = _to_numpy(pred_x)
    pred_y = _to_numpy(pred_y)
    re_phi = _to_numpy(re_phi)

    # try to access inp channels like inp[0], inp[1]
    try:
        inp0 = _to_numpy(inp[0])
        inp1 = _to_numpy(inp[1])
    except Exception:
        # fallback: if inp is already two arrays packed differently
        raise ValueError("`inp` must contain two channels accessible as inp[0], inp[1].")

    # gradient error
    err_x = pred_x - gt_grad[0]
    err_y = pred_y - gt_grad[1]

    # phase error (zero-mean / remove DC ambiguity as you had)
    re_phi = re_phi - re_phi.min()
    phi_err = re_phi - gt_phi

    titles = [
        'Input Ix', 'Input Iy',
        'GT φx', 'GT φy',
        'Pred φx', 'Pred φy',
        'Err φx', 'Err φy',
        'GT φ', 'Reconstructed φ', 'φ Error'
    ]

    images = [
        inp0, inp1,
        gt_grad[0], gt_grad[1],
        pred_x, pred_y,
        err_x, err_y,
        gt_phi, re_phi, phi_err
    ]

    n_images = len(images)

    # normalize vmin_vmax argument into list of length n_images
    if vmin_vmax is None:
        vlist = [None] * n_images
    elif isinstance(vmin_vmax, (list, tuple)) and len(vmin_vmax) == n_images:
        vlist = list(vmin_vmax)
    elif isinstance(vmin_vmax, tuple) and len(vmin_vmax) == 2:
        vlist = [vmin_vmax] * n_images
    else:
        raise ValueError("vmin_vmax must be None, a (vmin,vmax) tuple, or a list of length equal to the number of images.")

    # create save directory if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    fname_prefix = prefix if prefix is not None else "sample"

    # --- Create montage figure (3x4) ---
    cols = 4
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(14, 9))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:n_images]):
        im = ax.imshow(images[i], cmap=cmap, vmin=(vlist[i][0] if vlist[i] else None),
                       vmax=(vlist[i][1] if vlist[i] else None))
        ax.set_title(titles[i])
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046)

    # hide any remaining empty subplots (if any)
    for ax in axes[n_images:]:
        ax.axis('off')

    plt.suptitle(f'Test Sample #{idx}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- Save each image separately (with its own colorbar) ---
    if save_dir is not None and save_each:
        for i, img in enumerate(images):
            fig2 = plt.figure(figsize=(6,5))
            ax2 = fig2.add_subplot(111)
            vmin = (vlist[i][0] if vlist[i] else None)
            vmax = (vlist[i][1] if vlist[i] else None)
            im2 = ax2.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            #ax2.set_title(titles[i])
            ax2.axis('off')
            cbar = fig2.colorbar(im2, fraction=0.046)
            safe_title = _slugify(titles[i])
            filename = f"{fname_prefix}_{idx:04d}_{i+1:02d}_{safe_title}.png"
            path = os.path.join(save_dir, filename)
            fig2.savefig(path, bbox_inches='tight', dpi=dpi)
            plt.close(fig2)

    # --- Save montage figure ---
    if save_dir is not None and save_montage:
        montage_name = f"{fname_prefix}_{idx:04d}_montage.png"
        montage_path = os.path.join(save_dir, montage_name)
        fig.savefig(montage_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    return {
        "saved_each": save_each and (save_dir is not None),
        "saved_montage": save_montage and (save_dir is not None),
        "save_dir": save_dir
    }



# def visualize(inp, gt_grad, gt_phi, pred_x, pred_y, re_phi, idx=1):
#
#     # squeeze batch/channel if needed
#     if gt_phi.ndim == 3:
#         gt_phi = gt_phi.squeeze(0)
#     if re_phi.ndim == 3:
#         re_phi = re_phi.squeeze(0)
#     if pred_x.ndim == 3:
#         pred_x = pred_x.squeeze(0)
#     if pred_y.ndim == 3:
#         pred_y = pred_y.squeeze(0)
#
#     # gradient error
#
#     err_x = pred_x - gt_grad[0]
#     err_y = pred_y - gt_grad[1]
#
#     # phase error (zero-mean to remove DC ambiguity)
#     re_phi = re_phi - re_phi.min()
#     phi_err = re_phi - gt_phi  #绝对误差
#
#     # phi_err2 = re_phi - gt_phi
#     # phi_err2 = phi_err2 - phi_err2.mean() #去偏后的随机误差
#
#     titles = [
#         'Input Ix', 'Input Iy',
#         'GT φx', 'GT φy',
#         'Pred φx', 'Pred φy',
#         'Err φx', 'Err φy',
#         'GT φ', 'Reconstructed φ', 'φ Error'
#     ]
#
#     images = [
#         inp[0], inp[1],
#         gt_grad[0], gt_grad[1],
#         pred_x, pred_y,
#         err_x, err_y,
#         gt_phi, re_phi, phi_err
#     ]
#
#     fig, axes = plt.subplots(3, 4, figsize=(14, 9))
#     axes = axes.flatten()
#
#     for i, ax in enumerate(axes[:len(images)]):
#         im = ax.imshow(images[i], cmap='plasma') #plasma,viridis
#         ax.set_title(titles[i])
#         ax.axis('off')
#         fig.colorbar(im, ax=ax, fraction=0.046)
#
#     plt.suptitle(f'Test Sample #{idx}', fontsize=14)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

def test_model(test_input_dir,
               test_label_dir,
               phi_dir,
               n_test,
               model_path,
               device='cuda',
               batch_size=1,
               visualize_n=3):

    # choose device: prefer CUDA only if requested and available
    use_cuda_requested = isinstance(device, str) and 'cuda' in device.lower()
    device = torch.device('cuda' if use_cuda_requested and torch.cuda.is_available() else 'cpu')

    # dataset & loader
    test_set = MatDatasetStrict(test_input_dir, test_label_dir, phi_dir, n_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # model
    model = GradNet().to(device)
    poisson = DCTPoisson(precompute_shape=(256,256)).to(device)

    # try newer (weights_only) API first, fallback for older torch
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        # older torch versions do not support weights_only argument
        ckpt = torch.load(model_path, map_location=device)

    # ckpt could be either a state_dict or a dict containing 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # assume it's already a state_dict
        state_dict = ckpt
    else:
        raise RuntimeError('Unrecognized checkpoint format. Expected state_dict or dict with "model_state_dict".')

    model.load_state_dict(state_dict)
    model.eval()

    print(f'\u2713 模型已加载：{model_path}')
    print(f'\u2713 测试样本数：{n_test}')

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inp = batch['inp'].to(device)
            gt = batch['lbl'].to(device)
            gt_phi = batch['gtp'].to(device)
            # print(gt_phi.shape)

            pred_x, pred_y = model(inp)
            pred_x[:, :, :, 255] = 0.0
            pred_y[:, :, 255, :] = 0.0
            # print(pred_x.shape)
            re_phi = poisson(pred_x, pred_y)
            # print(re_phi.shape)

            # convert to numpy for visualization (take first item in batch)
            inp_np = inp[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            pred_x_np = pred_x[0].cpu().numpy()
            print(pred_x_np.shape)
            pred_y_np = pred_y[0].cpu().numpy()
            re_phi_np = re_phi[0].cpu().numpy()
            gt_phi_np = gt_phi[0].cpu().numpy()

            # visualize(
            #     inp_np,
            #     gt_np,
            #     gt_phi_np,
            #     pred_x_np,
            #     pred_y_np,
            #     re_phi_np,
            #     idx
            # )
            processed_data = save_pre(re_phi_np, idx=idx, save_path=f"test_results/all_Gen_data.npy")
            print(f"第{idx + 1}次处理完成，处理后的数据形状: {processed_data.shape}")

            if idx + 1 >= visualize_n:
                 break

    return
# ===============================
# 5. 直接运行入口
# 保存函数
def save_pre(re_phi: np.ndarray, idx: int, save_path: [str] = None) -> np.ndarray:
    """
    对输入数组进行预处理并保存到三维数组中

    参数:
    re_phi: 输入的二维数组，形状应为 (height, width)
    idx: 当前测试的索引，用于确定在第三维的位置
    save_path: 保存文件的路径（可选）

    返回:
    预处理后的二维数组
    """
    # 确保输入是二维数组
    if re_phi.ndim == 3:
        # 如果是 (1, H, W)，压缩第一维
        if re_phi.shape[0] == 1:
            re_phi = np.squeeze(re_phi, axis=0)
        else:
            # 如果是 (H, W, 1)，压缩最后一维
            re_phi = np.squeeze(re_phi, axis=-1)

    # 数据预处理：减去最小值
    re_phi = re_phi - re_phi.min()

    # 使用静态变量存储所有结果
    if not hasattr(save_pre, 'all_results'):
        save_pre.all_results = []
        save_pre.current_idx = 0

    # 扩展列表以容纳当前索引的数据
    while len(save_pre.all_results) <= idx:
        save_pre.all_results.append(None)

    # 将预处理后的数据存储到对应位置
    save_pre.all_results[idx] = re_phi.copy()
    save_pre.current_idx = idx + 1

    # 如果提供了保存路径，保存到文件
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # 过滤掉None值，只保留已处理的数据
        valid_results = []
        for arr in save_pre.all_results:
            if arr is not None:
                # 确保每个数组是二维的
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = np.squeeze(arr, axis=0)
                valid_results.append(arr)

        if valid_results:
            # 在第三维上堆叠，得到 (H, W, N)
            all_test_array = np.stack(valid_results, axis=2)

            # 保存为.npy文件
            np.save(save_path, all_test_array)
            print(f"数据已保存到: {save_path}, 形状: {all_test_array.shape}")

    return re_phi


def get_all_results() -> np.ndarray:
    """
    获取所有保存的结果并转换为三维数组

    返回:
    三维数组 (height, width, num_tests)
    """
    if hasattr(save_pre, 'all_results') and save_pre.all_results:
        # 过滤掉None值
        valid_results = [arr for arr in save_pre.all_results if arr is not None]
        if valid_results:
            return np.stack(valid_results, axis=2)
    return np.array([])


def clear_saved_results():
    """清除所有保存的结果"""
    if hasattr(save_pre, 'all_results'):
        save_pre.all_results.clear()




# ===============================

if __name__ == '__main__':
    clear_saved_results()

    #TEST_INPUT_DIR = r'D:\ls\包裹相位\实验数据\general_phi\train_in'  # 修改为实际路径
    TEST_INPUT_DIR = r'D:\ls\包裹相位\实验数据\OPEL_phi\train_in3'
    #TEST_INPUT_DIR = r'D:\ls\包裹相位\wrap_phi2\train_in2'
    #TEST_INPUT_DIR = r'D:\ls\包裹相位\warpzer\train_in'  # zer相位路径
    # TEST_INPUT_DIR = r'D:\pycharmproject\ls\DL\uw_phi\data2_k\train_in'  # 修改为实际路径
    TEST_LABEL_DIR = r'D:\ls\包裹相位\数据集创建\data9_k\train_gt'  # 修改为实际路径
    P_dir = r'D:\ls\包裹相位\数据集创建\data9_k\gt_phi'
    MODEL_PATH = ('real_model(V6).pth')
    N_TEST = 201

    test_model(
        test_input_dir=TEST_INPUT_DIR,
        test_label_dir=TEST_LABEL_DIR,
        phi_dir=P_dir,
        n_test=N_TEST,
        model_path=MODEL_PATH,
        device='cuda',
        batch_size=1,
        visualize_n=100
    )

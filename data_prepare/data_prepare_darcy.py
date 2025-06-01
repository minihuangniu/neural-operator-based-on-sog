from scipy.io import loadmat
import torch
import os
import numpy as np

def process_and_save_darcy_data(mat_path, save_dir, resolutions, num_train, num_test):
    os.makedirs(save_dir, exist_ok=True)

    raw_data = loadmat(mat_path)
    K = raw_data['coeff']
    u = raw_data['sol']

    K = torch.tensor(K, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)

    print(f"原始数据 K: {K.shape}, u: {u.shape}")

    for res in resolutions:
        step = (421 - 1) // (res - 1)  # 等间隔采样步长
        K_sub = K[:, ::step, ::step]  # [N, res, res]
        u_sub = u[:, ::step, ::step]

        # 合并通道，得到 [N, 2, res, res]
        data = torch.stack([K_sub, u_sub], dim=1)

        # 打乱
        #total = data.shape[0]
        #perm = torch.randperm(total)
        #data = data[perm]

        # 保存训练集
        if num_train > 0:
            train_data = data[:num_train]
            train_path = os.path.join(save_dir, f'darcy_train_{res}_n{num_train}.pt')
            torch.save(train_data, train_path)
            print(f"分辨率 {res}: 已保存训练集 {train_path}，shape: {train_data.shape}")

        # 保存测试集
        if num_test > 0:
            test_data = data[:num_test]
            test_path = os.path.join(save_dir, f'darcy_test_{res}_n{num_test}.pt')
            torch.save(test_data, test_path)
            print(f"分辨率 {res}: 已保存测试集 {test_path}，shape: {test_data.shape}")

if __name__ == "__main__":
    # 保存 smooth1 的训练集
    process_and_save_darcy_data(
        mat_path='../data/piececonst_r421_N1024_smooth1.mat',
        save_dir='../data/processed/',
        resolutions=[43, 85, 141, 211],
        num_train=1000,
        num_test=0
    )

    # 保存 smooth2 的测试集
    process_and_save_darcy_data(
        mat_path='../data/piececonst_r421_N1024_smooth2.mat',
        save_dir='../data/processed/',
        resolutions=[43, 85, 141, 211],
        num_train=0,
        num_test=100
    )
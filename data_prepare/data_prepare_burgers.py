from scipy.io import loadmat
import torch
import os

mat_path = 'data/burgers_data_R10.mat'
def process_and_save_burgers_data(mat_path, save_dir, resolutions, num_train=1000, num_test=200):
    os.makedirs(save_dir, exist_ok=True)  # 自动创建保存目录

    raw_data = loadmat(mat_path)
    a = torch.tensor(raw_data['a'], dtype=torch.float32)
    u = torch.tensor(raw_data['u'], dtype=torch.float32)

    print(f"原始数据 a: {a.shape}, u: {u.shape}")

    for res in resolutions:
        # 计算下采样步长
        downsample_ratio = a.shape[1] // res

        # 下采样
        a_sub = a[:, ::downsample_ratio]  # shape: [2048, res]
        u_sub = u[:, ::downsample_ratio]

        # 增加通道维度
        a_sub = a_sub.unsqueeze(1)  # shape: [2048, 1, res]
        u_sub = u_sub.unsqueeze(1)

        #打乱数据顺序
        total = a_sub.shape[0]
        perm = torch.randperm(total)
        a_sub = a_sub[perm]
        u_sub = u_sub[perm]

        # 拆分训练和测试集
        a_train = a_sub[:num_train]
        u_train = u_sub[:num_train]
        a_test = a_sub[-num_test:]
        u_test = u_sub[-num_test:]

        # 保存
        torch.save((a_train, u_train), os.path.join(save_dir, f'burgers_train_{res}_n{num_train}.pt'))
        torch.save((a_test, u_test), os.path.join(save_dir, f'burgers_test_{res}_n{num_test}.pt'))

        print(f"分辨率 {res}: 已保存 train/test 数据集")


if __name__ == "__main__":
    process_and_save_burgers_data(
        mat_path=mat_path,
        save_dir='data/processed/',
        resolutions=[64, 128, 256, 512, 1024],
        num_train=1000,
        num_test=200
    )
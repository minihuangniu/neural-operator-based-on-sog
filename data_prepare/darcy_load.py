import torch
from torch.utils.data import Dataset, DataLoader
import os
from fno_sog.utilities3 import *

class DarcyDataset(Dataset):
    def __init__(self, a, u):
        """
        :param a: 输入张量，形状 [N, 1, res, res]
        :param u: 输出张量，形状 [N, 1, res, res]
        """
        self.a = a
        self.u = u

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        return {
            'a': self.a[idx],
            'u': self.u[idx]
        }

def add_grid(a_tensor):
    # a_tensor: [N, 1, H, W]
    N, _, H, W = a_tensor.shape
    gridx = torch.linspace(0, 1, H)
    gridy = torch.linspace(0, 1, W)
    grid = torch.stack(torch.meshgrid(gridx, gridy, indexing='ij'), dim=0)  # [2, H, W]
    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # [N, 2, H, W]
    a_with_grid = torch.cat([a_tensor, grid], dim=1)  # [N, 3, H, W]
    return a_with_grid

def get_darcy_dataloader(resolution, batch_size, data_dir='../data/processed', normalize=True):
    """
    加载 darcy 数据集的 dataloader，支持全局归一化
    :param resolution: e.g. 21, 43, 85, 141
    :param batch_size: 批次大小
    :param data_dir: .pt 文件目录
    :param normalize: 是否对整个数据集做归一化
    :return: train_loader, test_loader, a_normalizer, u_normalizer
    """
    train_file = os.path.join(data_dir, f'darcy_train_{resolution}_n1000.pt')
    test_file = os.path.join(data_dir, f'darcy_test_{resolution}_n100.pt')

    train_data = torch.load(train_file)
    test_data = torch.load(test_file)

    a_train = train_data[:, 0:1, :, :]
    u_train = train_data[:, 1:2, :, :]
    a_test = test_data[:, 0:1, :, :]
    u_test = test_data[:, 1:2, :, :]

    if normalize:
        a_normalizer = UnitGaussianNormalizer(a_train)
        u_normalizer = UnitGaussianNormalizer(u_train)

        a_train = a_normalizer.encode(a_train)
        u_train = u_normalizer.encode(u_train)
        a_test = a_normalizer.encode(a_test)
    else:
        a_normalizer = None
        u_normalizer = None

    a_train = add_grid(a_train)  # [N, 3, H, W]
    a_test = add_grid(a_test)

    train_dataset = DarcyDataset(a_train, u_train)
    test_dataset = DarcyDataset(a_test, u_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, a_normalizer, u_normalizer
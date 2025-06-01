import torch
import os
from torch.utils.data import DataLoader, Dataset


class BurgersDataset(Dataset):
    def __init__(self, x_data, y_data):
        """
        初始化数据集
        :param x_data: 输入数据 (例如 a_train)
        :param y_data: 标签数据 (例如 u_train)
        """
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.x_data)

    def __getitem__(self, idx):
        """
        通过索引返回数据对
        :param idx: 数据索引
        :return: 返回输入和标签
        """
        x = self.x_data[idx]  # x_data[idx] 应该是一个 tensor
        y = self.y_data[idx]  # y_data[idx] 应该是一个 tensor
        return {'a': x, 'u': y}

def get_burgers_dataloader(resolution, batch_size, data_dir='../data/processed'):
    """
    传入训练和测试数据的路径，返回相应的 DataLoader
    :param data_dir:默认文件根目录
    :param resolution: res
    :param batch_size: 批量大小
    :return: 训练和测试的 DataLoader
    """

    # 加载训练和测试数据
    train_file = os.path.join(data_dir, f'burgers_train_{resolution}_n1000.pt')
    test_file = os.path.join(data_dir, f'burgers_test_{resolution}_n200.pt')

    x_train, y_train = torch.load(train_file)
    x_test, y_test = torch.load(test_file)

    print(f"a's type in .pt file is: {type(x_train)}")  # 应该是 tensor
    print(f"u's type in .pt file is: {type(y_train)}")  # 应该是 tensor

    # 创建自定义 Dataset
    train_dataset = BurgersDataset(x_train, y_train)
    test_dataset = BurgersDataset(x_test, y_test)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from fno_sog.models.lploss import LpLoss
from fno_sog.models.SFNO2d import sfno2d
from fno_sog.data_prepare.darcy_load import get_darcy_dataloader
from fno_sog.utils import *

def darcy_test(model, test_loader, criterion, device, u_normalizer):
    model.eval()
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for data in test_loader:
            a = data['a'].to(device)
            u = data['u'].to(device)
            testbt = a.size(0)
            output = model(a)
            output = output.squeeze()
            output = u_normalizer.decode(output)
            loss = criterion(output.view(testbt, -1), u.view(testbt, -1))

            test_loss += loss.item()
            test_samples += testbt

    return test_loss / test_samples

def plot_prediction_vs_truth(model, data_loader, device, u_normalizer, sample_idx=0):
    """
    绘制模型预测值与真实值的3D表面图对比。

    Args:
        model: 训练好的模型
        data_loader: 测试集的dataloader
        device: 设备 (cpu or cuda)
        sample_idx: 取batch中第几个样本进行绘制
    """
    model.eval()
    data_iter = iter(data_loader)
    data = next(data_iter)
    a = data['a'].to(device)
    u_true = data['u'].to(device)

    with torch.no_grad():
        u_pred = model(a)
        u_pred = u_normalizer.decode(u_pred)

    true_sample = u_true[sample_idx, 0].cpu().numpy()  # 取第sample_idx个样本
    pred_sample = u_pred[sample_idx, 0].cpu().numpy()

    H, W = true_sample.shape
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 6))

    # 画真实值
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, true_sample, cmap='viridis')
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')

    # 画预测值
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, pred_sample, cmap='plasma')
    ax2.set_title('Prediction')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u_pred(x,y)')

    plt.suptitle('Comparison between Ground Truth and Prediction')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 600
    modes = 12
    width = 32
    hidden_channels = 64
    learning_rate = 1e-3
    weight_decay = 1e-4
    resolution = 141
    batchsize = 20
    stepsize = 100
    gamma = 0.5

    train_loader, test_loader, a_normalizer, u_normalizer = get_darcy_dataloader(resolution, batchsize)
    u_normalizer.cuda()
    model = sfno2d(modes, modes, width, hidden_channels, sog_mode=True).to(device)
    n_params = count_model_params(model)
    print(f'\nThe model has {n_params} parameters.')
    myloss = LpLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    training_loss = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_samples = 0
        for i, data in enumerate(train_loader, 0):
            a = data['a'].to(device)
            u = data['u'].to(device)
            batch_size = a.size(0)
            optimizer.zero_grad()
            y_pred = model(a)
            y_pred = u_normalizer.decode(y_pred)
            u = u_normalizer.decode(u)
            loss = myloss(y_pred.view(batchsize, -1), u.view(batchsize, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += batch_size
            training_loss.append(loss.item()/batch_size)
            print(f'Epoch: {epoch}/{epochs} Total Loss: {loss.item()/batch_size:.6f}')
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch: {epoch}/{epochs} Total Loss: {total_loss / total_samples:.6f} learning rate: {current_lr}')
        scheduler.step()

    test_loss = darcy_test(model, test_loader, myloss, device, u_normalizer)
    print(f'Loss on test set is: {test_loss:.6f}')

    plt.figure(figsize=(8, 8))
    plt.plot(training_loss, label='train loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss per Steps')
    plt.legend()
    plt.show()

    plot_prediction_vs_truth(model, test_loader, device, u_normalizer)


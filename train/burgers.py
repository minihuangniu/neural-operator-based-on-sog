import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from fno_sog.models.lploss import LpLoss
from fno_sog.models.SFNO1d import sfno1d
from fno_sog.data_prepare.burgers_load import get_burgers_dataloader
from fno_sog.utils import *

def burgers_test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for data in test_loader:
            a = data['a'].to(device)
            u = data['u'].to(device)
            test_bt = a.size(0)
            output = model(a)
            loss = criterion(output.view(test_bt, -1), u.view(test_bt, -1))

            test_loss += loss.item()
            test_samples += test_bt

    return test_loss / test_samples

def plot_burgers(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            a = data['a'].to(device)
            u = data['u'].to(device)
            y_pred = model(a)
            true_u = u[0, 0].cpu().numpy()
            pred_u = y_pred[0, 0].cpu().numpy()

            x = np.linspace(0, 1, len(true_u))

            plt.figure(figsize=(8, 8))
            plt.plot(x, true_u, label='Ground Truth', marker='o')
            plt.plot(x, pred_u, label='Prediction', marker='x')
            plt.xlabel('x')
            plt.ylabel(f'u(x)')
            plt.title(f'Burgers Equation Fitting')
            plt.legend()
            plt.grid(True)
            plt.show()
            break

if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    epochs = 600
    modes = 16
    width = 64
    hidden_channels = 256
    learning_rate = 1e-3
    weight_decay = 1e-4
    resolution = 1024
    batchsize = 20
    stepsize = 100
    gamma = 0.5

    train_loader, test_loader = get_burgers_dataloader(resolution, batchsize)

    model = sfno1d(modes, width, hidden_channels, sog_mode=True).to(device)
    n_params = count_model_params(model)
    print(f'\nThe model has {n_params} parameters.')
    myloss = LpLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} | mean: {param.data.mean():.6f} | std: {param.data.std():.6f}")


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
            loss = myloss(y_pred.view(batch_size, -1), u.view(batch_size, -1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_samples += batch_size
            training_loss.append(loss.item()/batch_size)
            print(f'Epoch: {epoch}/{epochs} Total Loss: {loss.item()/batch_size:.6f}')
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch: {epoch}/{epochs} Total Loss: {(total_loss / total_samples):.6f} learning rate: {current_lr}')
        scheduler.step()

    test_loss = burgers_test(model, test_loader, myloss, device)
    print(f'Loss on test set is: {test_loss:.6f}')

    plt.figure(figsize=(8, 8))
    plt.plot(training_loss, label='train loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss per Steps')
    plt.legend()
    plt.show()

    plot_burgers(model, device, test_loader)

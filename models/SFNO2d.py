import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        x_ft = torch.fft.rfftn(x, dim=(-2,-1), norm='ortho')
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device,
                             dtype=torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        return x

class sog_net2d(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(sog_net2d, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        s = 1/self.hidden_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1, self.hidden_channels, 1, 1))
        self.conv3 = nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=1)


    def forward(self, x):
        x = torch.square(x)
        x = self.conv1(x)
        x = -F.relu(self.conv2(x))
        x = torch.exp(x) * self.scale
        x = self.conv3(x)
        return x

class sfno2d(nn.Module):
    def __init__(self, modes1, modes2, width, hidden_channels, sog_mode=True):
        super(sfno2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.hidden_channels = hidden_channels
        self.sog_mode = sog_mode
        self.fc0 = nn.Linear(3, self.width)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        if self.sog_mode:
            print(f"当前已启用sog模块")
            self.sog0 = sog_net2d(self.width, self.hidden_channels)
            self.sog1 = sog_net2d(self.width, self.hidden_channels)
            self.sog2 = sog_net2d(self.width, self.hidden_channels)
        else:
            print(f"当前已关闭sog模块")

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        #print(x.shape)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        if self.sog_mode:
            x = x + self.sog0(x)
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        if self.sog_mode:
            x = x + self.sog1(x)
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        if self.sog_mode:
            x = x + self.sog2(x)
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)
        x = F.relu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        return x








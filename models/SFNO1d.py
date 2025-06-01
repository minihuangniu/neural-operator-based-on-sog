import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes are kept, at most floor(N/2) + 1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # Compute Fourier coeffcients
        x_ft = torch.fft.rfft(x)  # [Batch, C_in, Nx] -> [Batch, C_in, Nx//2 + 1], eg. [20, 64, 128] -> [20, 64, 65]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-1) // 2 + 1, device=x.device,
                             dtype=torch.cfloat)  # [Batch, Nc, Nx//2 + 1], eg. [20, 64, 65]
        # [Batch, C_in, self.modes1] * [C_in, C_out, self.modes1] -> [Batch, C_out, self.modes1]
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(
            -1))  # [Batch, C_out, self.modes1] -> [Batch, C_out, Nx], eg. [20, 64, 65] -> [20, 64, 128]
        return x

class sog_net1d(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(sog_net1d, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        s = 1/self.hidden_channels
        self.conv1 = nn.Conv1d(self.in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(hidden_channels, 1))
        self.conv3 = nn.Conv1d(hidden_channels, self.in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)


    def forward(self, x):
        x = torch.square(x)
        x = self.conv1(x)
        x = -F.relu(self.conv2(x))
        x = torch.exp(x) * self.scale
        x = self.conv3(x)
        return x

class sfno1d(nn.Module):
    def __init__(self, modes, width, hidden_channels, sog_mode=True):
        """
        1D Fourier Neural Operator model.

        Args:
            modes (int): Number of spectral modes.
            width (int): Number of hidden channels.
            sog_mode (bool): Whether to enable the sog module.
        """
        super(sfno1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.hidden_channels = hidden_channels
        self.sog_mode = sog_mode
        self.fc0 = nn.Linear(1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        if self.sog_mode:
            print(f"当前已启用sog模块")
            self.sog0 = sog_net1d(self.width, self.hidden_channels)
            self.sog1 = sog_net1d(self.width, self.hidden_channels)
            self.sog2 = sog_net1d(self.width, self.hidden_channels)
        else:
            print(f"当前已关闭sog模块")

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.fc0(x)  # [Batch, Nx, C] -> [Batch, Nx, Width], eg. [20, 128, 2] -> [20, 128, 64]
        x = x.permute(0, 2, 1)  # [Batch, C, Nx], eg. [20, 64, 128]


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        if self.sog_mode:
            x = x + self.sog0(x)
        x = F.relu(x)



        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        if self.sog_mode:
            x = x + self.sog1(x)
        x = F.relu(x)



        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        if self.sog_mode:
            x = x + self.sog2(x)
        x = F.relu(x)


        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2


        # stage 3: put the channel back to 1
        x = x.permute(0, 2, 1)  # [Batch, Nx, C], eg. [20, 128, 64]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.permute(0, 2, 1)

        return x
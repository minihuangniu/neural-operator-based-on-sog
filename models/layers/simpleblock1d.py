from fno_sog.models.layers.spectralconv1d import SpectralConv1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width, hidden_channels):
        """
        1D Fourier Neural Operator model.

        Args:
            modes (int): Number of spectral modes.
            width (int): Number of hidden channel.
        """
        super(SimpleBlock1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.hidden_channels = hidden_channels
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

        #self.sog0 = sog_net1d(self.width, self.hidden_channels)
        #self.sog1 = sog_net1d(self.width, self.hidden_channels)
        #self.sog2 = sog_net1d(self.width, self.hidden_channels)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.fc0(x)  # [Batch, Nx, C] -> [Batch, Nx, Width], eg. [20, 128, 2] -> [20, 128, 64]
        x = x.permute(0, 2, 1)  # [Batch, C, Nx], eg. [20, 64, 128]


        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        #x = x + self.sog0(x)
        x = F.relu(x)



        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        #x = x + self.sog1(x)
        x = F.relu(x)



        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        #x = x + self.sog2(x)
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
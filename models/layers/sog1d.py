import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np

torch.manual_seed(0)
np.random.seed(0)


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




import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        def compl_mul2d(self, input, weights):
            # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
            return torch.einsum("bixy,ioxy->boxy", input, weights)

        def forward(self, x):
            x_ft = torch.fft.rfftn(x, dim=(-2,-1))

            out_ft = torch.zeros(x.shape[0], self.out_channels, x.size(-2), x.size(-1)//2 + 1, device=x.device,
                                 dtype=torch.cfloat)
            out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

            x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)))
            return x
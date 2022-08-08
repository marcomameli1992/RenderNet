import torch
from torch import nn as NN
from torchvision import ops as OPS

class SqueezeExcitationBlock(NN.Module):

    def __init__(self, in_channels, squeeze_channels):
        super(SqueezeExcitationBlock, self).__init__()
        self.SE = OPS.SqueezeExcitation(input_channels=in_channels, squeeze_channels=squeeze_channels, activation=NN.SiLU)

    def forward(self, x):
        return self.SE(x)


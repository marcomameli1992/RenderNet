import torch
from torch import nn as NN


class ConvolutionalBlock(NN.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 batch_norm=True, activation=True):
        super(ConvolutionalBlock, self).__init__()
        self.conv = NN.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm = NN.BatchNorm2d(out_channels) if batch_norm else NN.Identity()
        self.activation = NN.SiLU(True) if activation else NN.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

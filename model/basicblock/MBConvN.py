import torch
from torch import nn as NN
from .ConvolutionalBlock import ConvolutionalBlock
from .SqueezeExcitationBlock import SqueezeExcitationBlock
from .DropBlock import DropBlock

class MBConvN(NN.Module):
    def __init__(self, input_channels, output_channels, expansion_factor, kernel_size=3, stride=1, r=24, p=0):
        super(MBConvN, self).__init__()
        padding = (kernel_size - 1) // 2
        expanded = expansion_factor * input_channels
        self.skip_connection = (input_channels == output_channels) and (stride == 1)
        self.expanded_conv = NN.Identity() if (expansion_factor == 1) else ConvolutionalBlock(input_channels, expanded, kernel_size, stride, padding, groups=expanded)

        self.squeeze_and_excitation = SqueezeExcitationBlock(expanded, r)
        self.reduce_conv = ConvolutionalBlock(expanded, output_channels, kernel_size=1, activation=False)

        self.dropblock = DropBlock(p)

    def forward(self, x):

        residual = x
        x = self.expanded_conv(x)
        x = self.squeeze_and_excitation(x)
        x = self.reduce_conv(x)
        if self.skip_connection:
            x = self.dropblock(x)
            x = x + residual
        return x


class MBConv1(MBConvN):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, r=24, p=0):
        super(MBConv1, self).__init__(input_channels, output_channels, expansion_factor=1, kernel_size=kernel_size, stride=stride, r=r, p=p)

class MBConv6(MBConvN):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, r=24, p=0):
        super(MBConv6, self).__init__(input_channels, output_channels, expansion_factor=6, kernel_size=kernel_size, stride=stride, r=r, p=p)
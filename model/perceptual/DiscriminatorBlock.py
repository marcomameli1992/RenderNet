import torch
from torch import nn as NN

class CGLLayer(NN.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(CGLLayer, self).__init__()
        self.conv = NN.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn = NN.GroupNorm(8, out_channels)
        self.relu = NN.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class CLCLayer(NN.Module):

    def __init__(self, in_channels, out_channels, mid_channels, kernel_size: list = [3, 3], stride: list = [1, 1], padding: list = [1, 1]):
        super(CLCLayer, self).__init__()
        self.conv1 = NN.Conv2d(in_channels, mid_channels, kernel_size[0], stride[0], padding[0])
        self.act = NN.LeakyReLU()
        self.conv2 = NN.Conv2d(mid_channels, out_channels, kernel_size[1], stride[1], padding[1])

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class CGLBlock(NN.Module):

    def __init__(self, in_channels, out_channels, d=64):
        super(CGLBlock, self).__init__()
        self.l1 = CGLLayer(in_channels, d)
        self.l2 = CGLLayer(d, d * 2)
        self.l3 = CGLLayer(d * 2, d * 4)
        self.l4 = CGLLayer(d * 4, d * 8, stride=1)
        self.l5 = CGLLayer(d * 8, out_channels, stride=1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        return x

class DiscriminatorBlock(NN.Module):
    def __init__(self, input_channel, d=64):
        super(DiscriminatorBlock, self).__init__()

        self.cgl_block = CGLBlock(input_channel, 256, d)
        self.clc_layer = CLCLayer(256, 1, 64)

    def forward(self, x):
        x = self.cgl_block(x)
        x = self.clc_layer(x)
        return x
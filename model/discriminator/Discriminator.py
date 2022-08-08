from collections import namedtuple

import torch
from torch import nn as NN
from .PerceptualNetwork import PerceptualNetwork
from .DiscriminatorBlock import DiscriminatorBlock


class PerceptualDiscriminator(NN.Module):

    def __init__(self):
        super(PerceptualDiscriminator, self).__init__()

        self.perceptualNetwork = PerceptualNetwork()
        self.D1 = DiscriminatorBlock(64, d=64)
        self.D2 = DiscriminatorBlock(128, d=32)
        self.D3 = DiscriminatorBlock(256, d=16)
        self.D4 = DiscriminatorBlock(512, d=8)

    def forward(self, x):
        x = self.perceptualNetwork(x)
        d1 = self.D1(x.relu1_2)
        d2 = self.D2(x.relu2_2)
        d3 = self.D3(x.relu3_3)
        d4 = self.D4(x.relu4_3)

        perceptual_output = namedtuple("PerceptOut", ['d1', 'd2', 'd3', 'd4'])
        out = perceptual_output(d1, d2, d3, d4)
        return out
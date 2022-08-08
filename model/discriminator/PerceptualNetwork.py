from collections import namedtuple

import torch
from torch import nn as NN
from torchvision.models import vgg16

class PerceptualLossNetwork(NN.Module):

    def __init__(self):
        super(PerceptualLossNetwork, self).__init__()
        self.vgg16_layers = vgg16(pretrained=True).features

        self.layer_name_mapping = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3',
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg16_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return namedtuple('PerceptualLossOutput', output.keys())(**output)
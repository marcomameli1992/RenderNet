import torch
from torch import nn as NN
from .Decoder import Decoder
from .Encoder import Encoder
from torchvision import ops as OPS

class Generator(NN.Module):

        def __init__(self, in_channels, out_channels, multiplier, use_all=True, use_albedo=False, use_depth=False, use_emissive=False, use_metalness=False, use_normal=False, use_roughness=False, use_position=False):

            super(Generator, self).__init__()

            self.encoder = Encoder(use_all=use_all, use_albedo=use_albedo, use_depth=use_depth, use_emissive=use_emissive, use_metalness=use_metalness, use_normal=use_normal, use_roughness=use_roughness, use_position=use_position)
            self.squeeze_and_excitation_layer = OPS.SqueezeExcitation(input_channels=7680, squeeze_channels=7680, activation=NN.SiLU)
            self.decoder = Decoder(in_channels, out_channels, multiplier=multiplier)

        def forward(self, x):
            x1 = self.encoder(x)

            x2 = self.decoder(x1)

            return x2
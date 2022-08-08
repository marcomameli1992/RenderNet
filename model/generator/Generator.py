import torch
from torch import nn as NN
from .Decoder import Decoder
from .Encoder import Encoder
from torchvision import ops as OPS

class Generator(NN.Module):

        def __init__(self, in_channels, out_channels):

            super(Generator, self).__init__()

            self.encoder = Encoder()
            self.squeeze_and_excitation_layer = OPS.SqueezeExcitation(input_channels=7680, squeeze_channels=7680, activation=NN.SiLU)
            self.decoder = Decoder(in_channels, out_channels)

        def forward(self, x):
            x1 = self.encoder(x)

            x2 = self.decoder(x1)

            return x2
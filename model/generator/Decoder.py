import torch
from torch import nn as NN
from basicblock.MBConvN import MBConv1

class Decoder(NN.Module):

    def __init__(self, in_channels, out_channels):

        super(Decoder, self).__init__()

        self.transpose_convolutional_block = NN.ConvTranspose2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1, bias=True)

        self.mb_conv_1 = MBConv1(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, r=24, p=1)

        self.mb_conv_2 = MBConv1(in_channels=out_channels, out_channels=16, kernel_size=3, stride=1, r=24, p=1)

        self.sigmoid = NN.Sigmoid()

        self.mb_conv_3 = MBConv1(in_channels=16, out_channels=out_channels, kernel_size=3, stride=1, r=24, p=1)


    def forward(self, x):
        x = self.transpose_convolutional_block(x)
        x = self.mb_conv_1(x)

        x = self.mb_conv_2(x)

        x = self.sigmoid(x)

        x = self.mb_conv_3(x)

        return x
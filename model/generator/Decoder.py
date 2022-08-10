import torch
from torch import nn as NN

class Decoder(NN.Module):

    def __init__(self, in_channels, out_channels=3):

        super(Decoder, self).__init__()

        self.block7 = NN.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=1, stride=1, padding=0)
        self.block7_1 = NN.Conv2d(int(in_channels / 2), int(in_channels / 2), kernel_size=1, stride=1, padding=0)
        self.block6 = NN.ConvTranspose2d(4928, int(in_channels / 4), kernel_size=4, stride=1, padding=0, output_padding=1, dilation=2)
        self.block6_1 = NN.Conv2d(int(in_channels / 4), int(in_channels / 4), kernel_size=1, stride=1, padding=0)
        self.block5 = NN.ConvTranspose2d(2688, int(in_channels / 8), kernel_size=1, stride=1, padding=0)
        self.block5_1 = NN.Conv2d(int(in_channels / 8), int(in_channels / 8), kernel_size=1, stride=1, padding=0)
        self.block4 = NN.ConvTranspose2d(1680, int(in_channels / 16), kernel_size=4, stride=2, padding=1)
        self.block4_1 = NN.Conv2d(int(in_channels / 16), int(in_channels / 16), kernel_size=1, stride=1, padding=0)
        self.block3 = NN.ConvTranspose2d(840, int(in_channels / 32), kernel_size=4, stride=2, padding=1)
        self.block3_1 = NN.Conv2d(int(in_channels / 32), int(in_channels / 32), kernel_size=1, stride=1, padding=0)
        self.block2 = NN.ConvTranspose2d(476, int(in_channels / 64), kernel_size=4, stride=2, padding=1)
        self.block2_1 = NN.Conv2d(int(in_channels / 64), int(in_channels / 64), kernel_size=1, stride=1, padding=0)
        self.block1 = NN.ConvTranspose2d(294, out_channels, kernel_size=4, stride=2, padding=1)
        self.block1_1 = NN.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x: dict):

        block7_input = []
        for channel in x.keys():
            block7_input.append(x[channel]['7'])
        block7_input = torch.cat(block7_input, dim=1)

        block7_out = self.block7(block7_input)
        block7_out = self.block7_1(block7_out)

        block6_input = []
        for channel in x.keys():
            block6_input.append(x[channel]['6'])
        block6_input = torch.cat(block6_input, dim=1)
        block6_input = torch.cat([block7_out, block6_input], dim=1)

        block6_out = self.block6(block6_input)
        block6_out = self.block6_1(block6_out)

        block5_input = []
        for channel in x.keys():
            block5_input.append(x[channel]['5'])
        block5_input = torch.cat(block5_input, dim=1)
        block5_input = torch.cat([block6_out, block5_input], dim=1)

        block5_out = self.block5(block5_input)
        block5_out = self.block5_1(block5_out)

        block4_input = []
        for channel in x.keys():
            block4_input.append(x[channel]['4'])
        block4_input = torch.cat(block4_input, dim=1)
        block4_input = torch.cat([block5_out, block4_input], dim=1)

        block4_out = self.block4(block4_input)
        block4_out = self.block4_1(block4_out)

        block3_input = []
        for channel in x.keys():
            block3_input.append(x[channel]['3'])
        block3_input = torch.cat(block3_input, dim=1)
        block3_input = torch.cat([block4_out, block3_input], dim=1)

        block3_out = self.block3(block3_input)
        block3_out = self.block3_1(block3_out)

        block2_input = []
        for channel in x.keys():
            block2_input.append(x[channel]['2'])
        block2_input = torch.cat(block2_input, dim=1)
        block2_input = torch.cat([block3_out, block2_input], dim=1)

        block2_out = self.block2(block2_input)
        block2_out = self.block2_1(block2_out)

        block1_input = []
        for channel in x.keys():
            block1_input.append(x[channel]['1'])
        block1_input = torch.cat(block1_input, dim=1)
        block1_input = torch.cat([block2_out, block1_input], dim=1)

        block1_out = self.block1(block1_input)
        block1_out = self.block1_1(block1_out)

        return block1_out
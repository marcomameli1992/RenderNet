import torch
from torch import nn as NN

class Decoder(NN.Module):

    def __init__(self, in_channels, out_channels=3):

        super(Decoder, self).__init__()

        self.block7 = NN.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.block6 = NN.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                         output_padding=1)
        self.block5 = NN.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                         output_padding=1)
        self.block4 = NN.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                         output_padding=1)
        self.block3 = NN.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1,
                                         output_padding=1)
        self.block2 = NN.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x: dict):

        block7_input = []
        for channel in x.keys():
            block7_input.append(x[channel]['7'])
        block7_input = torch.cat(block7_input, dim=1)
        block7_out = self.block7(block7_input)

        block6_input = []
        for channel in x.keys():
            block6_input.append(x[channel]['6'])
        block6_input = torch.cat(block6_input, dim=1)
        print("Block 6 input", block6_input.shape)
        print("Black 7 out", block7_out.shape)
        block6_input = torch.cat([block7_out, block6_input], dim=1)

        block6_out = self.block6(block6_input)
        block5_input = []
        for channel in x.keys():
            block5_input.append(x[channel]['4'])
        block5_input = torch.cat(block5_input, dim=1)
        block5_input = torch.cat([block6_out, block5_input], dim=1)

        block5_out = self.block5(block5_input)
        block4_input = []
        for channel in x.keys():
            block4_input.append(x[channel]['3'])
        block4_input = torch.cat(block4_input, dim=1)
        block4_input = torch.cat([block5_out, block4_input], dim=1)

        block4_out = self.block4(block4_input)
        block3_input = []
        for channel in x.keys():
            block3_input.append(x[channel]['2'])
        block3_input = torch.cat(block3_input, dim=1)
        block3_input = torch.cat([block4_out, block3_input], dim=1)

        block3_out = self.block3(block3_input)

        block2_out = self.block2(block3_out)

        return block2_out
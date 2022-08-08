import torch
from torch import nn as NN

class DropBlock(NN.Module):
    def __init__(self, p=0):
        super(DropBlock, self).__init__()
        self.p = p

    def forward(self, x):
        if (not self.p) or (not self.training):
            return x

        batch_size = len(x)
        random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        bit_mask = self.p<random_tensor

        x = x.div(1-self.p)
        x = x * bit_mask
        return x


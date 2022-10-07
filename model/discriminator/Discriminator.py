import torch
from torch import nn as NN
from torchvision.models import vgg16
import torchvision.models as models


class Discriminator(NN.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.vgg16 = vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.avgpool = NN.AdaptiveAvgPool2d((7, 7))
        self.classifier = NN.Sequential(
            NN.Linear(512 * 7 * 7, 4096),
            NN.ReLU(True),
            NN.Dropout(p=0.5),
            NN.Linear(4096, 4096),
            NN.ReLU(True),
            NN.Dropout(p=0.5),
            NN.Linear(4096, 2),
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

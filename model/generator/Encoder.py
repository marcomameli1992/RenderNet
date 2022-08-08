import torch
from torch import nn as NN
from torchvision.models import efficientnet_b7

class Encoder(NN.Module):

    def __init__(self, use_pretrained=True):
        super(Encoder, self).__init__()

        self.metalness_encoder = efficientnet_b7(use_pretrained)
        self.roughness_encoder = efficientnet_b7(use_pretrained)
        self.depth_encoder = efficientnet_b7(use_pretrained)
        self.albedo_encoder = efficientnet_b7(use_pretrained)
        self.normal_encoder = efficientnet_b7(use_pretrained)
        self.emissive_encoder = efficientnet_b7(use_pretrained)

    def forward(self, x):
        metalness_feature = self.metalness_encoder(x['metalness']).e
        roughness_feature = self.roughness_encoder(x['roughness'])
        depth_feature = self.depth_encoder(x['depth'])
        albedo_feature = self.albedo_encoder(x['albedo'])
        normal_feature = self.normal_encoder(x['normal'])
        emissive_feature = self.emissive_encoder(x['emissive'])

        return {'metalness': metalness_feature, 'roughness': roughness_feature, 'depth': depth_feature, 'albedo': albedo_feature, 'normal': normal_feature, 'emissive': emissive_feature}



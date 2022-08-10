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
        self.eevee_encoder = efficientnet_b7(use_pretrained)

        self.layer_name_mapping = ['1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        metalness_feature = {}
        metalness_f = x['metalness']
        for name, module in self.metalness_encoder.features._modules.items():
            metalness_f = module(metalness_f)
            if name in self.layer_name_mapping:
                metalness_feature[name] = metalness_f

        roughness_feature = {}
        roughness_f = x['roughness']
        for name, module in self.roughness_encoder.features._modules.items():
            roughness_f = module(roughness_f)
            if name in self.layer_name_mapping:
                roughness_feature[name] = roughness_f

        depth_feature = {}
        depth_f = x['depth']
        for name, module in self.depth_encoder.features._modules.items():
            depth_f = module(depth_f)
            if name in self.layer_name_mapping:
                depth_feature[name] = depth_f

        albedo_feature = {}
        albedo_f = x['albedo']
        for name, module in self.albedo_encoder.features._modules.items():
            albedo_f = module(albedo_f)
            if name in self.layer_name_mapping:
                albedo_feature[name] = albedo_f

        normal_feature = {}
        normal_f = x['normal']
        for name, module in self.normal_encoder.features._modules.items():
            normal_f = module(normal_f)
            if name in self.layer_name_mapping:
                normal_feature[name] = normal_f

        emissive_feature = {}
        emissive_f = x['emissive']
        for name, module in self.emissive_encoder.features._modules.items():
            emissive_f = module(emissive_f)
            if name in self.layer_name_mapping:
                emissive_feature[name] = emissive_f

        eevee_feature = {}
        eevee_f = x['eevee']
        for name, module in self.eevee_encoder.features._modules.items():
            eevee_f = module(eevee_f)
            if name in self.layer_name_mapping:
                eevee_feature[name] = eevee_f


        return {'metalness': metalness_feature, 'roughness': roughness_feature, 'depth': depth_feature, 'albedo': albedo_feature, 'normal': normal_feature, 'emissive': emissive_feature, 'eevee': eevee_feature}



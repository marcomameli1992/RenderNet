import torch
from torch import nn as NN
from torchvision.models import efficientnet_b7
import torchvision.models as models

class Encoder(NN.Module):

    def __init__(self, use_pretrained=True, use_all=True, use_albedo=False, use_depth=False, use_emissive=False, use_metalness=False, use_normal=False, use_roughness=False, use_position=False):
        super(Encoder, self).__init__()
        if use_all:
            self.metalness_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.roughness_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.depth_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.albedo_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.normal_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.emissive_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
            self.position_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        self.eevee_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_albedo:
            self.albedo_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_depth:
            self.depth_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_emissive:
            self.emissive_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_metalness:
            self.metalness_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_normal:
            self.normal_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_roughness:
            self.roughness_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)
        if use_position:
            self.position_encoder = efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_FEATURES)

        self.use_all = use_all
        self.use_albedo = use_albedo
        self.use_depth = use_depth
        self.use_emissive = use_emissive
        self.use_metalness = use_metalness
        self.use_normal = use_normal
        self.use_roughness = use_roughness
        self.use_position = use_position

        self.layer_name_mapping = ['1', '2', '3', '4', '5', '6', '7']

    def forward(self, x):
        if self.use_all:
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

            position_feature = {}
            position_f = x['position']
            for name, module in self.position_encoder.features._modules.items():
                position_f = module(position_f)
                if name in self.layer_name_mapping:
                    position_feature[name] = position_f

        eevee_feature = {}
        eevee_f = x['eevee']
        for name, module in self.eevee_encoder.features._modules.items():
            eevee_f = module(eevee_f)
            if name in self.layer_name_mapping:
                eevee_feature[name] = eevee_f

        if self.use_albedo:
            albedo_feature = {}
            albedo_f = x['albedo']
            for name, module in self.albedo_encoder.features._modules.items():
                albedo_f = module(albedo_f)
                if name in self.layer_name_mapping:
                    albedo_feature[name] = albedo_f

        if self.use_depth:
            depth_feature = {}
            depth_f = x['depth']
            for name, module in self.depth_encoder.features._modules.items():
                depth_f = module(depth_f)
                if name in self.layer_name_mapping:
                    depth_feature[name] = depth_f

        if self.use_emissive:
            emissive_feature = {}
            emissive_f = x['emissive']
            for name, module in self.emissive_encoder.features._modules.items():
                emissive_f = module(emissive_f)
                if name in self.layer_name_mapping:
                    emissive_feature[name] = emissive_f

        if self.use_metalness:
            metalness_feature = {}
            metalness_f = x['metalness']
            for name, module in self.metalness_encoder.features._modules.items():
                metalness_f = module(metalness_f)
                if name in self.layer_name_mapping:
                    metalness_feature[name] = metalness_f

        if self.use_normal:
            normal_feature = {}
            normal_f = x['normal']
            for name, module in self.normal_encoder.features._modules.items():
                normal_f = module(normal_f)
                if name in self.layer_name_mapping:
                    normal_feature[name] = normal_f

        if self.use_roughness:
            roughness_feature = {}
            roughness_f = x['roughness']
            for name, module in self.roughness_encoder.features._modules.items():
                roughness_f = module(roughness_f)
                if name in self.layer_name_mapping:
                    roughness_feature[name] = roughness_f

        if self.use_position:
            position_feature = {}
            position_f = x['position']
            for name, module in self.position_encoder.features._modules.items():
                position_f = module(position_f)
                if name in self.layer_name_mapping:
                    position_feature[name] = position_f

        if self.use_all:
            return {
                'metalness': metalness_feature,
                'roughness': roughness_feature,
                'depth': depth_feature,
                'albedo': albedo_feature,
                'normal': normal_feature,
                'emissive': emissive_feature,
                'position': position_feature,
                'eevee': eevee_feature
            }

        return_dict = {'eevee': eevee_feature}
        if self.use_albedo:
            return_dict['albedo'] = albedo_feature
        if self.use_depth:
            return_dict['depth'] = depth_feature
        if self.use_emissive:
            return_dict['emissive'] = emissive_feature
        if self.use_metalness:
            return_dict['metalness'] = metalness_feature
        if self.use_normal:
            return_dict['normal'] = normal_feature
        if self.use_roughness:
            return_dict['roughness'] = roughness_feature
        if self.use_position:
            return_dict['position'] = position_feature

        return return_dict
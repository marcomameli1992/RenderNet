import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage.io import imread
from kornia.color import rgb_to_hsv, hsv_to_rgb


class RenderDataset(Dataset):

    def __init__(self, files, image_dir, transform=None, get_all=False, get_albedo=False, get_depth=False, get_position=False, get_normal=False, get_metalness=False, get_roughness=False, get_emissive=False):
        super(RenderDataset, self).__init__()
        self.file_list = pd.read_csv(files, sep=",", header=0)
        self.image_dir = image_dir
        self.transform = transform
        self.get_all = get_all
        self.get_albedo = get_albedo
        self.get_depth = get_depth
        self.get_position = get_position
        self.get_normal = get_normal
        self.get_metalness = get_metalness
        self.get_roughness = get_roughness
        self.get_emissive = get_emissive


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):

        image_row = self.file_list.iloc[item]

        eevee_image = imread(os.path.join(self.image_dir, image_row['eevee']))[:, :, :3]
        if self.get_all:
            albedo_image = imread(os.path.join(self.image_dir, image_row['albedo']))[:, :, :3]
            depth_image = imread(os.path.join(self.image_dir, image_row['depth']))
            depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=2)
            position_image = imread(os.path.join(self.image_dir, image_row['position']))[:, :, :3]
            normal_image = imread(os.path.join(self.image_dir, image_row['normal']))[:, :, :3]
            metalness_image = imread(os.path.join(self.image_dir, image_row['metalness']))
            metalness_image = np.repeat(metalness_image[:, :, np.newaxis], 3, axis=2)
            roughness_image = imread(os.path.join(self.image_dir, image_row['roughness']))
            roughness_image = np.repeat(roughness_image[:, :, np.newaxis], 3, axis=2)
            emissive_image = imread(os.path.join(self.image_dir, image_row['emissive']))[:, :, :3]

        if self.get_normal:
            normal_image = imread(os.path.join(self.image_dir, image_row['normal']))[:, :, :3]
        if self.get_albedo:
            albedo_image = imread(os.path.join(self.image_dir, image_row['albedo']))[:, :, :3]
        if self.get_depth:
            depth_image = imread(os.path.join(self.image_dir, image_row['depth']))
            depth_image = np.repeat(depth_image[:, :, np.newaxis], 3, axis=2)
        if self.get_position:
            position_image = imread(os.path.join(self.image_dir, image_row['position']))[:, :, :3]
        if self.get_metalness:
            metalness_image = imread(os.path.join(self.image_dir, image_row['metalness']))
            metalness_image = np.repeat(metalness_image[:, :, np.newaxis], 3, axis=2)
        if self.get_roughness:
            roughness_image = imread(os.path.join(self.image_dir, image_row['roughness']))
            roughness_image = np.repeat(roughness_image[:, :, np.newaxis], 3, axis=2)
        if self.get_emissive:
            emissive_image = imread(os.path.join(self.image_dir, image_row['emissive']))[:, :, :3]

        cycles_image = imread(os.path.join(self.image_dir, image_row['cycles']))[:, :, :3]

        totensor = ToTensor()

        # To tensor convertions
        # eevee_image = rgb_to_hsv(totensor(eevee_image))
        # albedo_image = rgb_to_hsv(totensor(albedo_image))
        # depth_image = totensor(depth_image)
        # position_image = rgb_to_hsv(totensor(position_image))
        # normal_image = rgb_to_hsv(totensor(normal_image))
        # metalness_image = totensor(metalness_image)
        # roughness_image = totensor(roughness_image)
        # emissive_image = rgb_to_hsv(totensor(emissive_image))
        # cycles_image = totensor(cycles_image)
        eevee_image = totensor(eevee_image)
        if self.get_all:
            albedo_image = totensor(albedo_image)
            depth_image = totensor(depth_image)
            position_image = totensor(position_image)
            normal_image = totensor(normal_image)
            metalness_image = totensor(metalness_image)
            roughness_image = totensor(roughness_image)
            emissive_image = totensor(emissive_image)
        if self.get_albedo:
            albedo_image = totensor(albedo_image)
        if self.get_depth:
            depth_image = totensor(depth_image)
        if self.get_position:
            position_image = totensor(position_image)
        if self.get_normal:
            normal_image = totensor(normal_image)
        if self.get_metalness:
            metalness_image = totensor(metalness_image)
        if self.get_roughness:
            roughness_image = totensor(roughness_image)
        if self.get_emissive:
            emissive_image = totensor(emissive_image)
        cycles_image = totensor(cycles_image)

        # applico le trasformazioni
        if self.transform:
            # print("Faccio resize")

            eevee_image = self.transform(eevee_image)
            if self.get_all:
                albedo_image = self.transform(albedo_image)
                depth_image = self.transform(depth_image)
                position_image = self.transform(position_image)
                normal_image = self.transform(normal_image)
                metalness_image = self.transform(metalness_image)
                roughness_image = self.transform(roughness_image)
                emissive_image = self.transform(emissive_image)
            if self.get_albedo:
                albedo_image = self.transform(albedo_image)
            if self.get_depth:
                depth_image = self.transform(depth_image)
            if self.get_position:
                position_image = self.transform(position_image)
            if self.get_normal:
                normal_image = self.transform(normal_image)
            if self.get_metalness:
                metalness_image = self.transform(metalness_image)
            if self.get_roughness:
                roughness_image = self.transform(roughness_image)
            if self.get_emissive:
                emissive_image = self.transform(emissive_image)
            cycles_image = self.transform(cycles_image)

        if self.get_all:
            return {'eevee': eevee_image, 'albedo': albedo_image, 'depth': depth_image, 'position': position_image, 'normal': normal_image, 'metalness': metalness_image, 'roughness': roughness_image, 'emissive': emissive_image, 'cycles': cycles_image}
        return_dict = {'eevee': eevee_image, 'cycles': cycles_image}
        if self.get_albedo:
            return_dict['albedo'] = albedo_image
            assert torch.isnan(albedo_image).sum() == 0
        if self.get_depth:
            return_dict['depth'] = depth_image
            assert torch.isnan(depth_image).sum() == 0
        if self.get_position:
            return_dict['position'] = position_image
            assert torch.isnan(position_image).sum() == 0
        if self.get_normal:
            return_dict['normal'] = normal_image
            assert torch.isnan(normal_image).sum() == 0
        if self.get_metalness:
            return_dict['metalness'] = metalness_image
            assert torch.isnan(metalness_image).sum() == 0
        if self.get_roughness:
            return_dict['roughness'] = roughness_image
            assert torch.isnan(roughness_image).sum() == 0
        if self.get_emissive:
            return_dict['emissive'] = emissive_image
            assert torch.isnan(emissive_image).sum() == 0
        if self.get_position:
            return_dict['position'] = position_image
            assert torch.isnan(position_image).sum() == 0

        assert torch.isnan(eevee_image).sum() == 0
        assert torch.isnan(cycles_image).sum() == 0

        return return_dict

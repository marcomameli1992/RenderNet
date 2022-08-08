import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import pandas as pd
import os
from skimage.io import imread
from kornia.color import rgb_to_hsv, hsv_to_rgb


class RenderDataset(Dataset):

    def __init__(self, files, image_dir, transform=None):
        super(RenderDataset, self).__init__()
        self.file_list = pd.read_csv(files, sep=",", header=0)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):

        image_row = self.file_list.iloc[item]

        eevee_image = imread(os.path.join(self.image_dir, image_row['eevee']))[:, :, :3]
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
        cycles_image = imread(os.path.join(self.image_dir, image_row['cycles']))[:, :, :3]

        totensor = ToTensor()

        # To tensor convertions
        eevee_image = rgb_to_hsv(totensor(eevee_image))
        albedo_image = rgb_to_hsv(totensor(albedo_image))
        depth_image = totensor(depth_image)
        position_image = rgb_to_hsv(totensor(position_image))
        normal_image = rgb_to_hsv(totensor(normal_image))
        metalness_image = totensor(metalness_image)
        roughness_image = totensor(roughness_image)
        emissive_image = rgb_to_hsv(totensor(emissive_image))
        cycles_image = totensor(cycles_image)

        # applico le trasformazioni
        if self.transform:
            # print("Faccio resize")

            eevee_image = self.transform(eevee_image)
            albedo_image = self.transform(albedo_image)
            depth_image = self.transform(depth_image)
            position_image = self.transform(position_image)
            normal_image = self.transform(normal_image)
            metalness_image = self.transform(metalness_image)
            roughness_image = self.transform(roughness_image)
            emissive_image = self.transform(emissive_image)
            cycles_image = self.transform(cycles_image)

        return {'eevee': eevee_image, 'albedo': albedo_image, 'depth': depth_image, 'position': position_image, 'normal': normal_image, 'metalness': metalness_image, 'roughness': roughness_image, 'emissive': emissive_image, 'cycles': cycles_image}
import torch
from torch import nn as NN
from model.generator.Generator import Generator
from dataset.RenderDataset import RenderDataset
from model.discriminator.Discriminator import PerceptualDiscriminator
from torch.utils.data import DataLoader

from kornia.color import rgb_to_hsv, hsv_to_rgb

from torchvision.transforms import Resize

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='RenderNet training script')

parser.add_argument('--data', type=str, default=None, metavar='D',)
parser.add_argument('--image_folder', type=str, default=None, metavar='F',)
parser.add_argument('--epochs', type=int, default=10, metavar='E',)
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',)
parser.add_argument('--gan_loss', type=str, default='mse', metavar='GL',)
parser.add_argument('--batch_size', type=int, default=1, help='the batch size')

args = parser.parse_args()

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Model construction
generator = Generator(4480, 3) ##
discriminator = PerceptualDiscriminator()

generator.to(device)
discriminator.to(device)

#%% dataset opening
transform = Resize((224, 224))
dataset = RenderDataset(args.data, args.image_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

## Loss definition
if args.gan_loss == 'mse':
    gan_loss = NN.MSELoss()
elif args.gan_loss == 'bce':
    gan_loss = NN.BCEWithLogitsLoss()

discriminator_loss = NN.L1Loss()

## Optimizator
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

generator.train()
discriminator.train()

for epoch in range(args.epochs):
    with tqdm(dataloader, unit='batch') as tbatch:
        for data in tbatch:

            for key in data.keys():
                data[key] = data[key].to(device)


            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            ## Discriminator
            real_discriminator = discriminator(data['cycles'])

            ## Generator
            fake_generated = generator(data)

            fake_discriminator = discriminator(hsv_to_rgb(fake_generated))

            ## Loss
            discriminator_loss_1 = discriminator_loss(real_discriminator.d1, fake_discriminator.d1)
            discriminator_loss_2 = discriminator_loss(real_discriminator.d1, fake_discriminator.d1)
            discriminator_loss_3 = discriminator_loss(real_discriminator.d1, fake_discriminator.d1)
            discriminator_loss_4 = discriminator_loss(real_discriminator.d1, fake_discriminator.d1)

            discriminator_loss = (0.25 * discriminator_loss_1) + (0.25 * discriminator_loss_2) + (0.25 * discriminator_loss_3) + (0.25 * discriminator_loss_4)

            discriminator_loss.backward()

            fake_discriminator = discriminator(hsv_to_rgb(fake_generated))

            generator_loss = (0.25 * (-torch.mean(fake_discriminator.d1))) + (0.25 * (-torch.mean(fake_discriminator.d2))) + (0.25 * (-torch.mean(fake_discriminator.d3))) + (0.25 * (-torch.mean(fake_discriminator.d4)))
            generator_distance = gan_loss(rgb_to_hsv(data['cycles']), fake_generated)

            generator_loss = generator_loss + generator_distance

            generator_loss.backward()

            generator_optimizer.step()
            discriminator_optimizer.step()
            


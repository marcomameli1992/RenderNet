import os
from glob import glob
import torch
from torch import nn as NN
from model.generator.Generator import Generator
from dataset.RenderDataset import RenderDataset
from model.discriminator.Discriminator import PerceptualDiscriminator
from torch.utils.data import DataLoader

from kornia.color import rgb_to_hsv, hsv_to_rgb

from torchvision.transforms import Resize, ToPILImage

from tqdm import tqdm

import argparse

import neptune.new as neptune

parser = argparse.ArgumentParser(description='RenderNet training script')

parser.add_argument('--data', type=str, default=None, metavar='D',)
parser.add_argument('--image_folder', type=str, default=None, metavar='F',)
parser.add_argument('--epochs', type=int, default=10, metavar='E',)
parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',)
parser.add_argument('--gan_loss', type=str, default='mse', metavar='GL',)
parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
parser.add_argument('--save_path', type=str, default=None, metavar='SP')
parser.add_argument('--continue_train', action='store_true')

args = parser.parse_args()

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## configure neptune
run = neptune.init(
    project="marcomameli1992/RenderNet",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkZWJkNDEyYS01NjI0LTRjMDAtODI5Yi0wMzI4NWU5NDc0ZmMifQ==",
)  # your credentials

if args.save_path == None:
    save_path = './checkpoints/'
else:
    save_path = args.save_path

os.makedirs(save_path, exist_ok=True)

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

params = {"learning_rate": args.lr, "discriminator_optimizer": "RMSProp", "generator_optimizer": "RMSProp", "batch_size": args.batch_size, "epochs": args.epochs, "gan_loss": args.gan_loss}
run["parameters"] = params

image_transform = ToPILImage()

if args.continue_train and (len(os.listdir(save_path)) > 0):
    print('Continue from checkpoint')
    list_of_checkpoints = glob(save_path + '/*.pth')
    latest_checkpoint = max(list_of_checkpoints, key=os.path.getctime)
    print('Loading checkpoint: {}'.format(latest_checkpoint))
    checkpoint = torch.load(latest_checkpoint)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
    s_epoch = checkpoint['epoch']
    print('Loaded checkpoint at epoch: {}'.format(s_epoch))
else:
    print('No checkpoint found')
    s_epoch = 0

for epoch in range(s_epoch, args.epochs):
    with tqdm(dataloader, unit='batch') as tbatch:
        for i, data in enumerate(tbatch):

            for key in data.keys():
                data[key] = data[key].to(device)


            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            ## Discriminator
            real_discriminator = discriminator(data['cycles'])

            ## Generator
            fake_generated = generator(data)

            fake_gen_noback = fake_generated.clone()
            fake_gen_noback = fake_gen_noback.detach()#hsv_to_rgb(fake_gen_noback.detach())
            fake_discriminator = discriminator(fake_gen_noback)

            for n in range(fake_generated.shape[0]):
                fake_pillow = image_transform(fake_generated[n].cpu())
                real_pillow = image_transform(data['cycles'][n].cpu())

                fake_pillow.save(os.path.join(save_path, 'images', 'epoch_{}'.format(epoch),'fake_{}.png'.format(n)))
                real_pillow.save('real_{}_{}.png'.format(epoch, n))

                #run["fake_generated_epoch_" + str(epoch) + "_batch_" + str(i)].log(fake_pillow)
                #run["real_image_epoch_" + str(epoch) + "_batch_" + str(i)].log(real_pillow)


            ## Discriminator Loss
            discriminator_loss_1 = -(torch.mean(real_discriminator.d1) - torch.mean(fake_discriminator.d1))
            discriminator_loss_2 = -(torch.mean(real_discriminator.d2) - torch.mean(fake_discriminator.d2))
            discriminator_loss_3 = -(torch.mean(real_discriminator.d3) - torch.mean(fake_discriminator.d3))
            discriminator_loss_4 = -(torch.mean(real_discriminator.d4) - torch.mean(fake_discriminator.d4))

            discriminator_loss = (0.25 * discriminator_loss_1) + (0.25 * discriminator_loss_2) + (0.25 * discriminator_loss_3) + (0.25 * discriminator_loss_4)
            discriminator_loss = (0.5 * discriminator_loss) + (0.5 * (gan_loss(real_discriminator.d1, fake_discriminator.d1) + gan_loss(real_discriminator.d2, fake_discriminator.d2) + gan_loss(real_discriminator.d3, fake_discriminator.d3) + gan_loss(real_discriminator.d4, fake_discriminator.d4)))

            run["train/discriminator_loss"].log(discriminator_loss)

            discriminator_loss.backward()
            discriminator_optimizer.step()

            ## Generator Loss
            discriminator.requires_grad_(False)
            generator_loss = (0.25 * (-torch.mean(fake_discriminator.d1.detach()))) + (0.25 * (-torch.mean(fake_discriminator.d2.detach()))) + (0.25 * (-torch.mean(fake_discriminator.d3.detach()))) + (0.25 * (-torch.mean(fake_discriminator.d4.detach())))
            generator_distance = gan_loss(rgb_to_hsv(data['cycles']), fake_generated)

            generator_loss = generator_loss + generator_distance

            run["train/generator_loss"].log(generator_loss)

            generator_loss.backward()
            generator_optimizer.step()
            discriminator.requires_grad_(True)

            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'discriminator_loss': discriminator_loss,
                'generator_loss': generator_loss,
            }, os.path.join(os.path.join(save_path, 'state'), 'checkpoint_' + str(epoch) + '.pth'))


run.stop()
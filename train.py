import os
from glob import glob
import torch
from torch import nn as NN
from model.generator.Generator import Generator
from dataset.RenderDataset import RenderDataset
from model.perceptual.Discriminator import PerceptualDiscriminator
from torch.utils.data import DataLoader

from torchmetrics.functional import structural_similarity_index_measure, multiscale_structural_similarity_index_measure, \
    universal_image_quality_index, error_relative_global_dimensionless_synthesis

from torchvision.transforms import Resize, ToPILImage

from tqdm import tqdm

import argparse

import neptune.new as neptune

parser = argparse.ArgumentParser(description='RenderNet training script')

parser.add_argument('--data', type=str, default=None, metavar='D',)
parser.add_argument('--image_folder', type=str, default=None, metavar='F',)
parser.add_argument('--epochs', type=int, default=10, metavar='E',)
parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',)
parser.add_argument('--gan_loss', type=str, default='mse', metavar='GL',)
parser.add_argument('--batch_size', type=int, default=1, help='the batch size')
parser.add_argument('--save_path', type=str, default=None, metavar='SP')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--use_all', action='store_true')
parser.add_argument('--use_albedo', action='store_true')
parser.add_argument('--use_normal', action='store_true')
parser.add_argument('--use_depth', action='store_true')
parser.add_argument('--use_emissive', action='store_true')
parser.add_argument('--use_metalness', action='store_true')
parser.add_argument('--use_roughness', action='store_true')
parser.add_argument('--use_position', action='store_true')
parser.add_argument('--save_from', type=int, default=250, help='the epoch to start saving')

args = parser.parse_args()

#%%
if torch.has_mps:
    print('Using MPS')
    device = torch.device('mps')
else:
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

#%%
use_all = False
use_albedo = False
use_normal = False
use_depth = False
use_emissive = False
use_metalness = False
use_roughness = False
use_position = False

multiplier = 1

if args.use_all:
    use_all = True
    multiplier += 7
if args.use_albedo:
    use_albedo = True
    multiplier += 1
if args.use_normal:
    use_normal = True
    multiplier += 1
if args.use_depth:
    use_depth = True
    multiplier += 1
if args.use_emissive:
    use_emissive = True
    multiplier += 1
if args.use_metalness:
    use_metalness = True
    multiplier += 1
if args.use_roughness:
    use_roughness = True
    multiplier += 1
if args.use_position:
    use_position = True
    multiplier += 1

decoder_input_channels = 640 * multiplier


#%% Model construction
generator = Generator(decoder_input_channels, 3, multiplier=multiplier, use_all=use_all, use_albedo=use_albedo, use_depth=use_depth, use_emissive=use_emissive, use_metalness=use_metalness, use_normal=use_normal, use_roughness=use_roughness, use_position=use_position) ##
discriminator = PerceptualDiscriminator()

generator.to(device)
discriminator.to(device)

#%% dataset opening
transform = Resize((224, 224))
dataset = RenderDataset(args.data, args.image_folder, transform=transform, get_all=use_all, get_albedo=use_albedo, get_depth=use_depth, get_emissive=use_emissive, get_metalness=use_metalness, get_normal=use_normal, get_roughness=use_roughness, get_position=use_position)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

## Loss definition
if args.gan_loss == 'mse':
    gan_loss = NN.MSELoss()
elif args.gan_loss == 'bce':
    gan_loss = NN.L1Loss()

discriminator_loss = NN.SmoothL1Loss()
g_loss = NN.SmoothL1Loss()

similarity_loss1 = structural_similarity_index_measure
similarity_loss2 = multiscale_structural_similarity_index_measure
similarity_loss3 = universal_image_quality_index

## Optimizator
generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
discriminator_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

generator.train()
discriminator.train()

params = {"learning_rate": args.lr, "discriminator_optimizer": "RMSProp", "generator_optimizer": "RMSProp", "batch_size": args.batch_size, "epochs": args.epochs, "gan_loss": args.gan_loss}
run["parameters"] = params
run["data"] = {"use_all": use_all, "use_albedo": use_albedo, "use_depth": use_depth, "use_emissive": use_emissive, "use_metalness": use_metalness, "use_normal": use_normal, "use_roughness": use_roughness, "use_position": use_position}

image_transform = ToPILImage()

if args.continue_train and (len(os.listdir(save_path)) > 0):
    print('Continue from checkpoint')
    list_of_checkpoints = glob(os.path.join(save_path, 'state') + '/*.pth')
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

epoch_bar = tqdm(total=args.epochs, initial=s_epoch, desc='Epoch', position=0, unit='epoch')
for epoch in range(s_epoch, args.epochs):
    with tqdm(dataloader, unit='batch', desc='Batch', position=1) as tbatch:
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


            ## Discriminator Loss
            discriminator_loss_1 = -(torch.mean(real_discriminator.d1) - torch.mean(fake_discriminator.d1))
            discriminator_loss_2 = -(torch.mean(real_discriminator.d2) - torch.mean(fake_discriminator.d2))
            discriminator_loss_3 = -(torch.mean(real_discriminator.d3) - torch.mean(fake_discriminator.d3))
            discriminator_loss_4 = -(torch.mean(real_discriminator.d4) - torch.mean(fake_discriminator.d4))

            discriminator_loss = (0.25 * discriminator_loss_1) + (0.25 * discriminator_loss_2) + \
                                 (0.25 * discriminator_loss_3) + (0.25 * discriminator_loss_4)
            discriminator_loss = (0.5 * discriminator_loss) + \
                                 (0.5 * (g_loss(real_discriminator.d1, fake_discriminator.d1) +
                                          g_loss(real_discriminator.d2, fake_discriminator.d2) +
                                          g_loss(real_discriminator.d3, fake_discriminator.d3) +
                                          g_loss(real_discriminator.d4, fake_discriminator.d4)))

            run["train/discriminator_loss"].log(10 if torch.isnan(discriminator_loss) else discriminator_loss.item())

            discriminator_loss.backward()
            discriminator_optimizer.step()

            ## Generator Loss
            discriminator.requires_grad_(False)
            generator_loss = (0.25 * (-torch.mean(fake_discriminator.d1.detach())))\
                             + (0.25 * (-torch.mean(fake_discriminator.d2.detach()))) \
                             + (0.25 * (-torch.mean(fake_discriminator.d3.detach()))) \
                             + (0.25 * (-torch.mean(fake_discriminator.d4.detach())))
            generator_distance = gan_loss(data['cycles'], fake_generated)

            s_loss1 = similarity_loss1(data['cycles'], fake_generated)
            s_loss2 = similarity_loss2(data['cycles'], fake_generated, normalize='relu')
            s_loss3 = similarity_loss3(data['cycles'], fake_generated)

            generator_loss = (0.2 * generator_loss) + (0.2 * generator_distance) + (0.2 * (1/s_loss1)) + (0.2 * (1/s_loss2)) + (0.2 * (1/s_loss3))

            run["train/generator_loss"].log(10 if torch.isnan(generator_loss) else generator_loss)
            run["train/generator_distance"].log(10 if torch.isnan(generator_distance) else generator_distance)
            run["train/SSIM"].log(-1 if torch.isnan(s_loss1) else s_loss1)
            run["train/MSSSIM"].log(-1 if torch.isnan(s_loss2) else s_loss2)
            run["train/UIQI"].log(-1 if torch.isnan(s_loss3) else s_loss3)

            generator_loss.backward()
            generator_optimizer.step()
            discriminator.requires_grad_(True)

            os.makedirs(os.path.join(save_path, 'state'), exist_ok=True)

    if epoch % 50 == 0 or (epoch + 1) % 50 == 0:
        for n in range(fake_generated.shape[0]):
            fake_pillow = image_transform(fake_generated[n].cpu())
            real_pillow = image_transform(data['cycles'][n].cpu())

            os.makedirs(os.path.join(save_path, 'images', 'epoch_{}'.format(epoch)), exist_ok=True)

            fake_pillow.save(os.path.join(save_path, 'images', 'epoch_{}'.format(epoch), 'fake_{}.png'.format(n)))
            real_pillow.save(os.path.join(save_path, 'images', 'epoch_{}'.format(epoch), 'real_{}.png'.format(n)))

            run["fake_generated_epoch_" + str(epoch) + "_batch_" + str(i)].log(fake_pillow)
            run["real_image_epoch_" + str(epoch) + "_batch_" + str(i)].log(real_pillow)
    if (epoch > args.save_from and epoch % 25 == 0) or epoch == (args.epochs - 1):
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
        }, os.path.join(os.path.join(save_path, 'state'), 'checkpoint_' + str(epoch) + '.pth'))

    epoch_bar.update(1)


run.stop()
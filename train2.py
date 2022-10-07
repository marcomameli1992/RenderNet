import os
from glob import glob
import torch
from torch import nn as NN
from model.generator.Generator import Generator
from dataset.RenderDataset import RenderDataset
from model.perceptual.PerceptualNetwork2 import PerceptualLoss
from torch.utils.data import DataLoader
from model.discriminator.Discriminator import Discriminator

from torchmetrics.functional import structural_similarity_index_measure, multiscale_structural_similarity_index_measure, \
    universal_image_quality_index

from torchvision.transforms import Resize, ToPILImage

from tqdm import tqdm

import argparse

import neptune.new as neptune

parser = argparse.ArgumentParser(description='RenderNet training script')

parser.add_argument('--data', type=str, default=None, metavar='D',)
parser.add_argument('--image_folder', type=str, default=None, metavar='F',)
parser.add_argument('--epochs', type=int, default=10, metavar='E',)
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',)
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
parser.add_argument('--n_critic', type=int, default=3, help='how much training the discriminator')

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

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(args.batch_size, int(real_data.nelement()/args.batch_size)).contiguous().view(args.batch_size, 3, 32, 32)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if netD.cuda():
        interpolates = interpolates.cuda(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

#%% Model construction
generator = Generator(decoder_input_channels, 3, multiplier=multiplier, use_all=use_all, use_albedo=use_albedo, use_depth=use_depth, use_emissive=use_emissive, use_metalness=use_metalness, use_normal=use_normal, use_roughness=use_roughness, use_position=use_position) ##
discriminator = Discriminator()
perceptual_network = PerceptualLoss(network='vgg16', layers=['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_3'])

generator.to(device)
discriminator.to(device)
perceptual_network.to(device)
perceptual_network.requires_grad_(False)
#%% dataset opening
transform = Resize((224, 224))
dataset = RenderDataset(args.data, args.image_folder, transform=transform, get_all=use_all, get_albedo=use_albedo, get_depth=use_depth, get_emissive=use_emissive, get_metalness=use_metalness, get_normal=use_normal, get_roughness=use_roughness, get_position=use_position)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

## Loss definition
if args.gan_loss == 'mse':
    gan_loss = NN.MSELoss()
elif args.gan_loss == 'bce':
    gan_loss = NN.L1Loss()

discriminator_loss = NN.BCELoss()

similarity_loss1 = structural_similarity_index_measure
similarity_loss2 = multiscale_structural_similarity_index_measure
similarity_loss3 = universal_image_quality_index

## Optimizator
generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.lr * 10, betas=(0.5, 0.999))

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

perceptual_network.eval()

epoch_bar = tqdm(total=args.epochs, initial=s_epoch, desc='Epoch', position=0, unit='epoch')
for epoch in range(s_epoch, args.epochs):
    with tqdm(dataloader, unit='batch', desc='Batch', position=1) as tbatch:
        for i, data in enumerate(tbatch):

            for key in data.keys():
                data[key] = data[key].to(device)

            #-------------------
            # Train Discriminator
            #-------------------
            discriminator_optimizer.zero_grad()

            fake_images = generator(data).detach()

            discriminator_loss = -torch.mean(discriminator(data['cycles'])) + torch.mean(discriminator(fake_images))


            gp = calc_gradient_penalty(discriminator, data['cycles'], fake_images)
            discriminator_loss += 0.2 * gp

            run["train/discriminator_loss"].log(discriminator_loss.item())
            run["train/gradient_penalty"].log(gp.item())

            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            #-------------------
            # Train Generator
            #-------------------

            # Train the generator every n_critic steps
            if i % args.n_critic == 0:
                generator_optimizer.zero_grad()

                fake_images = generator(data)

                # Perceptual loss
                perceptual_loss = perceptual_network(fake_images, data['cycles'])
                s_loss1 = similarity_loss1(data['cycles'], fake_images)
                s_loss2 = similarity_loss2(data['cycles'], fake_images, normalize='relu')
                s_loss3 = similarity_loss3(data['cycles'], fake_images)
                generator_distance = gan_loss(data['cycles'], fake_images)
                g_loss = 0.5 * generator_distance + 0.5 * perceptual_loss

                generator_loss = -torch.mean(discriminator(fake_images)) * 0.5
                generator_loss += g_loss * 0.5

                run["train/generator_loss"].log(generator_loss)
                run["train/generator_distance"].log(generator_distance)
                run["train/SSIM"].log(s_loss1)
                run["train/MSSSIM"].log(s_loss2)
                run["train/UIQI"].log(s_loss3)
                run["train/perceptual_loss"].log(perceptual_loss.item())

                generator_loss.backward()
                generator_optimizer.step()

        os.makedirs(os.path.join(save_path, 'state'), exist_ok=True)

        if epoch % 50 == 0 or (epoch + 1) % 50 == 0:
            for n in range(fake_images.shape[0]):
                fake_pillow = image_transform(fake_images[n].cpu())
                real_pillow = image_transform(data['cycles'][n].cpu())

                os.makedirs(os.path.join(save_path, 'images', 'epoch_{}'.format(epoch)), exist_ok=True)

                fake_pillow.save(
                    os.path.join(save_path, 'images', 'epoch_{}'.format(epoch), 'fake_{}.png'.format(n)))
                real_pillow.save(
                    os.path.join(save_path, 'images', 'epoch_{}'.format(epoch), 'real_{}.png'.format(n)))

                run["fake_generated_epoch_" + str(epoch) + "_batch_" + str(i)].log(fake_pillow)
                run["real_image_epoch_" + str(epoch) + "_batch_" + str(i)].log(real_pillow)
        if (epoch > args.save_from and epoch % 25 == 0) or epoch == (args.epochs - 1):
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'discriminator_loss': perceptual_loss,
                'generator_loss': generator_loss,
            }, os.path.join(os.path.join(save_path, 'state'), 'checkpoint_' + str(epoch) + '.pth'))

        epoch_bar.update(1)

run.stop()
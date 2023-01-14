import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F


def kernel_initializer(w, mean=0., std=0.02):
    return nn.init.normal_(w, mean, std)


def padding(x, p=3):
    return F.pad(x, [p, p, p, p], mode='reflect')


def cycle_loss(real_a, cycle_a, real_b, cycle_b):
    return F.l1_loss(cycle_a, real_a, reduction='mean') + F.l1_loss(cycle_b, real_b, reduction='mean')


class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.ks = kernel_size
        self.s = stride
        self.p = (kernel_size - 1) // 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=False),
            nn.InstanceNorm2d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        self.layer1[0].weight = kernel_initializer(self.layer1[0].weight)

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=kernel_size,
                      stride=stride,
                      bias=False),
            nn.InstanceNorm2d(num_features=dim),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2[0].weight = kernel_initializer(self.layer2[0].weight)

    def forward(self, x):
        y = padding(x, self.p)
        # (bs,256,18,23)
        y = self.layer1(y)
        # (bs,256,16,21)
        y = padding(y, self.p)
        # (bs,256,18,23)
        y = self.layer2(y)
        # (bs,256,16,21)
        out = self.relu(x + y)
        return out


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=dim,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,32,42)

            nn.Conv2d(in_channels=dim,
                      out_channels=4 * dim,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
            nn.InstanceNorm2d(num_features=4 * dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,16,21)

            nn.Conv2d(in_channels=4 * dim,
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False)
            # (bs,1,16,21)
        )

        for i in [0, 2, 5]:
            self.discriminator[i].weight = kernel_initializer(self.discriminator[i].weight)

    def forward(self, x):
        x = self.discriminator(x)
        return x


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=dim,
                      kernel_size=7,
                      stride=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=dim),
            nn.ReLU(inplace=True),
            # (bs,64,64,84)

            nn.Conv2d(in_channels=dim,
                      out_channels=2 * dim,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=2 * dim),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)

            nn.Conv2d(in_channels=2 * dim,
                      out_channels=4 * dim,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.InstanceNorm2d(num_features=4 * dim),
            nn.ReLU(inplace=True)
            # (bs,256,16,21)
        )

        self.ResNet = nn.Sequential(
            *[ResNetBlock(dim=4 * dim,
                          kernel_size=3,
                          stride=1) for _ in range(10)])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4 * dim,
                               out_channels=2 * dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,  # FIXME: what is this? need it?
                               bias=False),
            nn.InstanceNorm2d(num_features=2 * dim),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)

            nn.ConvTranspose2d(in_channels=2 * dim,
                               out_channels=dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
            # (bs,64,64,84)
        )

        self.output = nn.Conv2d(in_channels=dim,
                                out_channels=1,
                                kernel_size=7,
                                stride=1,
                                bias=False)
        # (bs,1,64,84)

        for i in range(3):
            self.encoder[3 * i].weight = \
                kernel_initializer(self.encoder[3 * i].weight)

        for i in range(2):
            self.decoder[3 * i].weight = \
                kernel_initializer(self.decoder[3 * i].weight)

        self.output.weight = kernel_initializer(self.output.weight)

    def forward(self, x):
        x = padding(x)
        x = self.encoder(x)
        # (bs, 256, 16, 21)
        for i in range(10):
            x = self.ResNet[i](x)
        # (bs,256,16,21)
        x = self.decoder(x)
        x = padding(x)
        x = self.output(x)
        return torch.sigmoid(x)


class Classifier(nn.Module):
    def __init__(self, dim=64):
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=dim,
                      kernel_size=(1, 12),
                      stride=(1, 12),
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,64,7)

            nn.Conv2d(in_channels=dim,
                      out_channels=2 * dim,
                      kernel_size=(4, 1),
                      stride=(4, 1),
                      bias=False),
            nn.InstanceNorm2d(num_features=2 * dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,128,16,7)

            nn.Conv2d(in_channels=2 * dim,
                      out_channels=4 * dim,
                      kernel_size=(2, 1),
                      stride=(2, 1),
                      bias=False),
            nn.InstanceNorm2d(num_features=4 * dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,8,7)

            nn.Conv2d(in_channels=4 * dim,
                      out_channels=8 * dim,
                      kernel_size=(8, 1),
                      stride=(8, 1),
                      bias=False),
            nn.InstanceNorm2d(num_features=8 * dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,512,1,7)

            nn.Conv2d(in_channels=8 * dim,
                      out_channels=2,
                      kernel_size=(1, 7),
                      stride=(1, 7),
                      bias=False)
            # (bs,2,1,1)
        )
        self.softmax = nn.Softmax(dim=1)

        for i in [0, 2, 5, 8, 11]:
            self.clf[i].weight = kernel_initializer(self.clf[i].weight)

    def forward(self, x):
        x = self.clf(x)
        # x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)
        return x.squeeze(-1).squeeze(-1)


class Sampler(object):
    def __init__(self, max_length=50):
        self.maxsize = max_length
        self.num = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num < self.maxsize:
            self.images.append(image)
            self.num += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


class CycleGAN(nn.Module):
    def __init__(
            self, g_dim=64, d_dim=64, sigma=0.01, sample_size=50, lamb=10, mode='train', lr=2e-3, wd=1e-2, gamma=1,
            device="cpu"):
        super(CycleGAN, self).__init__()
        assert mode in ['train', 'A2B', 'B2A']

        self.G_A2B = Generator(g_dim)
        self.G_B2A = Generator(g_dim)
        self.D_A = Discriminator(d_dim)
        self.D_B = Discriminator(d_dim)
        self.D_A_all = Discriminator(d_dim)
        self.D_B_all = Discriminator(d_dim)

        self.l2loss = nn.MSELoss(reduction='mean')
        self.sigma = sigma
        self.gamma = gamma
        self.mode = mode
        self.sampler = Sampler(sample_size)
        self.lamb = lamb
        self.device = device

        if self.mode == 'train':
            self.optimizer_GA2B = optim.AdamW(self.G_A2B.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)
            self.optimizer_GB2A = optim.AdamW(self.G_B2A.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)
            self.optimizer_DA = optim.AdamW(self.D_A.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)
            self.optimizer_DB = optim.AdamW(self.D_B.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)
            self.optimizer_DA_all = optim.AdamW(self.D_A_all.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)
            self.optimizer_DB_all = optim.AdamW(self.D_B_all.parameters(), lr=lr, betas=(
                0.5, 0.999), eps=1e-08, weight_decay=wd)

    def set_input(self, real_a, real_b, real_mixed):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        A2B = self.mode == 'A2B'
        self.real_A = (real_a if A2B else real_b).to(self.device)
        self.real_B = (real_b if A2B else real_a).to(self.device)
        self.real_mixed = real_mixed.to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Cycle loss
        self.c_loss = self.lamb * cycle_loss(self.real_A, self.cycle_A, self.real_B, self.cycle_B)

        DA_fake = self.D_A(self.fake_A + self.gauss_noise)
        DB_fake = self.D_B(self.fake_B + self.gauss_noise)

        # Generator losses
        self.G_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake)) + self.c_loss
        self.G_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake)) + self.c_loss

        self.G_A2B_loss.backward(retain_graph=True)
        self.G_B2A_loss.backward(retain_graph=True)

    def backward_D_basic(self, D, D_all, real, sample_fake, real_mixed):
        """Calculate GAN loss for the discriminator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = D(real + self.gauss_noise)
        pred_real_all = D_all(real_mixed + self.gauss_noise)
        # Fake
        pred_fake_sample = D(sample_fake.detach() + self.gauss_noise)
        pred_fake_all = D_all(sample_fake.detach() + self.gauss_noise)

        # Discriminator losses
        D_loss_real = self.l2loss(pred_real, torch.ones_like(pred_real))
        D_loss_fake = self.l2loss(pred_fake_sample, torch.zeros_like(pred_fake_sample))
        D_loss = (D_loss_real + D_loss_fake) * 0.5

        D_loss.backward(retain_graph=True)

        # All losses
        D_all_loss_real = self.l2loss(pred_real_all, torch.ones_like(pred_real_all))
        D_all_loss_fake = self.l2loss(pred_fake_all, torch.zeros_like(pred_fake_all))
        D_all_loss = self.gamma * (D_all_loss_real + D_all_loss_fake) * 0.5

        D_all_loss.backward(retain_graph=True)

        return D_loss, D_all_loss

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.D_A_loss, self.D_A_all_loss = self.backward_D_basic(
            self.D_A, self.D_A_all, self.real_B, self.sample_fake_B, self.real_mixed)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.D_B_loss, self.D_B_all_loss = self.backward_D_basic(
            self.D_B, self.D_B_all, self.real_A, self.sample_fake_A, self.real_mixed)

    def forward(self):
        # blue line
        self.fake_B = self.G_A2B(self.real_A)
        self.cycle_A = self.G_B2A(self.fake_B)

        # red line
        self.fake_A = self.G_B2A(self.real_B)
        self.cycle_B = self.G_A2B(self.fake_A)

        if self.mode == 'train':
            [self.sample_fake_A, self.sample_fake_B] = self.sampler([self.fake_A, self.fake_B])
        elif self.mode == 'A2B':
            return self.fake_B, self.cycle_A
        elif self.mode == 'B2A':
            return self.fake_A, self.cycle_B

    def forward_and_optimize(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # zero the parameter gradients
        self.optimizer_GA2B.zero_grad()
        self.optimizer_GB2A.zero_grad()
        self.optimizer_DA.zero_grad()
        self.optimizer_DB.zero_grad()
        self.optimizer_DA_all.zero_grad()
        self.optimizer_DB_all.zero_grad()

        # forward
        self.forward()      # compute fake and reconstruction

        self.gauss_noise = torch.ones_like(self.real_A)
        self.gauss_noise = torch.abs(kernel_initializer(self.gauss_noise, mean=0, std=self.sigma))

        # G_A and G_B
        self.set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
        self.backward_G()             # calculate gradients for G_A and G_B
        # update G_A and G_B's weights
        self.optimizer_GA2B.step()
        self.optimizer_GB2A.step()

        self.G_loss = self.G_A2B_loss + self.G_B2A_loss - self.c_loss

        # D_A and D_B
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        # update D_A and D_B's weights
        self.optimizer_DA.step()
        self.optimizer_DB.step()
        self.optimizer_DA_all.step()
        self.optimizer_DB_all.step()

        d_loss = self.D_A_loss + self.D_B_loss

        d_all_loss = self.D_A_all_loss + self.D_B_all_loss
        self.D_loss = d_loss + d_all_loss

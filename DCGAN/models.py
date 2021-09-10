"""
DCGAN Pytorch : Model Classes

©️ Sagnik Roy, 2021.

"""


import torch
import torch.nn as nn
from torchsummary import summary


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 padding = 1,
                 stride = 1):

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d( kernel_size = (2,2))
        )

    def forward(self, x):
        return self.conv(x)


class ConvTBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 padding=1,
                 stride=1):

        super().__init__()
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        return self.convt(x)


class Generator(nn.Module):
    def __init__(self,
                 noise_ch_dim = 32,
                 kernel_size = (3, 3),
                 stride = 1,
                 padding = 1):

        super().__init__()
        self.gen_net = nn.Sequential(
            *[ConvTBlock(noise_ch_dim // 2**i,
                         noise_ch_dim // 2**(i+1) if i!=2 else 3,
                         kernel_size, padding, stride) for i in range(3)],
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen_net(x)


class Discriminator(nn.Module):

    def __init__(self, img_channels = 3,
                 latent_dim = 4,
                 kernel_size = (3, 3),
                 stride = 1,
                 padding = 1):
        super().__init__()
        self.disc_net = nn.Sequential(
            *[ConvBlock(latent_dim * 2**i if i!=0 else img_channels,
                        latent_dim * 2**(i+1), kernel_size, padding,
                         stride) for i in range(3)],
            nn.Flatten(),
            nn.Linear(32*32*32, 1024),
            nn.Linear(1024,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc_net(x)


def test():

    gen_noise = torch.rand(1, 32, 32, 32)
    gen = Generator()
    assert gen(gen_noise).shape == (1, 3, 256, 256), "Reconstruct The Generator"

    op = gen(gen_noise)
    disc = Discriminator()
    assert disc(op).shape == (1, 1), "Reconstruct The Discriminator"

def gen_sum():
    if torch.cuda.is_available():
        gen = Generator().cuda()
        print(summary(gen, (32, 32, 32), device="cuda"))
    else:
        gen = Generator()
        print(summary(gen, (32, 32, 32), device="cpu"))

def disc_sum():
    if torch.cuda.is_available():
        disc = Discriminator().cuda()
        print(summary(disc, (3, 256, 256), device="cuda"))
    else:
        disc = Discriminator()
        print(summary(disc, (3, 256, 256), device="cpu"))


if __name__ == '__main__':
    test()
    gen_sum()
    disc_sum()

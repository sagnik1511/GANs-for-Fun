"""
DCGAN Pytorch : Train the custom dataset on given DCGAN

©️Sagnik Roy, 2021.

"""


import argparse
from dataset import *
from models import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import *
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description = "Training Hyperparameters")
parser.add_argument('-d','--data-dir', type = str, required = True, metavar = '', help = 'Directory of the data')
parser.add_argument('-c','--check-dir', type = str,required = True, metavar = '', help = 'Directory to save the model checkpoint')
parser.add_argument('-b','--batch-size', type = int,default = 32, metavar = '', help = 'Batch size during training')
parser.add_argument('-e','--epochs', type = int, default = 50, metavar = '', help = 'Number of iterations')
parser.add_argument('-s','--seed', type = int, default = 0, metavar = '', help = 'seed value to mimic randomness')

args = parser.parse_args()

train_parser_validate(args)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

transform = transform()
train_ds = create_dataloader(root = args.data_dir, H = 256, W = 256, transform = transform, batch_size = args.batch_size)


gen = Generator().cuda()
disc = Discriminator().cuda()

gen.train()
disc.train()

gen_optim = optim.Adam(params = gen.parameters(), lr = 1e-4,weight_decay = 1e-5)
disc_optim = optim.Adam(params = disc.parameters(), lr = 1e-4, weight_decay = 1e-5)
criterion = nn.MSELoss()

for epoch in range(args.epochs):
    print(f"Epoch {epoch + 1} : \n")
    G_LOSS = 0.0
    D_LOSS = 0.0
    for index, (disc_patch, disc_true) in enumerate(train_ds):

        # DISC

        disc_optim.zero_grad()
        disc_patch = disc_patch.cuda()
        gen_noise = torch.rand(args.batch_size,32,32,32).cuda()
        gen_patch = gen(gen_noise).cuda()
        gen_true = torch.zeros(args.batch_size).float().cuda()
        disc_true = disc_true.float().cuda()

        disc_pred = disc(disc_patch).float().cuda()
        gen_pred = disc(gen_patch).float().cuda()

        disc_loss = criterion(disc_pred, disc_true)
        disc_loss += criterion(gen_pred, gen_true)

        D_LOSS += disc_loss.item()

        disc_loss.backward()
        disc_optim.step()

        # DISC

        # GEN

        gen_optim.zero_grad()
        gen_noise = torch.rand(args.batch_size, 32, 32, 32).cuda()
        gen_patch = gen(gen_noise).cuda()
        gen_true = torch.ones(args.batch_size).cuda()

        gen_pred = disc(gen_patch).cuda()
        gen_loss = criterion(gen_pred, gen_true)

        G_LOSS += gen_loss.item()

        gen_loss.backward()
        gen_optim.step()

        # GEN

        if index % 10 == 9:
            print(f"      step {index + 1} :=>  Generator Loss : {gen_loss.item()}   Discriminato Loss : {disc_loss.item()}")



    print(f"Generator Loss : {G_LOSS} , Discriminator Loss : {D_LOSS}\n\n")
    torch.save(gen, f"{args.check_dir}/epoch_{epoch + 1}.pth")


print('Training has been completed.....')

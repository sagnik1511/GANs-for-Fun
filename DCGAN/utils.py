"""
DCGAN Pytorch : Utility functions

©️Sagnik Roy, 2021.

"""


import cv2
import os
import torch.nn as nn



def load_image(path, H, W):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (H,W))

    return image

def train_parser_validate(args):
    assert os.path.isdir(args.data_dir) == True,"Data Directory not found!"
    assert os.path.isdir(args.check_dir) == True,"Checkpoint Directory not found!"
    assert args.epochs > 0,"Invalid number of iterations!"
    assert args.batch_size > 0,"Invalid batch size!"

    print('Train Arguemnets validated.....')

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
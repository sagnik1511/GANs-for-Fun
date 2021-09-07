"""
DCGAN Pytorch : Data Classes

©️Sagnik Roy, 2021.

"""


import cv2
from glob import glob
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import *





def transform() :

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

class GANDataset(Dataset):


    def __init__(self, root, H = 128, W = 128, augment = None, channels = 3):
        super().__init__()

        self.augment = augment
        self.channels = channels
        self.H = H
        self.root = root
        self.W = W


    def __len__(self):
        return len(os.listdir(self.root))


    def __repr__(self):
        return 'GANDataset'


    def __getitem__(self, index):

        image_path = glob( f"{self.root}/*.jpg")[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.H, self.W))

        if self.channels == 1:
            image = image[:, :, 0]

        image = Image.fromarray(image)

        if self.augment is not None:
            image = self.augment(image)

        return (image, 1)

    def validate(self):

        assert os.path.isdir(self.root) == True, "Directory not found!"
        assert self.H > 0 and self.W > 0, "Invalid dimension shapes!"
        assert self.channels == 3 or self.channels == 1, "Invalid channel number!"

        print('Dataset validated.....')


    def visualize(self, num_images_per_side = 4, heading = None):
        total_image = len(os.listdir(self.root))

        if total_image < num_images_per_side**2:
            print('Less image are present...')

        elif num_images_per_side == 0:
            print('Invalid number of images...')

        else:
            fig, ax = plt.subplots(num_images_per_side, num_images_per_side, figsize = (20, 20))

            if heading != None:
                plt.suptitle(f"{heading}", size = (15, 15))

            for index,path in enumerate(glob( f"{self.root}/*.jpg")[:num_images_per_side**2]):
                image = load_image(path, self.H, self.W)
                plt.subplot( num_images_per_side, num_images_per_side, index + 1)
                plt.imshow(image)
                plt.axis('off')
            plt.show()


def create_dataloader(root, H, W, transform = None, channels = 3, batch_size = 32):
    dataset = GANDataset(root = root,
                             H = H,
                             W =W,
                             augment = transform, channels = channels)
    dataset.validate()

    data = DataLoader( dataset = dataset, batch_size = batch_size, shuffle = True)

    return data

import cv2
import os
import glob

import numpy as np
from torch.utils.data import Dataset
import random


class Data_Loader(Dataset):
    """
        DataLoader, includes:
        1、dataLoader, define data loading method imread.(image_path,0)
        2、augment, data augmentation currently includes horizontal/vertical/horizontal/vertical flipping
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))
        # self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))
        self.labels_path = glob.glob(os.path.join(data_path, 'label/*.png'))

    def augment(self, image, flipCode):
        """
            Using cv2. flip for data augmentation,
            The filpCode is
            1. Horizontal flipping,
            0 Vertical Flip,
            -1 Horizontal+Vertical Flip
        """
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        image = cv2.imread(self.imgs_path[index], 0)
        label = cv2.imread(self.labels_path[index], 0)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # label_crack = label_crack.reshape(1, label_crack.shape[0], label_crack.shape[1])

        label = label / 255
        # label_crack = label_crack / 255
        background = np.uint8((np.ones([image.shape[0], image.shape[1]]) - label) > 0)

        label = np.concatenate((label, background), 0)
        # label = np.concatenate((label, label_crack, background), 0)

        # Randomly perform data augmentation, do not process when flipCode is 2
        flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        # return the size of the training set
        return len(self.imgs_path)

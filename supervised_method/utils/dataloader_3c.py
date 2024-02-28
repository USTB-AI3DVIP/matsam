import cv2
import os
import glob

import numpy as np
from torch.utils.data import Dataset
import random
from skimage import morphology


class Data_Loader(Dataset):
    """
        DataLoader for 3 channels, includes:
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
        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        image = cv2.imread(image_path)
        label = cv2.imread(label_path, 0)

        image = np.transpose(image, (2, 0, 1))
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素值为255的改为1

        label = label / 255

        # Randomly perform data augmentation, do not process when flipCode is 2
        flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)
        #         print(image.shape)
        #         print(label.shape)
        return image, label

    def __len__(self):
        # return the size of the training set
        return len(self.imgs_path)

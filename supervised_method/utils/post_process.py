import cv2
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.measure import label
import numpy as np
from PIL import Image
import os


class PostPrecess(object):
    """
        Image post-processing, including:
        1. Rescale to reshape dimensions
        2. Remove_small objects removing small connected domains
        3. Skeletonize
        4. Dilation expansion
        5. Erosion corrosion
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def rescale(self, tar_h, tar_w) -> None:
        """batch resize"""

        save_path = self.dir_path[0:-1] + "_rescale/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for path in os.listdir(self.dir_path):
            print(path)
            image = cv2.imread(self.dir_path + path, 0)

            image = cv2.resize(image, (tar_w, tar_h))
            image = np.int64(image > 0)

            cv2.imwrite(save_path + "rescale__" + path, image * 255)
        print("done!")

    @staticmethod
    def resize(image: np.ndarray, tar_w: int, tar_h: int) -> np.ndarray:
        """
        Scaling images using area based interpolation method
        This method amplifies without loss, while reducing the boundary will make it thicker
        """
        res = cv2.resize(image, (tar_w, tar_h), interpolation=cv2.INTER_AREA)
        return res

    @staticmethod
    def remove_small_objects(image: np.ndarray, min_size=50, connectivity=2) -> np.ndarray:
        """ Image removal of small connected domains """
        return morphology.remove_small_objects(label(image), min_size=min_size, connectivity=connectivity)

    @staticmethod
    def skeletonize(image: np.ndarray, method='lee') -> np.ndarray:
        """ Skeleton """
        return morphology.skeletonize(image, method=method)

    @staticmethod
    def dilation(image: np.ndarray, square=3) -> np.ndarray:
        """ Dilation """
        return morphology.dilation(image, morphology.square(square))

    @staticmethod
    def erosion(image: np.ndarray, square=3) -> np.ndarray:
        """ Erosion """
        return morphology.erosion(image, morphology.square(square))

    @staticmethod
    def remove_small_holes(image: np.ndarray, min_size=10, connectivity=2) -> np.ndarray:
        """ Remove small holes """
        return morphology.remove_small_holes(image, min_size, connectivity)

    @staticmethod
    def threshold(image: np.ndarray, the=240, maxval=255) -> np.ndarray:
        """ Threshold """
        dst, res = cv2.threshold(image, the, maxval, cv2.THRESH_BINARY_INV)
        return res

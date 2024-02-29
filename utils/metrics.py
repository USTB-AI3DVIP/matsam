import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import morphology, measure
from skimage.measure import label
from PIL import Image
import time
from natsort import ns, natsorted
import csv
import math
import gala.evaluate as ev
from typing import Tuple


# 后处理方法
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
        """ Batch Resize Images """

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
        """ remove small objects """
        return morphology.remove_small_objects(label(image), min_size=min_size, connectivity=connectivity)

    @staticmethod
    def skeletonize(image: np.ndarray, method='lee') -> np.ndarray:
        """ skeletonize """
        return morphology.skeletonize(image, method=method)

    @staticmethod
    def dilation(image: np.ndarray, square=3) -> np.ndarray:
        """ dilation """
        return morphology.dilation(image, morphology.square(square))

    @staticmethod
    def erosion(image: np.ndarray, square=3) -> np.ndarray:
        """ erosion """
        return morphology.erosion(image, morphology.square(square))

    @staticmethod
    def remove_small_holes(image: np.ndarray, min_size=10, connectivity=2) -> np.ndarray:
        """ remove small holes """
        return morphology.remove_small_holes(image, min_size, connectivity)

    @staticmethod
    def threshold(image: np.ndarray, the=240, maxval=255) -> np.ndarray:
        """ threshold """
        dst, res = cv2.threshold(image, the, maxval, cv2.THRESH_BINARY_INV)
        return res


def contingency_table(seg, gt, *, ignore_seg=(), ignore_gt=(), norm=True):
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, np.bool)
    data = np.ones(gtr.shape)
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsr()
    if norm:
        cont /= cont.sum()
    return cont


def rand_values(cont_table):
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n) / 2.0;
    b = (sum2 - sum1) / 2
    c = (sum3 - sum1) / 2
    d = (sum1 + n ** 2 - sum2 - sum3) / 2
    return a, b, c, d


def adj_rand_index(x, y=None):
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    nk = a + b + c + d
    return (nk * (a + d) - ((a + b) * (a + c) + (c + d) * (b + d))) / (
            nk ** 2 - ((a + b) * (a + c) + (c + d) * (b + d)))


# calculate grain information
def cal_atom_region_info(img_: np.ndarray):
    res = img_
    np.uint8(img_ > 0)
    label = measure.label(img_, background=1, connectivity=1)
    regions = measure.regionprops(label)

    return regions


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def get_iou(pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Referenced by:
        Long J , Shelhamer E , Darrell T . Fully Convolutional Networks for Semantic Segmentation[J].
        IEEE Transactions on Pattern Analysis & Machine Intelligence, 2014, 39(4):640-651.
        """
        class_num = np.amax(mask) + 1

        temp = 0.0
        for i_cl in range(class_num):
            n_ii = np.count_nonzero(mask[pred == i_cl] == i_cl)
            t_i = np.count_nonzero(mask == i_cl)
            temp += n_ii / (t_i + np.count_nonzero(pred == i_cl) - n_ii)
        value = temp / class_num
        return value

    @staticmethod
    def get_dice(pred: np.ndarray, mask: np.ndarray) -> float:
        """
        Dice score
        From now, it is suited to binary segmentation, where 0 is background and 1 is foreground
        """
        intersection = np.count_nonzero(mask[pred == 1] == 1)
        area_sum = np.count_nonzero(mask == 1) + np.count_nonzero(pred == 1)
        value = 2 * intersection / area_sum
        return value

    @staticmethod
    def get_ari(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0) -> float:
        """
        Adjusted rand index
        Implemented by gala (https://github.com/janelia-flyem/gala.)
        """
        label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
        label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
        value = adj_rand_index(label_pred, label_mask)
        return value

    def get_vi(pred: np.ndarray, mask: np.ndarray, bg_value: int = 0, method: int = 1) -> Tuple:
        """
        Referenced by:
        Marina Meilă (2007), Comparing clusterings—an information based distance,
        Journal of Multivariate Analysis, Volume 98, Issue 5, Pages 873-895, ISSN 0047-259X, DOI:10.1016/j.jmva.2006.11.013.
        :param method: 0: skimage implementation and 1: gala implementation (https://github.com/janelia-flyem/gala.)
        :return Tuple = (VI, merger_error, split_error)
        """
        vi, merger_error, split_error = 0.0, 0.0, 0.0

        label_pred, num_pred = label(pred, connectivity=1, background=bg_value, return_num=True)
        label_mask, num_mask = label(mask, connectivity=1, background=bg_value, return_num=True)
        if method == 1:
            # gala
            merger_error, split_error = ev.split_vi(label_pred, label_mask)
        vi = merger_error + split_error
        if math.isnan(vi):
            return 10, 5, 5
        return merger_error, split_error, vi

    def get_F1(pred: np.ndarray, mask: np.ndarray):  # mask, pred
        # bool type for calculatel
        mask = mask.astype(bool)
        pred = pred.astype(bool)

        # calculate the number of true cases, false positive cases, and false negative cases
        true_positive = np.logical_and(mask, pred).sum()
        false_positive = np.logical_and(~mask, pred).sum()
        false_negative = np.logical_and(mask, ~pred).sum()

        # calculate accuracy and recall
        precision = true_positive / (true_positive + false_positive + 1e-10)
        recall = true_positive / (true_positive + false_negative + 1e-10)

        # calculate F1 metrics
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        return f1_score

    def get_recall(pred: np.ndarray, mask: np.ndarray):
        pred = pred.astype(bool)
        mask = mask.astype(bool)

        true_positive = np.logical_and(mask, pred).sum()
        false_negative = np.logical_and(mask, ~pred).sum()

        recall = true_positive / (true_positive + false_negative + 1e-10)

        return recall

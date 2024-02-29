import cv2
import numpy as np
from .metrics import PostPrecess


class PromptGenerator(object):
    def __init__(self, image, layers=0, scales=3, n_per_side_base=32, method_type=1):
        self.image = image
        self.layers = layers
        self.scales = scales
        self.n_per_side_base = n_per_side_base
        self.method_type = method_type  # 1 Canny pre segmentation, 2 OTSU pre segmentation
        self.points_layers = []  # prompt points layer

    def generate_prompt_points(self):
        center_res = []
        np_center_res = []

        # generate region-aware prompt points using traditional segmentation methods
        if self.method_type == 1:
            dst = cv2.Canny(self.image, 80, 130)
            canny_img = PostPrecess.remove_small_objects(dst, min_size=15)
            canny_img = PostPrecess.dilation(np.uint8(canny_img > 0) * 255, square=5)
            contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        elif self.method_type == 2:
            image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
            _, otsu_image = cv2.threshold(blurred, 50, 225, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(otsu_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        # find contour extraction centroid points based on threshold segmentation
        for n, item1 in enumerate(contours):
            M = cv2.moments(item1)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"] / self.image.shape[0]
                cY = M["m01"] / M["m00"] / self.image.shape[1]
                np_center_res.append([cX, cY])
        center_res.append(np.array(np_center_res))

        # automatically adjust the number of grid points based on regional perception point density
        # use only on images with overly dense prompts, to improve model computational efficiency and reduce redundant operations
        # self.n_per_side_base -= int(self.image.shape[0] * self.image.shape[1] / len(center_res[0]))

        # add grid points
        for i in range(self.layers + 1):
            n_per_side = int(self.n_per_side_base / (self.scales ** i))
            offset = 1 / (2 * n_per_side)
            points_one_side = np.linspace(offset, 1 - offset, n_per_side)
            points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
            points_y = np.tile(points_one_side[:, None], (1, n_per_side))
            points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
            if len(center_res[0]) > 0:  # prevent missing centroid points
                points = np.concatenate((points, np.array(np_center_res)), 0)
            self.points_layers.append(points)

        return self.points_layers

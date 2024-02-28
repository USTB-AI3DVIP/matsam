import os
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import numpy as np
from model.nets.unet import UNet
from utils.crop_process import OverlapTile
from utils.post_process import PostPrecess
from utils.metric import get_ari, get_vi, get_map_2018kdsb
from natsort import ns, natsorted


def predict(dataset_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(1, 2)
    net.to(device=device)
    print('load model successfully.')

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    img_list = natsorted(os.listdir(dataset_path + 'image'), alg=ns.PATH)
    label_list = natsorted(os.listdir(dataset_path + 'label'), alg=ns.PATH)

    total_ARI = []
    total_VI = []
    total_mAP = []

    for item, index in zip(img_list, range(0, len(img_list))):
        img = cv2.imread(dataset_path + 'image/{0}'.format(item), 0)
        img_label = cv2.imread(dataset_path + 'label/{0}'.format(label_list[index]), 0)

        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        torch.cuda.empty_cache()
        pred = net(img_tensor)
        torch.cuda.empty_cache()

        pred_label = np.array(pred.data.cpu()[0])[0]  # 边界
        pred_label[pred_label >= 0.5] = 255
        pred_label[pred_label < 0.5] = 0


if __name__ == '__main__':
    model_path = "/root/data/checkpoints_new/Data_256_1.pth"

    dataset_path = '/root/data/dataset/PI_1/test/'

    predict(dataset_path, model_path)

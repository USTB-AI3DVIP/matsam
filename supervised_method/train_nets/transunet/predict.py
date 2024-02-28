import os
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import numpy as np
from model.nets.segnet import SegNet
from utils.crop_process import OverlapTile
from utils.post_process import PostPrecess
from utils.metric import get_ari, get_vi, get_map_2018kdsb
from natsort import ns, natsorted
from model.nets.transunet.vit_seg_modeling import VisionTransformer
from model.nets.transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def predict(dataset_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 2
    config_vit.n_skip = 3

    config_vit.patches.grid = (int(256 / 16), int(256 / 16))
    net = VisionTransformer(config_vit, img_size=256, num_classes=2)

    net.to(device=device)
    print('load model successfully.')

    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    img_list = natsorted(os.listdir(dataset_path + 'image'), alg=ns.PATH)

    total_ARI = []
    total_VI = []
    total_mAP = []

    for item, index in zip(img_list, range(0, len(img_list))):
        img = cv2.imread(dataset_path + 'image/{0}'.format(item))  # PPIG为-1

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        img = torch.from_numpy(img)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred = net(img)

        pred_label = np.array(pred.data.cpu()[0])[0]  # 边界
        pred_label[pred_label >= 0.5] = 255
        pred_label[pred_label < 0.5] = 0


if __name__ == '__main__':
    model_path = "/root/data/checkpoints_new/Data_256_1.pth"

    dataset_path = '/root/data/dataset/PI_1/test/'

    predict(dataset_path, model_path)

import os
import time

import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from utils.post_process import PostPrecess
from utils.metric import get_ari, get_vi, get_map_2018kdsb, get_F1, get_recall
from natsort import ns, natsorted

dataset_path = '/root/data/dataset/PI_1/test/label/'
pred_img_list = '/root/data1/unet/result/PI_1/'

img_list = natsorted(os.listdir(pred_img_list), alg=ns.PATH)
label_list = natsorted(os.listdir(dataset_path), alg=ns.PATH)

print(len(img_list))
print(len(label_list))

# calculation of overall indicators
total_ARI = []
total_VI = []
total_mAP = []
total_f1 = []
total_recall = []

for item, index in zip(img_list, range(0, len(img_list))):
    pred_label = cv2.imread('{0}{1}'.format(pred_img_list, item), 0)
    img_label = cv2.imread('{0}{1}'.format(dataset_path, label_list[index]), 0)

    img_label = PostPrecess.skeletonize(np.uint8(img_label) > 0) * 255

    pred_label = (np.uint8(pred_label) < 1) * 255

    # evaluate
    ari_score = round(get_ari(pred_label, img_label, 1), 6)
    _, _, vi_score = get_vi(pred_label, img_label, 1)
    vi_score = round(vi_score, 6)
    # map_score = round(get_map_2018kdsb(pred_label, img_label, 1), 6)
    map_score = 0
    f1_score = round(get_F1(pred_label, img_label), 6)
    recall_score = round(get_recall(pred_label, img_label), 6)

    total_ARI.append(ari_score)
    total_VI.append(vi_score)
    total_mAP.append(map_score)
    total_f1.append(f1_score)
    total_recall.append(recall_score)

    print('{0} done, ARI={1}, VI={2}, mAP={3}, f1={4}, recall={5}'
          .format(str(index + 1), str(ari_score), str(vi_score), str(map_score), str(f1_score), str(recall_score)))

print('average: ARI={0}, VI={1}, mAP={2},f1={3}, recall={4}'
      .format(str(sum(total_ARI) / len(total_ARI)), str(sum(total_VI) / len(total_VI)),
              str(sum(total_mAP) / len(total_mAP)), str(sum(total_f1) / len(total_f1)),
              str(sum(total_recall) / len(total_recall))))

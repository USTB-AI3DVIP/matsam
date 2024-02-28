import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import numpy as np
from utils.draw import draw_loss, draw_val_loss
from model.nets.segnet import SegNet
from utils.dataloader_3c import Data_Loader
from torch.utils.data import DataLoader, random_split
# from losses.dice_loss import DiceLoss
from losses.new_loss import WeightMapLoss, BalancedClassWeight
# from evaluate import Evaluate
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import tqdm


def train_net(net, device, train_data, val_data, epochs=20, batch_size=5, lr=0.0001, model_name='segnet_model.pth'):
    """ Train """
    """ 1 - Read and create a dataset loader """
    train_dataset = Data_Loader(train_data)
    val_dataset = Data_Loader(val_data)
    dataloader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_args)

    print(len(train_loader))
    print(len(val_loader))

    """ 2 - Defining optimizers and learning strategies """

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

    criterion = WeightMapLoss()

    """ 3 - Train """
    best_loss = float('inf')
    train_loss_array = []
    val_loss_array = []

    for epoch in range(epochs):
        net.train()
        for image, label in train_loader:

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            pred = net(image)

            bb = BalancedClassWeight(2)
            label_ = label.data.cpu().numpy()[:, 0, :, :]
            weights = bb.get_weight(label_)
            weights_tensor = torch.from_numpy(weights).float().to(device)
            loss = criterion(pred, label, weights_tensor)

            train_loss_array.append(loss.data.cpu())
            print('epoch:{0}  Loss/train:'.format(epoch), loss.item())
            if loss < best_loss:
                best_loss = loss
                # torch.save(net.state_dict(), 'best_model.pth')
                torch.save(net.state_dict(), "loss_{0}".format(str(round(best_loss.item(), 4))) + model_name)
                print('saving best model at epoch {0}'.format(epoch))

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        net.eval()

        for image, label in val_loader:
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                pred = net(image)

                bb = BalancedClassWeight(2)
                label_ = label.data.cpu().numpy()[:, 0, :, :]
                weights = bb.get_weight(label_)
                weights_tensor = torch.from_numpy(weights).float().to(device)
                loss = criterion(pred, label, weights_tensor)

                val_loss_array.append(loss.data.cpu())
                print('epoch:{0}  Loss/val:'.format(epoch), loss.item())

        # val_score = Evaluate.evaluate(net, val_loader, device)
        # scheduler.step(val_score)

        draw_loss(train_loss_array)
        draw_val_loss(val_loss_array)


""" 4 - Save training and validation results """

if __name__ == '__main__':
    """ load data """
    train_data = "/root/data/dataset/PI_1/train/"
    val_data = "/root/data/dataset/PI_1/val/"

    # choose device
    device = torch.device('cuda')

    # load Net
    net = SegNet(2)

    net.to(device=device)

    print("Network initialization completed!")

    # Specify the training set address to start training
    train_net(net, device, train_data, val_data, epochs=20, batch_size=16, lr=0.005, model_name="Data_256_1.pth")

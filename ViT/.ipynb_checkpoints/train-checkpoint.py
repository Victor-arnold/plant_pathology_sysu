# coding: utf-8 
# author --zzj--

import yaml
import argparse
import os
import pandas as pd
import torchvision
import datetime
from sklearn.metrics import accuracy_score
from utils import *
from copy import deepcopy
from typing import Dict
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
import warnings
import cv2
from tqdm import tqdm
from models.SoftTargetCrossEntropy import SoftTargetCrossEntropy
from models.vision_transformer import VisionTransformer
import torch.nn.parallel
import torch.optim as optim
from models.vision_transformer import VisionTransformer
import pdb

def calculate_acc(label, output, batch):
    y_pred = np.array(output.cpu())
    y_label = np.array(label.cpu())
    y_pred = np.where(y_pred > 0, 1, 0)
    correct_sum = 0
    num = y_pred.shape[0]
    for i in range(num):
        count = accuracy_score(y_pred[i],y_label[i]) 
        correct_sum += count
    return correct_sum / num

def train(net, loss, train_dataloader, valid_dataloader, epochs):
    
    lr = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    
    # print('training on:', device)
    # net.to(device)
    #优化器选择

    optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)
    #用来保存每个epoch的Loss和acc以便最后画图
    train_losses = []
    eval_loss_list = []
    eval_acc_list = []
    best_loss = 100
    #训练
    for epoch in range(epochs):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='train'):
            imgs, targets = batch
            imgs, targets = imgs.to(device), targets.to(device)
            output = net(imgs)

            Loss = loss(output.to(torch.float32), targets.to(torch.float32))
            
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
       
        scheduler.step()
        print("epoch: {}, Loss: {}".format(epoch, Loss.item()))
        train_losses.append(Loss.item())

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                eval_loss += Loss
                eval_acc = calculate_acc(targets, output, batch_size)
    
            eval_losses = eval_loss / (len(valid_dataloader))
            if eval_losses < best_loss:
                best_loss = eval_losses
                torch.save(net.state_dict(),'/mnt/zzj_program/plant_project/results/best_loss.pth')
                
            eval_loss_list.append(eval_losses)
            eval_acc_list.append(eval_acc)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的Acc: {}".format(eval_acc))
        
    return train_losses, eval_acc_list
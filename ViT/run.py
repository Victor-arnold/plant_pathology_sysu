# coding: utf-8 
# author --zzj--

import yaml
import argparse
import os
import pandas as pd
import torchvision
import datetime
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
# import timm
# from timm.data.mixup import Mixup
from models.SoftTargetCrossEntropy import SoftTargetCrossEntropy
from models.vision_transformer import VisionTransformer
# from torchtoolbox.transform import Cutout
import torch.nn.parallel
import torch.optim as optim
from models.vision_transformer import VisionTransformer
from train import train

# from datasets.data_utils.data_transform import data_transform
from datasets.read_data import read_data
from datasets.dataset import MyDataset
import torch, gc

gc.collect()
torch.cuda.empty_cache()


data_dict = {'complex': [0,0,0,1,0,0],
 'frog_eye_leaf_spot': [0,0,1,0,0,0],
 'frog_eye_leaf_spot complex':[0,0,1,1,0,0],
 'healthy':[1,0,0,0,0,0],
 'powdery_mildew':[0,0,0,0,1,0],
 'powdery_mildew complex':[0,0,0,1,1,0],
 'rust':[0,0,0,0,0,1],
 'rust complex':[0,0,0,1,0,1],
 'rust frog_eye_leaf_spot':[0,0,1,0,0,1],
 'scab':[0,1,0,0,0,0],
 'scab frog_eye_leaf_spot':[0,1,1,0,0,0],
 'scab frog_eye_leaf_spot complex':[0,1,1,1,0,0]}


data_transform = transforms.Compose([   
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
   
batch_size = 64

data_type = 'train'
train_dataset = MyDataset(data_type, data_dict, data_transform, read_data, None)
train_iter = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=500)

# sampler = DistributedSampler(test_dataset) # 多GPU分布式训练


# data_type = 'val'
# val_iter = DataLoader(MyDataset(data_type, True, 500), batch_size, shuffle=True)    

data_type = 'test'
test_dataset = MyDataset(data_type, data_dict, data_transform, read_data, None)
test_iter = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=6, pin_memory=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer()

# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)

loss = SoftTargetCrossEntropy()
loss1 = nn.BCELoss(reduction="mean")
train_losses, eval_losses = train(model,loss,train_iter,test_iter, 50)

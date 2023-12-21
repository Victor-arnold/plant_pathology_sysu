from PIL import Image
import torch
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from datasets.data_utils.data_transform import data_transform


def read_data(data_type, data_dict, data_transform=None, train_small_batch=None):
    
    data_dir = '/mnt/zzj_program/plant_dataset/'
    csv_fname = os.path.join(data_dir, data_type ,(data_type + '_label.csv'))
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('images')
    images, labels = [], []
    
    if train_small_batch is None:
        num = len(csv_data)
    else:
        num = train_small_batch
    
    for i in tqdm(range(num)):
        img_name = csv_data.index[i]
        target = csv_data.labels[i]
        image_dir = os.path.join(data_dir, data_type  , 'images', img_name)
        img = Image.open(image_dir).convert('RGB')
        if data_transform is not None:            
            img = data_transform(img)   
        images.append(img)
        labels.append(data_dict[target]) 
        
    return images, torch.tensor(labels)
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from datasets.data_utils.data_transform import data_transform
from datasets.read_data import read_data

class MyDataset(Dataset):
    def __init__(self, data_type, data_dict, data_transform, read_data, train_small_batch=None):
        super().__init__()
        self.train_small = train_small_batch
        self.data_dict = data_dict
        self.read_data = read_data
        self.data_transform = data_transform
        self.images, self.labels = read_data(data_type, self.data_dict, self.data_transform, self.train_small)
        print('read ' + str(len(self.images)) + ' ' + (data_type + ' examples'))
        
    def __getitem__(self, index):           
        return (self.images[index], self.labels[index])
    
    def __len__(self):
        return len(self.images)
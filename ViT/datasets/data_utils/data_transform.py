from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

def data_transform(transform):
    data_trans = transforms.Compose([   
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return data_trans
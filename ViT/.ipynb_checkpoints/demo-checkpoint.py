import yaml
import argparse
import os
import pandas as pd
import torchvision
import datetime
from copy import deepcopy
from typing import Dict
import pickle
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

def parse_args():
    
    parser = argparse.ArgumentParser() # 创建一个ArgumentParser对象
    parser.add_argument('--cfg', type=str)
    args = parser.parse_known_args()[0]

    return args

args = parse_args()

cfg_file = args.cfg 
with open(cfg_file) as f:
    cfg = yaml.safe_load(f.read())
    
print(cfg)
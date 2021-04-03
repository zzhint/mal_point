import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import joblib
#from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader

import sklearn.metrics as metrics
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from shutil import copyfile
import joblib
import copy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import pdb





train_data = joblib.load('./the_data_load/train_data.pkl')
train_label = joblib.load('./the_data_load/train_label.pkl')
test_data = joblib.load('./the_data_load/test_data.pkl')
test_label = joblib.load('./the_data_load/test_label.pkl')
def load_data(w):
    if w == 'train':
        return np.array(train_data),np.array(train_label)
    else:
        return np.array(test_data),np.array(test_label)


class ModelNet(Dataset):
    def __init__(self, num_points=300, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        
def load_data_all(args):
    train_loader = DataLoader(ModelNet(partition='train'), num_workers=0,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet(partition='test'), num_workers=0,
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader,test_loader


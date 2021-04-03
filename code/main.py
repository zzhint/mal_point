import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import joblib

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

from train import train
from test import test

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='mal_code_point')
    parser.add_argument('--model', type=str, default='dgcnn_100', metavar='N',
                        choices=['attention', 'dgcnn','lstm_l','dgcnn_100'],
                        help='Model to use, [attention, dgcnn,lstm_l]')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    #parser.add_argument('--gpu',type = int ,default=0)

    parser.add_argument('--output_channels',type = int ,default=17)
    parser.add_argument('--do',type=str,default='train')
    parser.add_argument('--use',type=str,default='yes')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.do == 'train':
        train(args)
    else:
        test(args)

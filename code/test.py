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
from my_model import ATTENTION,DGCNN,LSTM_L
from my_data import load_data_all


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    #gold = torch.LongTensor(gold)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def test(args):
    device = torch.device("cuda")
    if args.model == 'attention':
        model = ATTENTION(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args,device).to(device)
    else:
        model = LSTM_L(args).to(device)
    model.load_state_dict(torch.load('./save_model/model_{}'.format(args.model)))
    model.eval()

    criterion = cal_loss
    
    best_test_acc = 0

    train_loader,test_loader = load_data_all(args)




    ####################
    # Test
    ####################
    test_loss = 0.0
    count = 0.0
    
    test_pred = []
    test_true = []
    for data, label in test_loader:
        data = data.long()
        label = label.long()
        data, label = data.to(device), label.to(device).squeeze()
        
        #data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        loss = criterion(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    outstr = 'loss: %.6f, test acc: %.6f' %(test_loss*1.0/count,test_acc)
    print(outstr)
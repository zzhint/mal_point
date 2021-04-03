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
from my_model import ATTENTION,DGCNN,LSTM_L,DGCNN_100
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


def train(args):
    device = torch.device("cuda")
    if args.model == 'attention':
        model = ATTENTION(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args,device).to(device)
    elif args.model == 'dgcnn_100':
        model = DGCNN_100(args,device).to(device)
    else:
        model = LSTM_L(args).to(device)
    if args.use == 'yes':
        try:
            model.load_state_dict(torch.load('./save_model/model_{}'.format(args.model)))
        except:
            print('no model')
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt,10, eta_min=0.000001)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    
    criterion = cal_loss
    
    best_test_acc = 0

    train_loader,test_loader = load_data_all(args)




    for epoch in range(args.epochs):
       # print('finnal begin')
        
        ####################
        # Train
        ####################
        
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        cnt = 0
        for data, label in train_loader:
            data = data.long()
            label = label.long()
            data, label = data.to(device), label.to(device).squeeze()
            #import pdb;pdb.set_trace()
            if len(label)<=1:
                print('1')
                continue
            #data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            
            loss.backward()
            #print('loss back')
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch,train_loss*1.0/count,metrics.accuracy_score(train_true, train_pred))
        print(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
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
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), './save_model/model_{}'.format(args.model))

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, best acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              best_test_acc)
        print(outstr)
    scheduler.step()

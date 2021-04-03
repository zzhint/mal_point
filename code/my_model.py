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



class ATTENTION(nn.Module):
    def __init__(self,args):
        super(ATTENTION, self).__init__()
        self.batch_size = args.batch_size
        self.embedding = torch.nn.Embedding(6187, 50)
        self.lstm_hid_dim = 512
        #self.hidden_state = self.init_hidden()
        self.d_a = 256
        self.r = 128
        self.lstm = torch.nn.LSTM(50,self.lstm_hid_dim,1,batch_first=True)
        self.linear_first = torch.nn.Linear(self.lstm_hid_dim,self.d_a)
        #self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(self.d_a,self.r)
        #self.linear_second.bias.data.fill_(0)
        self.linear_final = torch.nn.Linear(self.lstm_hid_dim,args.output_channels)
    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
    def init_hidden(self):
        return (Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda(),\
            Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda())
    def forward(self,x):
        hidden_state = self.init_hidden()
        embeddings = self.embedding(x)
        
        outputs,hidden_state = self.lstm(embeddings.view(self.batch_size,300,-1),hidden_state)
        x = F.tanh(self.linear_first(outputs))
        x = self.linear_second(x)
        x = self.softmax(x,1)
        attention = x.transpose(1,2)
        sentence_embeddings = attention@outputs 
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r
        fin = self.linear_final(avg_sentence_embeddings)
        return fin

class LSTM_L(nn.Module):
    def __init__(self,args):
        super(LSTM_L, self).__init__()
        self.batch_size = args.batch_size
        self.embedding = torch.nn.Embedding(6187, 50)
        self.lstm_hid_dim = 512
        self.d_a = 256
        self.r = 128
        self.lstm = torch.nn.LSTM(50,self.lstm_hid_dim,1,batch_first=True)
        self.linear_first = nn.Sequential(torch.nn.Linear(self.lstm_hid_dim,self.d_a),nn.ReLU())
        self.linear_second = nn.Sequential(torch.nn.Linear(self.d_a,self.r),nn.ReLU())
        self.linear_final = torch.nn.Linear(self.r,args.output_channels)
    def forward(self,x):
        embeddings = self.embedding(x)
        outputs,_ = self.lstm(embeddings.view(self.batch_size,300,-1))
        x = self.linear_first(outputs[:,-1,:])
        x = self.linear_second(x)
        x = self.linear_final(x)
        return x 

class DGCNN(nn.Module):
    def __init__(self,args,device):
        super(DGCNN, self).__init__()
        self.k = 20
        self.embedding = torch.nn.Embedding(6187, 50)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256*2)
        self.bn4 = nn.BatchNorm2d(256*2)
        self.bn5 = nn.BatchNorm1d(50)
        self.device = device 
        self.conv1 = nn.Sequential(nn.Conv2d(100, 256, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256*2,256*2, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256*4, 256*2, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(1536, 50, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(50*2, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128, args.output_channels)

    def forward(self, x):
        def knn(x, k):
            inner = -2*torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
            idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
            return idx


        def get_graph_feature(x, k=20, idx=None):
            batch_size = x.size(0)
            num_points = x.size(2)
            x = x.view(batch_size, -1, num_points)
            if idx is None:
                idx = knn(x, k=k)   # (batch_size, num_points, k)
            

            idx_base = torch.arange(0, batch_size,device=self.device).view(-1, 1, 1)*num_points

            idx = idx + idx_base

            idx = idx.view(-1)
        
            _, num_dims, _ = x.size()

            x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
            feature = x.view(batch_size*num_points, -1)[idx, :]
            feature = feature.view(batch_size, num_points, k, num_dims) 
            x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
            
            feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        
            return feature
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNN_100(nn.Module):
    def __init__(self,args,device):
        super(DGCNN_100, self).__init__()
        self.k = 20
        self.embedding = torch.nn.Embedding(6187, 100)
        self.bn1 = nn.BatchNorm2d(256*2)
        self.bn2 = nn.BatchNorm2d(256*2)
        self.bn3 = nn.BatchNorm2d(256*4)
        self.bn4 = nn.BatchNorm2d(256*4)
        self.bn5 = nn.BatchNorm1d(100)
        self.device = device 
        self.conv1 = nn.Sequential(nn.Conv2d(200, 256*2, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(256*4, 256*2, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(256*4,256*4, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256*8, 256*4, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(1536*2, 100, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(50*4, 256*2, bias=False)
        self.bn6 = nn.BatchNorm1d(256*2)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256*2, 128*2)
        self.bn7 = nn.BatchNorm1d(128*2)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(128*2, args.output_channels)

    def forward(self, x):
        def knn(x, k):
            inner = -2*torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
            idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
            return idx


        def get_graph_feature(x, k=20, idx=None):
            batch_size = x.size(0)
            num_points = x.size(2)
            x = x.view(batch_size, -1, num_points)
            if idx is None:
                idx = knn(x, k=k)   # (batch_size, num_points, k)
            

            idx_base = torch.arange(0, batch_size,device=self.device).view(-1, 1, 1)*num_points

            idx = idx + idx_base

            idx = idx.view(-1)
        
            _, num_dims, _ = x.size()

            x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
            feature = x.view(batch_size*num_points, -1)[idx, :]
            feature = feature.view(batch_size, num_points, k, num_dims) 
            x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
            
            feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        
            return feature
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

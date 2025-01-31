#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:11:46 2018

@author: wei
"""

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
#from MNN_New3D import MNN
from MNN_New2D import MNN
from time import time
import matplotlib.pyplot as plt


class MNNNet(nn.Module):
    def __init__(self):
        super(MNNNet,self).__init__()
        self.MNN1 = MNN(1,10,5)
        #self.conv1 = nn.Conv2d(2, 4, kernel_size=5)
        self.MNN2 = MNN(10,1,5)
        self.MNN3 = MNN(10,1,3)
        #self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(360,100)
        self.fc2 = nn.Linear(100,10)
    
    def forward(self,x):
        #import pdb; pdb.set_trace()
        #output = F.max_pool2d(x,2)
        output = self.MNN1(x)
        output = F.max_pool2d(output,2)
        output = self.MNN2(output)
        output = self.MNN3(output)
        output = output.view(-1,360)
        output = F.relu(self.fc1(output))
        #output = F.dropout(output, training=self.training)
        output = self.fc2(output)
        return F.log_softmax(output,1)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,1)
    
start_whole = time()

parser = argparse.ArgumentParser(description='PyTorch MNIST with MNN and CNN')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--MNNmodel', type=str, default = './MNNModel_2D/model_96.pth',  help='model path')
parser.add_argument('--CNNmodel', type=str, default = './CNNModel_2D/model_43.pth',  help='model path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
import pdb; pdb.set_trace()   
model_CNN = CNNNet()
model_CNN.load_state_dict(torch.load(args.CNNmodel))
model_CNN.eval()
#model_CNN.cuda()

model_MNN = MNNNet()
model_MNN.load_state_dict(torch.load(args.MNNmodel))
model_MNN.eval()
#model_MNN.cuda()

count  = 0
for i, (data, target) in enumerate(test_loader,0):
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    
    data, target = Variable(data, volatile=True), Variable(target)
    outputCNN = model_CNN(data)
    outputMNN = model_MNN(data)
    predCNN = outputCNN.data.max(1, keepdim=True)[1]
    predMNN = outputMNN.data.max(1, keepdim=True)[1]
    
    if (predCNN.eq(target.data.view_as(predCNN)).cpu()==1).all() and (predMNN.eq(target.data.view_as(predMNN)).cpu()==0).all():
        CNNFmap1 = model_CNN.conv1(data)
        CNNPool1 = F.max_pool2d(CNNFmap1,2)
        CNNFmap2 = model_CNN.conv2(F.relu(CNNPool1))
        
        MNNFmap1 = model_MNN.MNN1(data)
        MNNPool1 = F.max_pool2d(MNNFmap1,2)
        MNNFmap2 = model_MNN.MNN2(MNNPool1)
        #import pdb; pdb.set_trace()
        plt.figure()
        plt.imshow(data[0,0].cpu().data.numpy(), cmap='gray')
        #plt.savefig('CNN0MNN0({})'.format(count))
        count += 1
        if count == 10: break
# =============================================================================
#         for i in range(CNNFmap1.size()[1]):
#             plt.figure()
#             plt.imshow(CNNFmap1[0,i].cpu().data.numpy(), cmap='gray')
#             plt.savefig('CNNFmap1({}))'.format(i))
#         
#             plt.figure()
#             plt.imshow(CNNPool1[0,i].cpu().data.numpy(), cmap='gray')
#             plt.savefig('CNNPool1({})'.format(i))
#         
#             plt.figure()
#             plt.imshow(CNNFmap2[0,i].cpu().data.numpy(), cmap='gray')
#             plt.savefig('CNNFmap2({}))'.format(i))
#     
#         for j in range (MNNFmap1.size()[1]):
#             plt.figure()
#             plt.imshow(MNNFmap1[0,j].cpu().data.numpy(), cmap='gray')
#             plt.savefig('MNNFmap1({}))'.format(j))
#         
#             plt.figure()
#             plt.imshow(MNNPool1[0,j].cpu().data.numpy(), cmap='gray')
#             plt.savefig('MNNPool1({})'.format(j))
#         
#             plt.figure()
#             plt.imshow(MNNFmap2[0,j].cpu().data.numpy(), cmap='gray')
#             plt.savefig('MNNFmap2({}))'.format(j))
#         break
# =============================================================================
 
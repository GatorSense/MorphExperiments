#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:57:53 2017

@author: weihuangxu
"""

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from MNN_New2D import MNN
#from MultiMNN import MulMNN
from time import time
import matplotlib.pyplot as plt
from comet_ml import start
from comet_ml.integration.pytorch import log_model

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)
# #torch.initial_seed()
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.MNN1 = MNN(1,10,5)
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.MNN2 = MNN(10,5,5)
        #self.MNN3 = MNN(5,1,5)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=5)
        self.fc1 = nn.Linear(1800,100)
        self.fc2 = nn.Linear(100,10)
    
    def forward(self,x):
        #import pdb; pdb.set_trace()
        output = F.max_pool2d(x,2)
        output = self.MNN1(output)
        #output = F.max_pool2d(output,2)
        output = self.MNN2(output)
        #output = self.MNN3(output)
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        #output = F.dropout(output, training=self.training)
        output = self.fc2(output)
        return F.log_softmax(output,1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
dtype = torch.FloatTensor

def train(epoch):
    model.train()
    start_train = time()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

def test():
    
    model.eval()
    start_test = time()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    stop_test = time()
    print('==== Test Cycle Time ====\n', str(stop_test - start_test))
    return correct

experiment = start(
  api_key="ACmLuj8t9U7VuG1PAr1yksnM2",
  project_name="morphological",
  workspace="joannekim"
)

accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch] = test()
    experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)

accuracy /= 100
print(accuracy.max(0))
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

plt.figure()
plt.plot(accuracy[1:].numpy())
plt.xlabel('Epoch')
<<<<<<< HEAD
plt.ylabel('Accuracy (%)')
=======
plt.ylabel('Accuracy (%)')
>>>>>>> 3c8fc1f93e7f3ddf43d0599f2efb4be18f008dcf

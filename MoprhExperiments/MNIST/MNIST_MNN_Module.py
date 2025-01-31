#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/7/17 12:42 PM
# @Author  : Weihuang Xu
# @Site    : 
# @File    : MNIST_pytorch.py
# @Software: PyCharm


import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from MNN_Module import Hitmiss
from time import time

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST 3,4 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

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
        super(Net, self).__init__()
        self.MNN1 = Hitmiss(7)
        self.MNN2 = Hitmiss(7)
        self.fc1 = nn.Linear(256, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        output = self.MNN2(self.MNN1(x))
        output = output.view(1, 256)
        output = F.relu(self.fc1(output))
        output = F.sigmoid(self.fc2(output))
        return output

dtype = torch.FloatTensor
model = Net()

if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):

    model.train()
    start_train = time()
    loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        indextrain34 = np.where(np.logical_or(target.numpy() == 3, target.numpy() == 4))
        # if batch set has no 3 and 4, error saying that numpy array has zero-sized dimensions
        if len(indextrain34[0]) == 0:
            continue
        x_train = data.numpy()[indextrain34]
        y_train = target.numpy()[indextrain34] - 3
        x_train = Variable(torch.from_numpy(x_train).type(dtype), requires_grad=False)
        y_train = Variable(torch.from_numpy(y_train).type(dtype), requires_grad=False)

        if args.cuda:
            x_train, y_train = x_train.cuda(), y_train.cuda()

        optimizer.zero_grad()

        for i in range(x_train.size()[0]):
            result = model(x_train[i, 0])
            loss += F.binary_cross_entropy(result[0], y_train[i]) / x_train.size()[0]

        loss.backward(retain_graph = True)
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

    # if batch_idx % 50 == 0:
    #     print(batch_idx, loss)


def test():

    model.eval()
    loss_test = 0
    accuracy = 0
    length = 0
    start_test = time()
    
    for data, target in test_loader:
        indextest34 = np.where(np.logical_or(target.numpy() == 3, target.numpy() == 4))
        # if batch set has no 3 and 4, error saying that numpy array has zero-sized dimensions
        if len(indextest34[0]) == 0:
            continue
        x_test = data.numpy()[indextest34]
        x_test = Variable(torch.from_numpy(x_test).type(dtype), requires_grad=False)
        y_test = target.numpy()[indextest34] - 3
        y_test = Variable(torch.from_numpy(y_test).type(dtype), requires_grad=False)
        # creat a veriable to save label of predict result
        label = Variable(torch.zeros(y_test.size()).type(dtype), requires_grad=False)
        length += len(y_test)

        if args.cuda:
            x_test, y_test = x_test.cuda(), y_test.cuda()

        for i in range(x_test.size()[0]):
            result = model(x_test[i, 0])
            loss_test += F.binary_cross_entropy(result[0], y_test[i])

            if result.data[0, 0] >= 0.5:
                label[i] = 1

        accuracy += (label.data == y_test.data).cpu().sum()


    loss_test /= length
    accuracy /= length
    print('==== Test Loss ====', loss_test)
    print('==== Test Accuracy ====', accuracy)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     loss_test, accuracy, len(test_loader.dataset),
    #     100. * accuracy / len(test_loader.dataset)))

    stop_test = time()
    print('==== Test Cycle Time ====\n', str(stop_test - start_test))

for epoch in range (1, args.epochs+1):
    train(epoch)
    test()










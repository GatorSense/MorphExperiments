#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:57:50 2017

@author: weihuangxu
"""
# two MNN layers with filter size 6x6, then two fully connection layers. Only test 3,4 in MNIST

import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from HitMiss import HitMiss
import numpy as np
import matplotlib.pyplot as plt
from time import time

start_whole = time()
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)

HitMiss = HitMiss()

dtype = torch.FloatTensor

K1_hit = Variable(torch.randn(7, 7).type(dtype) * 0.1, requires_grad=True)
K1_miss = Variable(torch.randn(7, 7).type(dtype) * 0.1, requires_grad=True)

K2_hit = Variable(torch.randn(7, 7).type(dtype) * 0.1, requires_grad=True)
K2_miss = Variable(torch.randn(7, 7).type(dtype) * 0.1, requires_grad=True)

Fc_weight1 = Variable(torch.randn(256, 50).type(dtype) * 0.1, requires_grad=True)
Fc_weight2 = Variable(torch.randn(50, 1).type(dtype) * 0.1, requires_grad=True)

# Fmap is for backward test in MNN layer
# Fmap = Variable(torch.randn(16,16).type(dtype), requires_grad = False)

learning_rate = 0.05
alpha = 0.5
momentum_K1_hit = 0
momentum_K1_miss = 0
momentum_K2_hit = 0
momentum_K2_miss = 0
momentum_Fc1 = 0
momentum_Fc2 = 0

# Training Part
start_wholetrain = time()

for epoch in range(10):

    start_train = time()

    # data.size() is 64x1x28x28; target.size() is 64
    # type(data) is torch.FloatTensor and type(target) is torch.LongTensor, all convert to FloatTensor
    # when wrapped in Variable
    # use np.where to find the index of 3 and 4 in train set(data and target)

    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0

        indexTrain34 = np.where(np.logical_or(target.numpy() == 3, target.numpy() == 4))
        # if batch set has no 3 and 4, error saying that numpy array has zero-sized dimensions
        if len(indexTrain34[0]) == 0:
            continue
        x_train = data.numpy()[indexTrain34]
        x_train = Variable(torch.from_numpy(x_train).type(dtype), requires_grad=False)
        y_train = target.numpy()[indexTrain34] - 3
        y_train = Variable(torch.from_numpy(y_train).type(dtype), requires_grad=False)
        # y_train is a tensor with size format [10] not [10,1]

        # ==============================================================================
        #         # Visualize the figures and label
        #         if batch_idx < 10:
        #             plt.figure(batch_idx)
        #             plt.imshow(x_train.data.numpy()[0][0],cmap = 'gray')
        #             print(y_train[0])
        # ==============================================================================

        for i in range(x_train.size()[0]):
            result = HitMiss.forward(x_train[i, 0], K1_hit, K1_miss)
            result = HitMiss.forward(result, K2_hit, K2_miss)

            # loss +=  0.5 * (result-Fmap).pow(2).sum()/x_train.size()[0]

            result = result.view(1, 256)
            result = F.relu(torch.mm(result, Fc_weight1))
            result = F.sigmoid(torch.mm(result, Fc_weight2))
            # print(result[0],result)
            loss += F.binary_cross_entropy(result[0], y_train[i]) / x_train.size()[0]

        loss.backward()

        delta_K1_hit = learning_rate * K1_hit.grad.data + alpha * momentum_K1_hit
        delta_K1_miss = learning_rate * K1_miss.grad.data + alpha * momentum_K1_miss
        delta_K2_hit = learning_rate * K2_hit.grad.data + alpha * momentum_K2_hit
        delta_K2_miss = learning_rate * K2_miss.grad.data + alpha * momentum_K2_miss

        delta_Fc_weight1 = learning_rate * Fc_weight1.grad.data + alpha * momentum_Fc1
        delta_Fc_weight2 = learning_rate * Fc_weight2.grad.data + alpha * momentum_Fc2

        K1_hit.data -= delta_K1_hit
        K1_miss.data -= delta_K1_miss
        K2_hit.data -= delta_K2_hit
        K2_miss.data -= delta_K2_miss
        Fc_weight1.data -= delta_Fc_weight1
        Fc_weight2.data -= delta_Fc_weight2
        # print(K1_hit.grad.data)

        momentum_K1_hit = delta_K1_hit
        momentum_K1_miss = delta_K1_miss
        momentum_K2_hit = delta_K2_hit
        momentum_K2_miss = delta_K2_miss
        momentum_Fc1 = delta_Fc_weight1
        momentum_Fc2 = delta_Fc_weight2

        K1_hit.grad.data.zero_()
        K1_miss.grad.data.zero_()
        K2_hit.grad.data.zero_()
        K2_miss.grad.data.zero_()
        Fc_weight1.grad.data.zero_()
        Fc_weight2.grad.data.zero_()

        if batch_idx % 50 == 0:
            print(batch_idx, loss)

    print(epoch)
    stop_train = time()
    print('==== Training Cycle Time ====\n', str(stop_train - start_train))

stop_wholetrain = time()
print('==== Whole Training Time ====\n', str(stop_wholetrain - start_wholetrain))

# Test Part

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=1000, shuffle=True)

start_wholetest = time()

loss_test = 0
accuracy = 0
length = 0

for batch_idx, (data, target) in enumerate(test_loader):

    start_test = time()

    indexTest34 = np.where(np.logical_or(target.numpy() == 3, target.numpy() == 4))
    # if batch set has no 3 and 4, error saying that numpy array has zero-sized dimensions
    if len(indexTest34[0]) == 0:
        continue
    x_test = data.numpy()[indexTest34]
    x_test = Variable(torch.from_numpy(x_test).type(dtype), requires_grad=False)
    y_test = target.numpy()[indexTest34] - 3
    y_test = Variable(torch.from_numpy(y_test).type(dtype), requires_grad=False)

    # creat a veriable to save label of predict result
    label = Variable(torch.zeros(y_test.size()).type(dtype), requires_grad=False)
    length += len(y_test)

    for i in range(x_test.size()[0]):

        result = HitMiss.forward(x_test[i, 0], K1_hit, K1_miss)
        result = HitMiss.forward(result, K2_hit, K2_miss)

        result = result.view(1, 256)
        result = F.relu(torch.mm(result, Fc_weight1))
        result = F.sigmoid(torch.mm(result, Fc_weight2))
        if result.data[0, 0] >= 0.5:
            label[i] = 1

        loss_test += F.binary_cross_entropy(result[0], y_test[i])

    stop_test = time()
    print('==== Test Cycle Time ====\n', str(stop_test - start_test))

    accuracy += (label.data == y_test.data).sum()

loss_test /= length
accuracy /= length
print('==== Test Loss ====', loss_test)
print('==== Test Accuracy ====', accuracy)

stop_wholetest = time()
print('==== Whole Test Time ====\n', str(stop_wholetest - start_wholetest))



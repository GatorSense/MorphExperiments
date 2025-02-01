#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:58:15 2017

@author: minggao
"""

import idx2numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
from hit_pytorch import Hit
from miss_pytorch import Miss
import matplotlib.pyplot as plt
from time import time


start_whole = time()
start_loaddata = time()

dtype = torch.FloatTensor

trainSet = idx2numpy.convert_from_file('train-images-idx3-ubyte')
labelTrain = idx2numpy.convert_from_file('train-labels-idx1-ubyte')

indexTrain34 = np.where(np.logical_or(labelTrain == 3, labelTrain == 4))

trainSet34 = trainSet[indexTrain34]
x_train = Variable(torch.from_numpy(trainSet34).type(dtype), requires_grad=False)/256
#print(type(labelTrain))
labelTrain34 = labelTrain[indexTrain34]
y_train = Variable(torch.zeros(np #result = result.view().size(labelTrain34), 1).type(dtype), requires_grad=False)
for i in range(y_train.size()[0]):
    if labelTrain34[i] == 3:
        y_train[i, 0] = 1
    
    if labelTrain34[i] == 4:
        y_train[i, 1] = 1
 
testSet = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')
labelTest = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')  

indexTest34 = np.where(np.logical_or(labelTest == 3, labelTest == 4))

testSet34 = testSet[indexTest34]
x_test = Variable(torch.from_numpy(testSet34).type(dtype), requires_grad=False)

labelTest34 = labelTest[indexTest34]
y_test = Variable(torch.zeros(np.size(labelTest34), 2).type(dtype), requires_grad=False)
for i in range(y_test.size()[0]):
    if labelTest34[i] == 3:
        y_test[i, 0] = 1
    
    if labelTest34[i] == 4:
        y_test[i, 1] = 1      
#==============================================================================
#     if i % 100 ==0:
#         print('=====',y[i])
#         print('======',labelTrain34[i])
#==============================================================================

dtype = torch.FloatTensor
#initial all the weights in filter and fully connection layer
K1_hit = Variable(torch.randn(15,15).type(dtype) * 0.1, requires_grad=True)
K1_miss = Variable(torch.randn(15,15).type(dtype) * 0.1, requires_grad=True)
                     
K2_hit = Variable(torch.randn(7,7).type(dtype) * 0.1, requires_grad=True)
K2_miss = Variable(torch.randn(7,7).type(dtype) * 0.1, requires_grad=True)
                     
K3_hit = Variable(torch.randn(5,5).type(dtype) * 0.1, requires_grad=True)
K3_miss = Variable(torch.randn(5,5).type(dtype) * 0.1, requires_grad=True)
                     
K4_hit = Variable(torch.randn(3,3).type(dtype) * 0.1, requires_grad=True)
K4_miss = Variable(torch.randn(3,3).type(dtype) * 0.1, requires_grad=True) 

Fc_weight1 = Variable(torch.randn(4,10).type(dtype) * 0.1, requires_grad=True)
Fc_weight2 = Variable(torch.randn(10,2).type(dtype) * 0.1, requires_grad=True)


# Forward propagation include 4 MNN layer then fully connection
Hit = Hit()
Miss = Miss()
Sigmoid = nn.Sigmoid()

y_MNN_train = Variable(torch.zeros(x_train.size()[0],4).type(dtype),requires_grad=False)

alpha = 0.5
learning_rate = 0.5
momentum_hit1 = 0
momentum_miss1 = 0
momentum_hit2 = 0
momentum_miss2 = 0
momentum_hit3 = 0
momentum_miss3 = 0
momentum_hit4 = 0
momentum_miss4 = 0
momentum_Fc1 = 0
momentum_Fc2 = 0

stop_loaddata = time()
print('=====  Loading Data Time  =====\n', str(stop_loaddata - start_loaddata)) 


for epoch in range (50):
    loss = 0
    start_train = time()
    
    for i in range(x_train.size()[0]):
    
        temp = Hit.forward(x_train[i], K1_hit) - Miss.forward(x_train[i], K1_miss)
        temp = Hit.forward(temp, K2_hit) - Miss.forward(temp, K2_miss)
        temp = Hit.forward(temp, K3_hit) - Miss.forward(temp, K3_miss)
        temp = Hit.forward(temp, K4_hit) - Miss.forward(temp, K4_miss)
        y_MNN_train[i] = temp.view(temp.numel())
    
    #y_pred size nx2    
    y_pred_train = F.relu(torch.mm(y_MNN_train, Fc_weight1))
    y_pred_train = Sigmoid(torch.mm(y_pred_train, Fc_weight2))
    
    loss += (- y_train[i] * torch.log(y_pred_train)-(1-y_train[i]) * torch.log(1-y_pred_train))/x_train.size()[0]
    loss.backward()
    
    #backward 

    
    delta_K1_hit = learning_rate * K1_hit.grad.data + alpha * momentum_hit1
    delta_K1_miss = learning_rate * K1_miss.grad.data + alpha * momentum_miss1
    delta_K2_hit = learning_rate * K2_hit.grad.data + alpha * momentum_hit2
    delta_K2_miss = learning_rate * K2_miss.grad.data + alpha * momentum_miss2
    delta_K3_hit = learning_rate * K3_hit.grad.data + alpha * momentum_hit3
    delta_K3_miss = learning_rate * K3_miss.grad.data + alpha * momentum_miss3
    delta_K4_hit = learning_rate * K4_hit.grad.data + alpha * momentum_hit4
    delta_K4_miss = learning_rate * K4_miss.grad.data + alpha * momentum_miss4
    delta_Fc1 = learning_rate * Fc_weight1.grad.data + alpha * momentum_Fc1
    delta_Fc2 = learning_rate * Fc_weight2.grad.data + alpha * momentum_Fc2
    
    
    K1_hit.data -= delta_K1_hit 
    K1_miss.data -= delta_K1_miss
    K2_hit.data -= delta_K2_hit
    K2_miss.data -= delta_K2_miss
    K3_hit.data -= delta_K3_hit
    K3_miss.data -= delta_K3_miss
    K4_hit.data -= delta_K4_hit
    K4_miss.data -= delta_K4_miss
    Fc_weight1.data -= delta_Fc1
    Fc_weight2.data -= delta_Fc2
    
    
    momentum_hit1 = delta_K1_hit
    momentum_miss1 = delta_K1_miss
    momentum_hit2 = delta_K2_hit
    momentum_miss2 = delta_K2_miss
    momentum_hit3 = delta_K3_hit
    momentum_miss3 = delta_K3_miss
    momentum_hit4 = delta_K4_hit
    momentum_miss4 = delta_K4_miss
    momentum_Fc1 = delta_Fc1
    momentum_Fc2 = delta_Fc2
    
    K1_hit.grad.data.zero_()
    K1_miss.grad.data.zero_()
    K2_hit.grad.data.zero_()
    K2_miss.grad.data.zero_()
    K3_hit.grad.data.zero_()
    K3_miss.grad.data.zero_()
    K4_hit.grad.data.zero_()
    K4_miss.grad.data.zero_()
    Fc_weight1.grad.data.zero_()
    Fc_weight2.grad.data.zero_()
    
    
    print(i,loss)
    stop_train = time()
    print('=====  Training Time  =====\n', str(stop_train - start_train)) 
    
# test
start_test = time()
count = 0
y_MNN_test = Variable(torch.zeros(x_test.size()[0],4).type(dtype),requires_grad=False)
for i in range(x_test.size()[0]):
    
    temp_test = Hit.forward(x_test[i], K1_hit) - Miss.forward(x_test[i], K1_miss)
    temp_test = Hit.forward(temp_test, K2_hit) - Miss.forward(temp_test, K2_miss)
    temp_test = Hit.forward(temp_test, K3_hit) - Miss.forward(temp_test, K3_miss)
    temp_test = Hit.forward(temp_test, K4_hit) - Miss.forward(temp_test, K4_miss)
    y_MNN_test[i] = temp_test.view(temp_test.numel())

#y_pred size nx2    
y_pred_test = F.relu(torch.mm(y_MNN_test, Fc_weight1))
y_pred_test = Sigmoid(torch.mm(y_pred_test, Fc_weight2))

loss = - y_test.dot(torch.log(y_pred_test))

max_pred_test, index_pred = y_pred_test.max(1)
max_true_test, index_true = y_test.max(1)
accuracy = (index_pred == index_true).sum()/y_pred_test.size()[0]
print(accuracy)

stop_test = time()
print('=====  Test Time  =====\n', str(stop_test - start_test))

stop_whole = time()
print('=====  Whole Time  =====\n', str(stop_whole - start_whole))  


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:01:10 2017

@author: minggao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 00:11:04 2017

@author: Weihuang Xu
"""

import torch
import torch.autograd
from torch.autograd import Variable
import numpy as np
from numpy import unravel_index
from HitMiss import HitMiss
import matplotlib.pyplot as plt
from comet_ml import start
from comet_ml.integration.pytorch import log_model
import os

dtype = torch.FloatTensor

# generate 8x8 matrix, filter is 3x3 so Fmap is 6x6; 
# cross center indices in range 1-6 for 8x8 matrix

tag_x = np.random.randint(3, size=(2000)) + 1
tag_y = np.random.randint(3, size=(2000)) + 1

# isn't this supposed to be 7x7?
x = torch.zeros(2000,5,10).type(dtype)

#generate 3x3 cross in tag place with different intensity

#cross with square
for i in range(2000):
    # noise
    x[i] = x[i] + torch.randn(5,10) * 0.01

    # horizontal line
    x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1

    #vertical lines
    x[i][tag_x[i]-1,tag_y[i]] += 1
    x[i][tag_x[i]+1,tag_y[i]] += 1

    # rectangle?
    # x[i][tag_x[i]-1:tag_x[i]+2,tag_y[i]+4:tag_y[i]+7] += 1

sample = x[0].cpu().numpy()

# Plot the sample using Matplotlib
plt.imshow(sample, cmap='gray')
plt.colorbar()
plt.title(f"Sample {0}")
plt.savefig('./only_cross.png') 

#==============================================================================
#cross with half T
# for i in range(2000):
#     x[i] = x[i] + torch.randn(5,10) * 0.01
#     x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1
#     x[i][tag_x[i]-1,tag_y[i]] += 1
#     x[i][tag_x[i]+1,tag_y[i]] += 1
#     x[i][tag_x[i]-1:tag_x[i]+2,tag_y[i]+4] += 1
#     x[i][tag_x[i]-1,tag_y[i]+5:tag_y[i]+7] += 1
#==============================================================================

#==============================================================================
# #cross with a vertical line
# for i in range(2000):
#     x[i] = x[i] + torch.randn(5,10) * 0.01
#     x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1
#     x[i][tag_x[i]-1,tag_y[i]] += 1
#     x[i][tag_x[i]+1,tag_y[i]] += 1
#     x[i][tag_x[i]-1:tag_x[i]+2,tag_y[i]+4 + np.random.randint(3)] += 1
# 
#==============================================================================
#==============================================================================
# #cross with a horizontal line
# for i in range(2000):
#     x[i] = x[i] + torch.randn(5,10) * 0.01
#     x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1
#     x[i][tag_x[i]-1,tag_y[i]] += 1
#     x[i][tag_x[i]+1,tag_y[i]] += 1
#     x[i][tag_x[i]-1 + np.random.randint(3),tag_y[i]+4:tag_y[i]+7] += 1
#==============================================================================

#==============================================================================
# #cross with a T
# for i in range(2000):
#     x[i] = x[i] + torch.randn(5,10) * 0.01
#     x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1
#     x[i][tag_x[i]-1,tag_y[i]] += 1
#     x[i][tag_x[i]+1,tag_y[i]] += 1
#     x[i][tag_x[i]-1, tag_y[i]+4:tag_y[i]+7] += 1
#     x[i][tag_x[i]:tag_x[i]+2,tag_y[i]+5] += 1
# 
#==============================================================================
#==============================================================================
# #cross with a half cross
# for i in range(2000):
#     x[i] = x[i] + torch.randn(5,10) * 0.01
#     x[i][tag_x[i],tag_y[i]-1:tag_y[i]+2] += 1
#     x[i][tag_x[i]-1,tag_y[i]] += 1
#     x[i][tag_x[i]+1,tag_y[i]] += 1
#     x[i][tag_x[i]-1:tag_x[i]+2, tag_y[i]+6] += 1
#     x[i][tag_x[i], tag_y[i]+4:tag_y[i]+6] += 1
#==============================================================================
    
    
# =============================================================================
# for j in range(5):
#     plt.figure(j)
#     plt.imshow(x[j].numpy(),cmap='gray')
# =============================================================================

images = Variable(x, requires_grad=False)

k1_hit = torch.ones((3,3)).type(dtype)
k1_hit[0,0] = 0
k1_hit[0,2] = 0
k1_hit[2,0] = 0
k1_hit[2,2] = 0
K_Hit = Variable(k1_hit, requires_grad=False)            
#print('====K_hit====',K_Hit)


k1_miss = np.ones((3,3))
k1_miss[1,0:3]=0
k1_miss[0:3,1]=0
k1_miss = -1 * np.rot90(k1_miss, 2)
K_Miss = torch.from_numpy(k1_miss).type(dtype).requires_grad_(False)
# print('====K_miss====',K_Miss)

HitMiss = HitMiss()

# generate true output for each image
y = torch.zeros((2000, 3, 8), dtype=None, requires_grad=False)
for j in range (2000):
    y[j] = HitMiss.forward(images[j], K_Hit, K_Miss)

# Train hit and miss filter        
# initiate the K_hit and K_miss filter     
hit_train = torch.randn(3,3).type(dtype) * 0.01
miss_train = torch.randn(3,3).type(dtype) * 0.01 
hit_train = Variable(hit_train, requires_grad=True)
miss_train = Variable(miss_train, requires_grad=True)
print('====Initial K_hit====',hit_train)
print('====Initial K_miss====',miss_train)

# initiate the learning rate
learning_rate = 1e-2
momentum_hit = 0
momentum_miss = 0
loss_last = Variable(torch.zeros(1).type(dtype),requires_grad=False)

experiment = start(
#   api_key=os.environ.get("COMET_API_KEY"),
  project_name="morphological",
  workspace="joannekim"
)

experiment.set_name("Cross With Square")

for i in range(100000):
#while True:
    
    loss = 0
    
    grad_hit_train = grad_miss_train = None
    
    for j in range (1000):
        result = HitMiss.forward(images[j], hit_train, miss_train)
        loss += 0.5 * (result - y[j]).pow(2).sum()/1000
    
    temp = loss
    
    loss.backward()
    hit_train.data -= learning_rate * hit_train.grad.data + 0.5 * momentum_hit
    miss_train.data -= learning_rate * miss_train.grad.data + 0.5 * momentum_miss
    momentum_hit = learning_rate * hit_train.grad.data + 0.5 * momentum_hit
    momentum_miss = learning_rate * miss_train.grad.data + 0.5 * momentum_miss
    hit_train.grad.data.zero_()
    miss_train.grad.data.zero_()
    
    if i % 5 == 0:
    
        print('==== i ====\n', i)
        print('==== loss ====\n', loss) 
        experiment.log_metric("Loss", loss, i)
    
    if abs(loss_last.item()-temp.item()) < 0.0005:
        experiment.log_metric("Stop Training at", i)
        break
    else:
        loss_last = temp
 
  
# test set is supposed to be compromised of 150 images of each class...
# so why is this just from image index 1000 to 2000?
# test set
count = 0
for k in range (1000,2000):
    test_result = HitMiss.forward(images[k], hit_train, miss_train)
    test_result_np = test_result.data.numpy()
   
    # index+1 for matching 8x8 matrix of input
    index_pred = unravel_index(test_result_np.argmax(), test_result_np.shape)
    index_truth = (tag_x[k]-1,tag_y[k]-1)
#==============================================================================
#     if k % 20 == 0:
#         print('==== index_pred ====\n', index_pred)
#         print('==== index_truth ====\n', index_truth)
#==============================================================================
    if index_pred == index_truth:
        count +=1

accuracy = count/1000
print (accuracy)
experiment.log_metric("Accuracy", accuracy)
        
plt.figure(6)
plt.imshow(hit_train.data.numpy(),cmap='gray')

plt.figure(7)
plt.imshow(miss_train.data.numpy(),cmap='gray')
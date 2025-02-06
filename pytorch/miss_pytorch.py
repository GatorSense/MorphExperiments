#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:25:47 2017

@author: minggao
"""

import torch
import torch.autograd
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

# Inherit from Function
class Miss(Function):
    
    def forward(self, input, K_miss):
        
        #sliding window
        center = int((K_miss.size()[0]-1)/2)
        Fmap = Variable(torch.zeros(input.size()[0]-2*center, input.size()[1]-2*center), requires_grad=False)
        for i in range(center,input.size()[0]-center):
            for j in range(center,input.size()[1]-center):
            
                # window is the area coverd by the filter
                window = input[(i-center):(i+center+1),(j-center):(j+center+1)]
                temp = window - K_miss
        
                max1, indice1 = temp.max(-1)
                max2, indice2 = max1.max(0)
            
                Fmap[i-center,j-center] = max2
        
                self.save_for_backward(input, K_miss, Fmap)
        
        return Fmap
    
#==============================================================================
# dtype = torch.FloatTensor
# 
# image = np.zeros((5,12))
# image[2,1:4]=1
# image[1:4,2]=1
# image[1:4,8:11]=1     
# input = Variable(torch.from_numpy(image).type(dtype), requires_grad=False)
# 
# R_Fmap = np.zeros((3,10))
# R_Fmap[1,1] = 1
# y = Variable(torch.from_numpy(R_Fmap).type(dtype),requires_grad=False)
# 
# weight = Variable(torch.randn(3, 3), requires_grad=True)
# 
# print(input, weight)
# 
# miss_test = Miss()
# 
# y_pred = miss_test.forward(input, weight)
# 
# print(y_pred)
# 
# print(y)
# 
# loss = (y_pred - y).pow(2).sum()
# 
# loss.backward()
# 
# #print(loss.creator.previous_functions[0][0].previous_functions[0][0].previous_functions[0][0].previous_functions[0][0])
# print(weight.grad.data)
#==============================================================================

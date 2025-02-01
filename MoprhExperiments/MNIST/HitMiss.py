#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:07:06 2017

@author: weihuangxu
"""

import torch
import torch.autograd
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

class HitMiss(Function):
    
    def forward(self, input, K_hit, K_miss):
        
        #sliding window
        center = int((K_hit.size()[0]-1)/2)
        Fmap = Variable(torch.zeros(input.size()[0]-2*center, input.size()[1]-2*center), requires_grad=False)
        for i in range(center,input.size()[0]-center):
            for j in range(center,input.size()[1]-center):
            
                # window is the area coverd by the filter
                window = input[(i-center):(i+center+1),(j-center):(j+center+1)]
                temp_hit = window - K_hit
                temp_miss = window - K_miss
        
                min1, indice1 = temp_hit.min(-1)
                min2, indice2 = min1.min(0)
                
                max1, indice1 = temp_miss.max(-1)
                max2, indice2 = max1.max(0)
            
                #Fmap.data[i-center,j-center] = min2.data[0][0]
                #Only when all the calculation is done by Variable, the autograd
                #can trace back to each step.
                Fmap[i-center,j-center] = min2 - max2
                #self.save_for_backward(input, K_hit, Fmap)
        
        return Fmap
    





#==============================================================================
# HitMiss = HitMiss()
# 
# dtype = torch.FloatTensor
# 
# image = np.zeros((5,12))
# image[2,1:4]=1
# image[1:4,2]=1
# image[1:4,8:11]=1  
# input = Variable(torch.from_numpy(image).type(dtype), requires_grad=False)
# print('====input====',input)
# 
# 
# 
# R_Fmap = np.zeros((3,10))-3
# R_Fmap[1,1] = -1
# R_Fmap[0:3,4:6] = -2
# R_Fmap[1,3] = -2
# y = Variable(torch.from_numpy(R_Fmap).type(dtype), requires_grad=False)
# print('====y====',y)
# 
# 
# k1_hit = np.zeros((3,3))
# k1_hit[1,0:3]=1
# k1_hit[0:3,1]=1
# K_Hit = Variable(torch.from_numpy(k1_hit).type(dtype), requires_grad=True)            
# print('====K_hit====',K_Hit)
# 
# 
# 
# k1_miss = np.ones((3,3))
# k1_miss[1,0:3]=0
# k1_miss[0:3,1]=0
# k1_miss = -1 * np.rot90(k1_miss, 2)
# K_Miss = Variable(torch.from_numpy(k1_miss).type(dtype),requires_grad=True)
# print('====K_miss====',K_Miss)
# 
# 
# result = HitMiss.forward(input, K_Hit, K_Miss)
# print('====y_pred====', result)
#  
#==============================================================================
    
    

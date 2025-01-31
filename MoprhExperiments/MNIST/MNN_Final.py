#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:15:49 2017

@author: weihuangxu
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import Parameter
import torch.nn.functional as F

class _Hitmiss(Function):
    
    def forward(self, input, K_hit, K_miss):
        
        #sliding window
        center = int((K_hit.size()[2]-1)/2)
        out_channel = K_hit.size()[0]
        in_channel = K_hit.size()[1]
        hit_map = Variable(torch.zeros(input.size()[0], in_channel*out_channel, input.size()[2]-2*center, input.size()[3]-2*center))
        miss_map = Variable(torch.zeros(input.size()[0], in_channel*out_channel, input.size()[2]-2*center, input.size()[3]-2*center))
        for i in range(center,input.size()[2]-center):
            for j in range(center,input.size()[3]-center):
                for k in range(out_channel):
            
                    # the center of filter will slide across image. K is the number of filters.
                    #Each filter will work on the images and return the feature map to the second 
                    #dimension of output. The dimension of output: 1st is mini-batch size; 2nd is 
                    #the number of filters; 3rd and 4th is the size of feature map.
                    temp_hit = input[:,:,(i-center):(i+center+1),(j-center):(j+center+1)] - K_hit[k]
                    temp_miss = input[:,:,(i-center):(i+center+1),(j-center):(j+center+1)] - K_miss[k]
                    hit_map[:,k*in_channel:(k+1)*in_channel,i-center,j-center] = F.max_pool2d(temp_hit*-1, K_hit.size()[2])
                    miss_map[:,k*in_channel:(k+1)*in_channel,i-center,j-center] = F.max_pool2d(temp_miss, K_miss.size()[2])
        Fmap = hit_map*-1 - miss_map 
        
        return Fmap

class MNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super(MNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K_hit = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.K_miss = Parameter(torch.Tensor(out_channels, in_channels,kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * math.pow(self.kernel_size,2)
        stdv = 1. / math.sqrt(n)
        self.K_hit.data.uniform_(-stdv, stdv)
        self.K_miss.data.uniform_(-stdv, stdv)


    def forward(self, input):
        return _Hitmiss().forward(input, self.K_hit, self.K_miss)
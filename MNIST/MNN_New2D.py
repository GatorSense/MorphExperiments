#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:58:05 2018

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
    
    def forward(self, input, K_hit, K_miss, kernel_size, out_channels):
        #import pdb; pdb.set_trace()
        batch_size, in_channels, ih, iw = input.size() #dimensions of input image
        fh = ih - kernel_size + 1 #size of feature map
        out_Fmap = in_channels*out_channels
        num_blocks = fh * fh *in_channels
        input = input.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).permute(0,2,3,1,4,5) # break whole image to blocks

        F_map = Variable(torch.zeros(batch_size, out_channels, num_blocks, 1, 1))  
        input = input.cuda()
        for i in range (out_channels):
            F_hit = (input-K_hit[i])*-1
            F_hit = F_hit.contiguous().view(batch_size, num_blocks, kernel_size, kernel_size)     # reshape tensor to be 5 dimension for max_pool3D function
            F_hit = -1 * F.relu(-1 * F_hit).sum()
            F_miss = input-K_miss[i]
            F_miss = F_miss.contiguous().view(batch_size, num_blocks, kernel_size, kernel_size)   # reshape tensor to be 5 dimension for max_pool3D function
            F_miss = F.relu(F_miss).sum()
            
            F_map[:,i] = F_hit *-1 - F_miss
        
        F_map = F_map.view(batch_size, out_Fmap, fh, fh)
        return  F_map


class MNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilated_filters=None, eroded_filters=None):

        super(MNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K_hit = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.K_miss = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        if (dilated_filters and eroded_filters):
            # self.set_hitmiss_filters_to_3(selected_3)
            self.set_hitmiss_filters_to_morphed_3(dilated_filters, eroded_filters)
        else:
            self.reset_parameters()

    # Initializes hit and miss filters
    def reset_parameters(self):
        n = self.in_channels * math.pow(self.kernel_size,2)
        stdv = 1. / math.sqrt(n)
        self.K_hit.data.uniform_(-stdv, stdv)
        self.K_miss.data.uniform_(-stdv, stdv)

    def set_hit_filters(self, selected_3):
        # with torch.no_grad:
        # print(self.K_hit.shape)
        new_K_hit = self.K_hit.clone()
        for i in range(10):
            image = selected_3[i][0][0]
            new_K_hit[i][0] = image
        self.K_hit.data = Parameter(new_K_hit.detach(), requires_grad=True)
    
    def set_miss_filters(self, selected_3):
        new_K_miss = self.K_miss.clone()
        for i in range(10):
            image = selected_3[i][0][0]
            new_K_miss[i][0] = image
        self.K_miss.data = Parameter(new_K_miss.detach(), requires_grad=True)

    def set_hitmiss_filters_to_3(self, selected_3):
        self.set_hit_filters(selected_3)
        self.set_miss_filters(selected_3)

    # def set_hit_filters_morphed(self, eroded_filters):
    #     new_K_hit = self.K_hit.clone()
    #     for i in range(10):
    #         image = eroded_filters[i][0][0]
    #         new_K_hit[i][0] = image
    #     self.K_hit.data = Parameter(new_K_hit.detach(), requires_grad=True)
    
    # def set_miss_filters_morphed(self, dilated_filters):
    #     new_K_miss = self.K_miss.clone()
    #     for i in range(10):
    #         image = dilated_filters[i][0][0]
    #         new_K_miss[i][0] = image
    #     self.K_miss.data = Parameter(new_K_miss.detach(), requires_grad=True)

    def set_hitmiss_filters_to_morphed_3(self, dilated_filters, eroded_filters):
        self.set_hit_filters(eroded_filters)
        self.set_miss_filters(dilated_filters)


    def forward(self, input):
        #import pdb; pdb.set_trace()
        return _Hitmiss().forward(input, self.K_hit, self.K_miss, self.kernel_size, self.out_channels)


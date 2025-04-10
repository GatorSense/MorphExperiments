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
# import matplotlib.pyplot as plt
import numpy as np

class _Hitmiss(Function):
    
    def forward(self, input, K_hit, K_miss, kernel_size, out_channels):

        batch_size, in_channels, ih, iw = input.size() #dimensions of input image
        fh = ih - kernel_size + 1 #size of feature map
        out_Fmap = in_channels*out_channels
        num_blocks = fh * fh *in_channels
        input = input.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1).permute(0,2,3,1,4,5) # break whole image to blocks

        F_map = Variable(torch.zeros(batch_size, out_channels, num_blocks, 1, 1))  
        input = input.cuda()
        # F_hit_list = []
        # F_miss_list = []
        F_hit_list = Variable(torch.zeros(batch_size, out_channels, num_blocks, 1, 1))
        F_miss_list = Variable(torch.zeros(batch_size, out_channels, num_blocks, 1, 1))
        for i in range (out_channels):
            F_hit = (input - K_hit[i]) * -1
            F_miss = input - K_miss[i]
            # if i == 0:
            #     plt.imshow(input[0][0].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            #     plt.colorbar(label='Pixel Value')  # Add colorbar with label
            #     plt.title("Input Image")
            #     plt.show()
            #     plt.imshow(F_hit[0][0].detach().cpu().squeeze(), cmap='gray', vmin=-1, vmax=0)
            #     plt.colorbar(label='Pixel Value')  # Add colorbar with label
            #     plt.title("F_hit Image")
            #     plt.show()
            #     plt.imshow(F_miss[0][0].detach().cpu().squeeze(), cmap='gray', vmin=0, vmax=1)
            #     plt.colorbar(label='Pixel Value')  # Add colorbar with label
            #     plt.title("F_miss Image")
            #     plt.show()
            F_hit = F_hit.contiguous().view(batch_size, num_blocks, kernel_size, kernel_size)     # reshape tensor to be 5 dimension for max_pool3D function
            F_hit = (-1 * F.relu(F_hit)).sum()
            F_miss = F_miss.contiguous().view(batch_size, num_blocks, kernel_size, kernel_size)   # reshape tensor to be 5 dimension for max_pool3D function
            F_miss = F.relu(F_miss).sum()
            F_map[:,i] = F_hit - F_miss
            # F_hit_list.append(F_hit.item())
            # F_miss_list.append(F_miss.item())
            F_hit_list[:, i] = F_hit.item()
            F_miss_list[:, i] = F_miss.item()
        
        F_map = F_map.view(batch_size, out_Fmap, fh, fh)
        # hit_tensor = torch.Tensor(F_hit_list)
        # miss_tensor = torch.Tensor(F_miss_list)
        # print('in HM forward')
        # print(f'fm: {F_map.shape}', f'hit: {hit_tensor.shape}', f'miss: {miss_tensor.shape}')
        F_hit_list = F_hit_list.view(batch_size, out_Fmap, fh, fh)
        F_miss_list = F_miss_list.view(batch_size, out_Fmap, fh, fh)
        return F_map, F_hit_list, F_miss_list #hit_tensor, miss_tensor

class MNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, filter_list=None):
        ''' Initializes MNN layer.

            Args: 
            filter_list: list of filters to be initialized as.
                         Accepts either [selected_three] or [dilated_filters, eroded_filters]. The order matters!!
        '''

        super(MNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K_hit = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.K_miss = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        if (filter_list):
            if (len(filter_list) == 1):
                self.set_hitmiss_filters_to_3(filter_list[0])
            elif (len(filter_list) == 2):
                self.set_hitmiss_filters_to_morphed_3(filter_list[0], filter_list[1])
            else:
                print("***ATTENTION***\nThe filter given to MNN layer is in the wrong format!")
                exit()
        else:
            self.reset_parameters()

    # Initializes hit and miss filters
    def reset_parameters(self):
        n = self.in_channels * math.pow(self.kernel_size,2)
        stdv = 1. / math.sqrt(n)
        self.K_hit.data.uniform_(-stdv, stdv)
        self.K_miss.data.uniform_(-stdv, stdv)

    def set_hit_filters(self, selected_3):
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

    def set_hitmiss_filters_to_morphed_3(self, dilated_filters, eroded_filters):
        self.set_hit_filters(eroded_filters)
        self.set_miss_filters(dilated_filters)


    def forward(self, input):
        #import pdb; pdb.set_trace()
        return _Hitmiss().forward(input, self.K_hit, self.K_miss, self.kernel_size, self.out_channels)


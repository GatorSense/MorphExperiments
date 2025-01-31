#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/25/17 12:54 PM
# @Author  : Weihuang Xu
# @Site    : 
# @File    : MNN_Pytorch.py
# @Software: PyCharm

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn import Parameter


class _Hitmiss(Function):
    def forward(self, input, K_hit, K_miss):

        # sliding window
        center = int((K_hit.size()[0] - 1) / 2)
        Fmap = Variable(torch.zeros(input.size()[0] - 2 * center, input.size()[1] - 2 * center), requires_grad=False)
        for i in range(center, input.size()[0] - center):
            for j in range(center, input.size()[1] - center):
                # window is the area coverd by the filter
                window = input[(i - center):(i + center + 1), (j - center):(j + center + 1)]
                temp_hit = window - K_hit
                temp_miss = window - K_miss

                min1, indice1 = temp_hit.min(-1)
                min2, indice2 = min1.min(0)

                max1, indice1 = temp_miss.max(-1)
                max2, indice2 = max1.max(0)

                # Fmap.data[i-center,j-center] = min2.data[0][0]
                # Only when all the calculation is done by Variable, the autograd
                # can trace back to each step.
                Fmap[i - center, j - center] = min2 - max2
                # self.save_for_backward(input, K_hit, Fmap)

        return Fmap


class Hitmiss(nn.Module):

    def __init__(self, kernel_size):

        super(Hitmiss, self).__init__()
        #self.in_channels = in_channels
        #self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.K_hit = Parameter(torch.Tensor(kernel_size, kernel_size))
        self.K_miss = Parameter(torch.Tensor(kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):

        # self.K_hit.data = torch.randn(3,3) * 0.01
        # self.K_miss.data = torch.randn(3, 3) * 0.01

        stdv = 1. / self.kernel_size
        self.K_hit.data.uniform_(-stdv, stdv)*0.1
        self.K_miss.data.uniform_(-stdv, stdv)*0.1


    def forward(self, input):
        return _Hitmiss().forward(input, self.K_hit, self.K_miss)

# dtype = torch.FloatTensor
#
# image = np.zeros((5,12))
# image[2,1:4]=1
# image[1:4,2]=1
# image[1:4,8:11]=1
# input = Variable(torch.from_numpy(image).type(dtype), requires_grad=False)
# #print('====input====',input)
#
#
#
# R_Fmap = np.zeros((3,10))-3
# R_Fmap[1,1] = -1
# R_Fmap[0:3,4:6] = -2
# R_Fmap[1,3] = -2
# y = Variable(torch.from_numpy(R_Fmap).type(dtype), requires_grad=False)
# #print('====y====',y)
#
# mnn = Hitmiss(3)
# output = mnn(input)
# print(output)





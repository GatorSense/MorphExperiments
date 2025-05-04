"""
This code is a heavily modified version of code originally written by Xu (2023), 
revised and adapted by Joanne Kim and Sam Gallic.

Original source:
Xu, W. (2023). Deep Morph-Convolutional Neural Network: Combining Morphological Transform and Convolution in Deep Neural Networks 
(Doctoral dissertation, University of Florida). UF Digital Collections. https://ufdc.ufl.edu/UFE0059487/00001/pd
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter
import torch.nn.functional as F

class _Hitmiss(Function):
    def forward(self, input, K_hit, K_miss, kernel_size, out_channels):
        activation = nn.LeakyReLU()
        batch_size, in_channels, ih, _ = input.size()  
        fh = ih - kernel_size + 1
        out_Fmap = in_channels * out_channels
        num_blocks = fh * fh * in_channels

        # Unfold => (B, fh, fh, in_channels, k, k)
        input = input.unfold(2, kernel_size, 1) \
                     .unfold(3, kernel_size, 1) \
                     .permute(0, 2, 3, 1, 4, 5)

        # K_hit, K_miss => (out_channels, in_channels, k, k)
        # Insert batch & (fh, fh) dims so they broadcast properly:
        input_  = input.unsqueeze(1)
        K_hit_  = K_hit.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # => (1, out_channels, 1, 1, in_channels, k, k)
        K_miss_ = K_miss.unsqueeze(0).unsqueeze(2).unsqueeze(3) # same shape

        # F_hit: shape (B, out_channels, fh, fh, in_channels, k, k)
        F_hit  = -1.0 * F.relu( (input_ - K_hit_) * -1 )
        # Sum over all of (fh, fh, in_channels, k, k) => (2,3,4,5,6)
        F_hit_sum = F_hit.sum(dim=(2, 3, 4, 5, 6))

        F_miss = F.relu(input_ - K_miss_)
        F_miss_sum = F_miss.sum(dim=(2, 3, 4, 5, 6))

        F_map_val = F_hit_sum - F_miss_sum  # => (B, out_channels)

        # Expand to match your original shape
        F_map_val_expanded = F_map_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, out_channels, 1, 1, 1)
        F_map_val_expanded = F_map_val_expanded.expand(batch_size, out_channels,
                                                       num_blocks, 1, 1)
        # Similarly for F_hit_list, F_miss_list:
        F_hit_list_val = F_hit_sum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, out_channels,
                                                                                    num_blocks, 1, 1)
        F_miss_list_val = F_miss_sum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, out_channels,
                                                                                      num_blocks, 1, 1)
        # Finally reshape to (B, out_Fmap, fh, fh)
        F_map       = F_map_val_expanded.view(batch_size, out_Fmap, fh, fh)
        F_hit_list  = F_hit_list_val.view(batch_size, out_Fmap, fh, fh)
        F_miss_list = F_miss_list_val.view(batch_size, out_Fmap, fh, fh)

        return F.max_pool1d(torch.squeeze(F_map), 10), F_hit_list, F_miss_list

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


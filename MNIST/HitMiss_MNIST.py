#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:40:22 2017

@author: weihuangxu
"""

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
    






    
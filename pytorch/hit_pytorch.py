
import torch
import torch.autograd
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

# Inherit from Function
class Hit(Function):
    
    def forward(self, input, K_hit):
        
        #sliding window
        center = int((K_hit.size()[0]-1)/2)
        Fmap = Variable(torch.zeros(input.size()[0]-2*center, input.size()[1]-2*center), requires_grad=False)
        for i in range(center,input.size()[0]-center):
            for j in range(center,input.size()[1]-center):
            
                # window is the area coverd by the filter
                window = input[(i-center):(i+center+1),(j-center):(j+center+1)]
                temp = window - K_hit
        
                min1, indice1 = temp.min(-1)
                min2, indice2 = min1.min(0)
            
                #Fmap.data[i-center,j-center] = min2.data[0][0]
                #Only when all the calculation is done by Variable, the autograd
                #can trace back to each step.
                Fmap[i-center,j-center] = min2
                self.save_for_backward(input, K_hit, Fmap)
        
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
#R_Fmap = np.zeros((3,10))
#R_Fmap[1,1] = 1
#y = Variable(torch.from_numpy(R_Fmap).type(dtype),requires_grad=False)
#==============================================================================
#==============================================================================
# dtype = torch.FloatTensor
# input = Variable(torch.randn(3, 3), requires_grad=False)
# 
# y = Variable(torch.zeros(1),requires_grad=False)
# 
# weight = Variable(torch.randn(3, 3), requires_grad=True)
# 
# print(input, weight, y)
# 
# hit_test = Hit()
# 
# y_pred = hit_test.forward(input, weight)
# 
# print(input-weight)
# print(y_pred-y)
# 
# 
# loss = 0.5 * (y_pred - y).pow(2).sum()
# 
# loss.backward()
# 
# #print(loss.creator.previous_functions[0][0].previous_functions[0][0].previous_functions[0][0].previous_functions[0][0])
# print(weight.grad.data)
#==============================================================================





 





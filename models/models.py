import torch
import torch.nn as nn
from models.MNN import *
from utils.logger import *
from utils.plot import *

class MorphNet(nn.Module):
    def __init__(self, filter_list=None):
        super(MorphNet,self).__init__()
        if (filter_list):
            self.MNN1 = MNN(1,10,28, filter_list)
        else:
            self.MNN1 = MNN(1,10,28)
        self.training = True
        self.passes = 0
        self.log_filters = False
    
    def forward(self, x, epoch, experiment):
        output = x
        output, hit, miss = self.MNN1(output)

        # Plot filters
        if self.log_filters and experiment:
            plot_filters_forward(self.MNN1.K_hit, experiment, epoch, "hit")
            plot_filters_forward(self.MNN1.K_miss, experiment, epoch, "miss")

        return output, hit, miss

class ConvNet(nn.Module):
    def __init__(self, experiment):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=28)
        self.training = True
        self.done = False

    def forward(self, x, experiment):
        output = x
        output = self.conv1(output)
        return output

class MNNModel(nn.Module):
    def __init__(self):
        super(MNNModel,self).__init__()
        self.morph = MorphNet()
        self.fc1 = nn.Linear(10,2)
        self.training = True
        self.activate = nn.LeakyReLU()
    
    def forward(self, x, epoch, experiment):
        self.morph.training = self.training
        m_output, hit, miss = self.morph(x, epoch, experiment)
        output = m_output
        output = output.view(output.size(0), -1)
        output = self.activate(self.fc1(output))
        return F.log_softmax(output,1), m_output, hit, miss
    
    # Turn on and off the logging
    def log_filters(self, bool):
        self.morph.log_filters = bool

class CNNModel(nn.Module):
    def __init__(self, selected_3=None):
        super(CNNModel,self).__init__()
        self.conv = ConvNet(selected_3)
        self.fc1 = nn.Linear(20,2)
        self.training = True
    
    def forward(self, x, epoch):
        self.conv.training = self.training
        c_output = self.conv(x.cuda(), epoch).cuda()
        output = c_output
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        return F.log_softmax(output,1)
    
class MCNNModel(nn.Module):
    def __init__(self):
        super(MCNNModel,self).__init__()
        self.morph = MorphNet()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(24000,10000)
        self.fc2 = nn.Linear(10000,1000)
        self.fc3 = nn.Linear(1000,100)
        self.fc4 = nn.Linear(100,2)
        self.training = True
    
    def forward(self, x, epoch):
        self.morph.training = self.training
        self.conv.training = self.training
        m_output = self.morph(x.cuda(), epoch).cuda()
        c_output = self.conv(x.cuda(), epoch).cuda()
        output = torch.cat((m_output, c_output), dim=1)
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        return F.log_softmax(output,1)
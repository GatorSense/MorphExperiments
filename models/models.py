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
            plot_morph_filters_forward(self.MNN1.K_hit, experiment, epoch, "hit")
            plot_morph_filters_forward(self.MNN1.K_miss, experiment, epoch, "miss")

        return output, hit, miss

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=28)
        self.training = True
        self.done = False
        self.passes = 0
        self.log_filters = False

    def forward(self, x, epoch, experiment):
        output = x
        output = self.conv1(output)

        # Plot filters
        if self.log_filters and experiment:
            plot_conv_filters_forward(self.conv1, experiment, epoch)

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
    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(20,2)
        self.training = True
        self.activate = nn.LeakyReLU()
    
    def forward(self, x, epoch, experiment):
        self.conv.training = self.training
        c_output = self.conv(x, epoch, experiment)
        output = c_output
        output = output.view(output.size(0), -1)
        output = self.activate(self.fc1(output))
        return F.log_softmax(output,1), c_output, None, None
    
    def log_filters(self, bool):
        self.conv.log_filters = bool
    
class MCNNModel(nn.Module):
    def __init__(self):
        super(MCNNModel,self).__init__()
        self.morph = MorphNet()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(30,2)
        self.training = True
        self.activate = nn.LeakyReLU()

    def log_filters(self, bool):
        self.morph.log_filters = bool
        self.conv.log_filters = bool
    
    def forward(self, x, epoch, experiment):
        self.morph.training = self.training
        self.conv.training = self.training
        m_output, hit, miss = self.morph(x, epoch, experiment)
        c_output = self.conv(x, epoch, experiment)
        mc_output = torch.cat((m_output, c_output), dim=1)
        output = mc_output
        output = output.view(output.size(0), -1)
        output = self.activate(self.fc1(output))

        return F.log_softmax(output,1), mc_output, hit, miss
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:57:53 2017

@author: weihuangxu
"""

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from MNN_New2D import MNN
#from MultiMNN import MulMNN
from time import time
import matplotlib.pyplot as plt
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed)
# #torch.initial_seed()
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = datasets.MNIST(
    root='../data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)

targets = train_dataset.targets
idx_3 = (targets == 3).nonzero(as_tuple=True)[0]
train_subset_3 = Subset(train_dataset, idx_3)
idx_9 = (targets == 9).nonzero(as_tuple=True)[0]
train_subset_9 = Subset(train_dataset, idx_9)

black_images_train = torch.zeros(6000, 1, 28, 28)
black_images_train += 0.1 * torch.randn_like(black_images_train)

class BlackAnd3And9(Dataset):
    def __init__(self, sixes, threes, nines):
        self.sixes = sixes
        self.threes = threes
        self.nines = nines
    
    def __len__(self):
        return len(self.sixes) + len(self.threes) + len(self.nines)
    
    def __getitem__(self, index):
        if index < len(self.nines):
            image, _ = self.nines[index]
            return image, 1
        elif index < len(self.threes) + len(self.nines):
            image, _ = self.threes[index - len(self.nines)]
            return image, 0
        else:
            return self.black_imgs[index - len(self.threes) - len(self.nines)], 2
        
train_loader = DataLoader(BlackAnd3And9(black_images_train, train_subset_3, train_subset_9), 
                          args.batch_size, shuffle=True, **kwargs)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class MorphNet(nn.Module):
    def __init__(self):
        super(MorphNet,self).__init__()
        self.MNN1 = MNN(1,10,5)
        self.MNN2 = MNN(10,5,5)
    
    def forward(self,x):
        output = F.max_pool2d(x,2)
        output = self.MNN1(output)
        output = self.MNN2(output)
        return output
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=5)
    
    def forward(self,x):
        output = F.max_pool2d(x,2)
        output = self.conv1(output)
        output = self.conv2(output)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.morph = MorphNet()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(2160,100)
        self.fc2 = nn.Linear(100,3)
    
    def forward(self,x):
        m_output = self.morph(x.cuda()).cuda()
        c_output = self.conv(x.cuda()).cuda()
        # output = c_output
        output = torch.cat((m_output, c_output), dim=1)
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return F.log_softmax(output,1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
dtype = torch.FloatTensor

def train(epoch):
    model.train()
    start_train = time()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

def test():
    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    model.eval()
    start_test = time()
    test_loss = 0
    correct = 0

    # Store counts predicted as 3 and 9 for each digits
    pred_dict = defaultdict(lambda: [0, 0, 0])
    
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        # Get original target to determine actual number
        original_target = target.cpu().detach().numpy()

        target = torch.where((target != 3) & (target != 9), 
                             torch.tensor(2, device=target.device),
                             target)
        target = torch.where(target == 3, 
                             torch.tensor(0, device=target.device),
                             target)
        target = torch.where(target == 9, 
                             torch.tensor(1, device=target.device),
                             target)
        
        output = model(data)
        target_total = np.concatenate([target_total, target.cpu().detach().numpy()], axis=None)
        output_total = np.concatenate([output_total, output.argmax(dim=1).cpu().detach().numpy()], axis=None)
        test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # store predicted labels in the dict
        pred_np = pred.cpu().detach().numpy().flatten()
        for i in range(len(original_target)):
            digit = str(original_target[i])
            pred_dict[digit][pred_np[i]] += 1

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    experiment.log_confusion_matrix(
        y_true=target_total,
        y_predicted=output_total,
        labels=["3", "9", "Other"],
    )

    pred_dict_sorted = dict(sorted(pred_dict.items()))
    pred_df = pd.DataFrame(
        [(key, val[0], val[1], val[2]) for key, val in pred_dict_sorted.items()],
        columns=["Label", "Three", "Nine", "Neither"]
    )
    
    stop_test = time()
    print('==== Test Cycle Time ====\n', str(stop_test - start_test))
    return correct, pred_df

experiment = start(
  api_key="ACmLuj8t9U7VuG1PAr1yksnM2",
  project_name="morphological",
  workspace="joannekim"
)

# add empty dictionary at index 0 to match indexing for accuracy list
pred_dict_list = [{}]
accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch], pred_dict = test()
    pred_dict_list.append(pred_dict)
    experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)

# Log the last predicted label
experiment.log_table("pred_label_digit.csv", pred_dict_list[args.epochs]) # result from last epoch

accuracy /= 100
print(accuracy.max(0))
# print(pred_dict)
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

plt.figure()
plt.plot(accuracy[1:].numpy())
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
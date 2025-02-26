#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Nov  8 14:57:53 2017

@author: weihuangxu
"""

import os
from comet_ml import start
from comet_ml.integration.pytorch import log_model
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
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import torchvision.utils as vutils
import random
from helper_functions.plot import plot_heatmap, plot_hit_filters, plot_miss_filters
from pprint import pprint

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
parser.add_argument('--model-type', type=str, default='morph', metavar='N',
                    help='type of layer to use (default: morph, could use conv or MCNN)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args_dict = vars(args)
pprint(args_dict)

# torch.manual_seed(args.seed)
# #torch.initial_seed()
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)

import os
import torch
import matplotlib.pyplot as plt

def visualize_filters(layer, dir='filters/', title="Filters"):
    # Extract filter weights
    K_hit = layer.K_hit.data.cpu().numpy()  # Convert to NumPy
    K_miss = layer.K_miss.data.cpu().numpy()
    
    out_channels, in_channels, kernel_size, _ = K_hit.shape

    # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Iterate over filters
    for i in range(out_channels):
        for j in range(in_channels):
            # Save K_hit
            plt.imshow(K_hit[i, j], cmap='gray', interpolation='nearest')
            plt.title(f"K_hit [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_hit.png"))
            plt.clf()

            # Save K_miss
            plt.imshow(K_miss[i, j], cmap='gray', interpolation='nearest')
            plt.title(f"K_miss [{i},{j}]")
            plt.axis('off')
            plt.savefig(os.path.join(dir, f"filter_{i}_miss.png"))
            plt.clf()

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

black_images_train = torch.zeros(6000, 1, 28, 28)
black_images_train += 0.1 * torch.randn_like(black_images_train)

class BlackAndThrees(Dataset):
    def __init__(self, black_imgs, threes):
        self.black_imgs = black_imgs
        self.threes = threes
    
    def __len__(self):
        return len(self.black_imgs) + len(self.threes)
    
    def __getitem__(self, index):
        if index < len(self.threes):
            image, _ = self.threes[index]
            return image, 1
        else:
            return self.black_imgs[index - len(self.threes)], 0
        
train_loader = DataLoader(BlackAndThrees(black_images_train, train_subset_3), 
                          args.batch_size, shuffle=True, **kwargs)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class MorphNet(nn.Module):
    def __init__(self, selected_3=None):
        super(MorphNet,self).__init__()
        if (selected_3):
            self.MNN1 = MNN(1,10,28, selected_3)
        else:
            self.MNN1 = MNN(1,10,28)
        self.MNN2 = MNN(10,5,5)
        self.training = True
        self.passes = 0
        self.done = False
    
    def forward(self, x, epoch):
        output = x
        # output = F.max_pool2d(x,2)
        output = self.MNN1(output)
        if not self.training and epoch == 100 and not self.done:
            visualize_filters(self.MNN1)
            fm_dir = 'feature_maps/morph'
            os.makedirs(fm_dir, exist_ok=True)
            os.makedirs('filters', exist_ok=True)
            for batch in range(output.shape[0]):
                plt.imshow(x[batch][0].cpu().detach().numpy(), cmap='gray')
                plt.savefig(os.path.join(fm_dir, f'Batch_{batch}_Original.png'))
                plt.clf()
                for channel in range(output.shape[1]):
                    plt.imshow(output[batch][channel].cpu().detach().numpy(), cmap='gray')
                    plt.savefig(os.path.join(fm_dir, f'Batch_{batch}_Channel_{channel}.png'))
                    plt.clf()
                if batch == 100:
                    self.done = True
                    break
        output = self.MNN2(output)
        return output
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=5)
        self.training = True
        self.done = False
    
    def forward(self, x, epoch):
        output = x
        output = self.conv1(output)
        if not self.training and epoch == 100 and not self.done:
            fm_dir = 'feature_maps/conv'
            os.makedirs(fm_dir, exist_ok=True)
            for batch in range(output.shape[0]):
                plt.imshow(x[batch][0].cpu().detach().numpy(), cmap='gray')
                plt.savefig(os.path.join(fm_dir, f'Batch_{batch}_Original.png'))
                plt.clf()
                for channel in range(output.shape[1]):
                    plt.imshow(output[batch][channel].cpu().detach().numpy(), cmap='gray')
                    plt.savefig(os.path.join(fm_dir, f'Batch_{batch}_Channel_{channel}.png'))
                    plt.clf()
                if batch == 100:
                    self.done = True
                    break
        output = self.conv2(output)
        return output

class MNNModel(nn.Module):
    def __init__(self, selected_3=None):
        super(MNNModel,self).__init__()
        if (selected_3):
            self.morph = MorphNet(selected_3)
        else:
            self.morph = MorphNet()
        self.fc1 = nn.Linear(200,100)
        self.fc2 = nn.Linear(100,2)
        # self.fc3 = nn.Linear(1000,100)
        # self.fc4 = nn.Linear(100,2)
        self.training = True
    
    def forward(self, x, epoch):
        self.morph.training = self.training
        m_output = self.morph(x.cuda(), epoch).cuda()
        output = m_output 
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        # output = F.dropout(output, p=0.5, training=self.training)
        # output = self.fc3(output)
        # output = self.fc4(output)
        return F.log_softmax(output,1)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(4000,2000)
        self.fc2 = nn.Linear(2000,200)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(100,2)
        self.training = True
    
    def forward(self, x, epoch):
        self.conv.training = self.training
        c_output = self.conv(x.cuda(), epoch).cuda()
        output = c_output
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = F.dropout(output, p=0.5, training=self.training)
        output = self.fc3(output)
        output = self.fc4(output)
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

experiment = start(
  api_key="ACmLuj8t9U7VuG1PAr1yksnM2",
  project_name="morphological",
  workspace="joannekim"
)

rand_index = (np.random.rand(10) * len(train_subset_3)).astype(int)
# print(rand_index)
selected_3 = Subset(train_subset_3, rand_index)

dir = "filters/initialize/"
hit_fig, hit_plt = plot_hit_filters(selected_3)
hit_plt.savefig(os.path.join(dir, "initial_filters_hit.png"))
experiment.log_figure(figure_name="filters_hit", figure=hit_fig)

miss_fig, miss_plt = plot_miss_filters(selected_3)
miss_plt.savefig(os.path.join(dir, "initial_filters_miss.png"))
experiment.log_figure(figure_name="filters_miss", figure=miss_fig)

if args.model_type == 'morph':
    model = MNNModel(selected_3)
elif args.model_type == 'conv':
    model = CNNModel()
else:
    model = MCNNModel()

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
dtype = torch.FloatTensor

def train(epoch):
    model.train()
    model.training = True
    start_train = time()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, epoch)
        loss = F.nll_loss(output.cuda(), target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

def test(epoch):
    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    model.eval()
    start_test = time()
    test_loss = 0
    correct = 0
    model.training = False

    # Store counts predicted as 3 for each digits
    pred_dict = defaultdict(lambda: [0, 0])
    heatmap_data = np.zeros((10, 2))
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # Get original target to determine actual number
            original_target = target.cpu().detach().numpy()

            target = torch.where(target == 3, 
                                torch.tensor(1, device=target.device), 
                                torch.tensor(0, device=target.device))
            
            output = model(data, epoch)
            target_total = np.concatenate([target_total, target.cpu().detach().numpy()], axis=None)
            output_total = np.concatenate([output_total, output.argmax(dim=1).cpu().detach().numpy()], axis=None)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # store predicted labels in the dict
            pred_np = pred.cpu().detach().numpy().flatten()
            for i in range(len(original_target)):
                digit = original_target[i]
                digit_str = str(digit)

                # update counts for each digit
                pred_dict[digit_str][pred_np[i]] += 1
                heatmap_data[digit][pred_np[i]] += 1
            
        # print(pred_dict)

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        experiment.log_confusion_matrix(
            y_true=target_total,
            y_predicted=output_total,
            labels=["Not Three", "Three"],
        )

        print(heatmap_data)
        heatmap = plot_heatmap(heatmap_data)
        experiment.log_figure(figure_name="heatmap", figure=heatmap, step=epoch)
        
        # sort dictionary by key
        pred_dict_sorted = dict(sorted(pred_dict.items()))
        pred_df = pd.DataFrame(
            [(key, val[0], val[1]) for key, val in pred_dict_sorted.items()],
            columns=["Label", "Not Three", "Three"]
        )

        stop_test = time()
        print('==== Test Cycle Time ====\n', str(stop_test - start_test))
    return correct, pred_df

# add empty dictionary at index 0 to match indexing for accuracy list
pred_df_list = [{}]
accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch], pred_df = test(epoch)
    pred_df_list.append(pred_df)
    experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)

# Log the last predicted label
experiment.log_table("Predicted_Digit_Labels_Last_Epoch.csv", pred_df_list[args.epochs]) # result from last epoch

accuracy /= 100
print(accuracy.max(0))
# print(pred_dict)
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

plt.figure()
plt.plot(accuracy[1:].numpy())
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
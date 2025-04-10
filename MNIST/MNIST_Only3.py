#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Nov  8 14:57:53 2017

@author: weihuangxu
"""

import matplotlib
matplotlib.use('Agg')
import os
from comet_ml import start
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from MNN_New2D import MNN
from time import time
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from utils.plot import *
from utils.logger import log_weights
from utils.custom_dataset import ThreesAndKMNIST, generate_hitmiss_morphed_filters
from pprint import pprint
from torch.nn import Parameter
from dotenv import load_dotenv
import kornia

load_dotenv()

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = datasets.MNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform
)

targets = train_dataset.targets
idx_3 = (targets == 3).nonzero(as_tuple=True)[0]
train_subset_3 = Subset(train_dataset, idx_3)

# Load KMNIST data
kmnist_dataset = datasets.KMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

kmnist_train = torch.stack([img for img, _ in kmnist_dataset])
idx_k = np.random.randint(0, len(kmnist_train), len(train_subset_3))
kmnist_train = kmnist_train[idx_k]

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class MorphNet(nn.Module):
    def __init__(self, filter_list=None):
        super(MorphNet,self).__init__()
        if (filter_list):
            self.MNN1 = MNN(1,10,28, filter_list)
        else:
            self.MNN1 = MNN(1,10,28)
        self.training = True
        self.passes = 0
        self.done = False
        self.log_filters = False
    
    def forward(self, x, epoch):
        output = x
        output, hit, miss = self.MNN1(output)

        # Plot filters
        if not self.done and self.log_filters:
            plot_filters_forward(self.MNN1.K_hit, experiment, epoch, "hit")
            plot_filters_forward(self.MNN1.K_miss, experiment, epoch, "miss")

        return output, hit, miss

class ConvNet(nn.Module):
    def set_conv_filters(self, selected_3):
        if selected_3 is None:
            return

        # Extract images from Subset object
        if isinstance(selected_3, Subset):
            selected_3 = [selected_3.dataset[i][0] for i in selected_3.indices]  # Extract only image tensors

        # Convert to tensor and ensure correct shape
        selected_3 = torch.stack(selected_3)  # Stack list into a single tensor

        # Ensure it has the correct shape
        expected_shape = self.conv1.weight.shape  # (out_channels, in_channels, kernel_height, kernel_width)
        if selected_3.shape[0] != expected_shape[0]:  # Ensure number of filters match
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {selected_3.shape}")

        # Assign weights safely
        with torch.no_grad():
            self.conv1.weight.copy_(selected_3.detach().clone())

    def __init__(self, selected_3=None):
        super(ConvNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=28)
        self.set_conv_filters(selected_3)
        self.training = True
        self.done = False
        os.makedirs('filters/', exist_ok=True)
        hit_filters_fig, hit_filters_plot = plot_conv_filters(self.conv1)
        hit_filters_plot.savefig(os.path.join('filters', f"conv_epoch_init.png"))
        experiment.log_figure(figure_name="filters_conv_init", figure=hit_filters_fig, step=0)

    def forward(self, x, epoch):
        output = x
        output = self.conv1(output)
        if not self.training and epoch == 100 and not self.done:
            os.makedirs('filters/', exist_ok=True)
            hit_filters_fig, hit_filters_plot = plot_conv_filters(self.conv1)
            hit_filters_plot.savefig(os.path.join('filters', f"conv_epoch{epoch}.png"))
            experiment.log_figure(figure_name="filters_conv", figure=hit_filters_fig, step=epoch)

        return output

class MNNModel(nn.Module):
    def __init__(self, filter_list=None):
        super(MNNModel,self).__init__()
        if (filter_list):
            self.morph = MorphNet(filter_list)
        else:
            self.morph = MorphNet()
        self.fc1 = nn.Linear(10,2)
        self.training = True
        self.activate = nn.LeakyReLU()
    
    def forward(self, x, epoch):
        self.morph.training = self.training
        m_output, hit, miss = self.morph(x.cuda(), epoch)
        output = m_output.cuda()
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

experiment = start(
  api_key=os.environ["COMET_API_KEY"],
  project_name="morphological",
  workspace="joannekim"
)

# Create custom train loader
train_loader = DataLoader(ThreesAndKMNIST(kmnist_train, train_subset_3),
                          args.batch_size, shuffle=True, **kwargs)

# Initialize model
if args.model_type == 'morph':
    # model = MNNModel(filter_list)
    model = MNNModel()
elif args.model_type == 'conv':
    model = CNNModel()
else:
    model = MCNNModel()

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
dtype = torch.FloatTensor

def train(epoch):
    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    fm_dict = {"0": [], "1": []}
    hit_dict = {"0": [], "1": []}
    miss_dict = {"0": [], "1": []}
    model.train()
    model.training = True
    start_train = time()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            
        data, target = Variable(data), Variable(target)
        labels = target.cpu().detach().numpy()

        optimizer.zero_grad()

        # Only log filters for the first batch
        if (batch_idx == 0):
            model.log_filters(True)
            output, fm_val, hit, miss = model(data, epoch)
            model.log_filters(False)
        else:
            output, fm_val, hit, miss = model(data, epoch)

        # Compute loss and set model parameters
        loss = F.nll_loss(output.cuda(), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        target_total = np.concatenate([target_total, target.cpu().detach().numpy()], axis=None)
        output_total = np.concatenate([output_total, output.argmax(dim=1).cpu().detach().numpy()], axis=None)

        # Computes and prints loss metrics
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        # Stores values for all three labels to plot 
        indices_0 = (labels == 0).nonzero()[0]
        indices_1 = (labels == 1).nonzero()[0]
        fm_val = fm_val.detach().cpu().numpy()
        hit = hit.detach().cpu().numpy()
        miss = miss.detach().cpu().numpy()

        # plt.imshow(data[indices_0][0][0].detach().cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.show()
        # plt.imshow(data[indices_1][0][0].detach().cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.show()

        fm_dict["0"].append(fm_val[indices_0])
        fm_dict["1"].append(fm_val[indices_1])

        hit_dict["0"].append(hit[indices_0])
        hit_dict["1"].append(hit[indices_1])

        miss_dict["0"].append(miss[indices_0])
        miss_dict["1"].append(miss[indices_1])

    plot_fm_histogram(fm_dict, experiment, epoch)
    plot_hit_miss_histogram(hit_dict, "Hit", experiment, epoch)
    plot_hit_miss_histogram(miss_dict, "Miss", experiment, epoch)
    experiment.log_confusion_matrix(
            y_true=target_total,
            y_predicted=output_total,
            labels=["Not Three", "Three"],
            title="Train Set",
            file_name="train.json"
        )
    experiment.log_metric('Loss', value=total_loss/args.batch_size, epoch=epoch)
    
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

def test(epoch):
    # Initialize variables
    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    model.eval()
    start_test = time()
    test_loss = 0
    correct = 0
    model.training = False
    heatmap_data = np.zeros((10, 2))
    model.eval()
 
    # Initialize lists to collect feature maps
    fms_0_2 = []
    fms_4_9 = []
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data.cuda()), Variable(target)

            original_target = target.cpu().detach().numpy()

            target = torch.where(target == 3, 
                                torch.tensor(1, device=target.device), 
                                torch.tensor(0, device=target.device))

            # Accumulate metrics  
            output, fms, _, _ = model(data, epoch)
            target_total = np.concatenate([target_total, target.cpu().detach().numpy()], axis=None)
            output_total = np.concatenate([output_total, output.argmax(dim=1).cpu().detach().numpy()], axis=None)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            # Get feature maps for original targets < 3 and > 3
            original_target_tensor = torch.tensor(original_target).to(fms.device)
            indices_0_2 = (original_target_tensor < 3).nonzero(as_tuple=True)[0]
            indices_4_9 = (original_target_tensor > 3).nonzero(as_tuple=True)[0]

            # Store feature map values in list
            if indices_0_2.numel() > 0:
                fms_0_2.append(fms[indices_0_2].cpu().detach().numpy())
            if indices_4_9.numel() > 0:
                fms_4_9.append(fms[indices_4_9].cpu().detach().numpy())

            # Store predicted value for each MNIST class
            pred_np = pred.cpu().detach().numpy().flatten()
            for i in range(len(original_target)):
                digit = original_target[i]
                heatmap_data[digit][pred_np[i]] += 1

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        experiment.log_confusion_matrix(
            y_true=target_total,
            y_predicted=output_total,
            labels=["Not Three", "Three"],
            title="Test Set",
            file_name="test.json"
        )

        # Now plot the histograms using accumulated feature maps
        fm_hists = {"0-2": fms_0_2, "4-9": fms_4_9}
        plot_fm_histogram_test(fm_hists, experiment, epoch)
        plot_heatmap(heatmap_data, experiment, epoch)

        stop_test = time()
        print('==== Test Cycle Time ====\n', str(stop_test - start_test))

    # Number of correct test outputs
    return correct

# Training loop
accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch] = test(epoch)
    experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)
    weights = log_weights(model)
    experiment.log_metrics(weights)

accuracy /= 100
print(accuracy.max(0))
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

plt.figure()
plt.plot(accuracy[1:].numpy())
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

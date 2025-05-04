"""
This code is a heavily modified version of code originally written by Xu (2023), 
revised and adapted by Joanne Kim and Sam Gallic.

Original source:
Xu, W. (2023). Deep Morph-Convolutional Neural Network: Combining Morphological Transform and Convolution in Deep Neural Networks
(Doctoral dissertation, University of Florida). UF Digital Collections. https://ufdc.ufl.edu/UFE0059487/00001/pd
"""

import matplotlib
matplotlib.use('Agg')
import os
from comet_ml import start
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from time import time
from torch.utils.data import DataLoader, Subset
from utils.plot import *
from utils.logger import log_weights
from utils.custom_dataset import TripletThreesAndNotThreeWithLabel
from pprint import pprint
from dotenv import load_dotenv
from models.models import *

load_dotenv()

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-type', type=str, default='morph',
                    help='type of layer to use (default: morph, could use conv or MCNN)')
parser.add_argument('--use-comet', action='store_true', default=False,
                    help='uses comet.ml to log training metrics and graphics')
parser.add_argument('--filter-threes', action='store_true', default=False,
                    help='initializes filters to randomly selected threes')
parser.add_argument('--model-filename', type=str, default=None,
                    help='filename for saved model (default: None')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args_dict = vars(args)
pprint(args_dict)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

experiment = None
if args.use_comet:
    experiment = start(
    api_key=os.environ["COMET_API_KEY"],
    project_name="morphological",
    workspace="joannekim")
    experiment.log_parameters(args_dict)

# Start loading data
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

criterion_cls   = nn.NLLLoss()
criterion_trip  = nn.TripletMarginLoss(margin=1000.0)

idx_48 = ((targets == 4) | (targets == 8)).nonzero(as_tuple=True)[0]
idx_48 = np.random.randint(0, len(idx_48), len(train_subset_3))
not_three = Subset(train_dataset, idx_48)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

experiment = None
if args.use_comet:
    experiment = start(
        api_key=os.environ["COMET_API_KEY"],
        project_name="morphological",
        workspace="joannekim"
    )
    experiment.log_parameters(args_dict)

if args.model_type == 'morph':
    if args.filter_threes:
        rand_index = (np.random.rand(10) * len(train_subset_3)).astype(int)
        selected_3 = Subset(train_subset_3, rand_index)
        filter_list = [selected_3]
        model = MorphTripletModel(filter_list=filter_list)
    else:
        model = MorphTripletModel() 
elif args.model_type == 'conv':
    model = CNNModel()
else:
    model = MCNNModel()

if args.cuda:
    model.cuda()

# remaining_indices = list(set(range(len(train_subset_3))) - set(rand_index))
# train_subset_3 = Subset(train_subset_3, remaining_indices)
# train_loader = DataLoader(FilterOutThrees(not_three, train_subset_3, selected_3),
#                           args.batch_size, shuffle=True, **kwargs)

# Create custom train loader
train_loader = DataLoader(TripletThreesAndNotThreeWithLabel(not_three, train_subset_3),
                                                            args.batch_size, shuffle=True, **kwargs)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
dtype = torch.FloatTensor

def train(epoch):
    total_class = 0.0
    total_triplet = 0.0

    model.train()
    model.training = True
    start_train = time()
    total_loss = 0.0

    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    fm_dict = {"0": [], "1": []}
    hit_dict = {"0": [], "1": []}
    miss_dict = {"0": [], "1": []}

    for batch_idx, (A, P, N, y) in enumerate(train_loader):
        device = torch.device("cuda" if args.cuda else "cpu")

        labels = y.detach().cpu()

        A, P, N = A.to(device), P.to(device), N.to(device)
        optimizer.zero_grad()

        # Only log filters for the first batch
        if (batch_idx == 0):
            model.log_filters(True)
            emb_a, emb_p, emb_n = model(A, P, N, epoch, experiment)
            model.log_filters(False)
        else:
            emb_a, emb_p, emb_n = model(A, P, N, epoch, experiment)

        loss_trip = criterion_trip(emb_a, emb_p, emb_n)

        logits = model.head(emb_a)
        loss_cls = criterion_cls(logits.to(device), y.to(device))

        loss = loss_trip

        total_triplet = total_triplet + loss_trip
        total_class = total_class + loss_cls

        # Compute loss and set model parameters
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Store predictions for confusion matrix
        target_total = np.concatenate([target_total, y.cpu().detach().numpy()], axis=None)
        output_total = np.concatenate([output_total, logits.argmax(dim=1).cpu().detach().numpy()], axis=None)

        # Computes and prints loss metrics
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

        # Store feature map values for histogram
        indices_0 = (y == 0).nonzero()[0]
        indices_1 = (y == 1).nonzero()[0]
        fm_val = emb_a.detach().cpu().numpy()

        # if hit is not None:
        #     hit = hit.detach().cpu().numpy()

        # if miss is not None:
        #     miss = miss.detach().cpu().numpy()

        fm_dict["0"].append(fm_val[indices_0])
        fm_dict["1"].append(fm_val[indices_1])

        # if hit is not None:
        #     hit_dict["0"].append(hit[indices_0])
        #     hit_dict["1"].append(hit[indices_1])

        # if miss is not None:
        #     miss_dict["0"].append(miss[indices_0])
        #     miss_dict["1"].append(miss[indices_1])

    if args.use_comet:
        plot_fm_histogram(fm_dict, experiment, epoch)

        # if hit is not None:
        #     plot_hit_miss_histogram(hit_dict, "Hit", experiment, epoch)

        # if miss is not None:
        #     plot_hit_miss_histogram(miss_dict, "Miss", experiment, epoch)

        experiment.log_confusion_matrix(
                y_true=target_total,
                y_predicted=output_total,
                labels=["Not Three", "Three"],
                title="Train Set",
                file_name="train.json"
            )
        
        experiment.log_metric('Loss', value=total_loss/args.batch_size, epoch=epoch)
        experiment.log_metric('Triplet Loss', value=total_triplet/args.batch_size, epoch=epoch)
        experiment.log_metric('Classifier Loss', value=total_class/args.batch_size, epoch=epoch)
    
    stop_train = time()
    print('==== Training Time ====', str(stop_train-start_train))

def test(epoch):
    start_test = time()
    model.training = False
    model.eval()
    test_loss = 0
    correct = 0

    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    heatmap_data = np.zeros((10, 2))
    fms_0_2 = []
    fms_4_9 = []
 
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)
            original_target = target.cpu().detach().numpy()
            target = torch.where(target == 3, 
                                torch.tensor(1, device=target.device), 
                                torch.tensor(0, device=target.device))

            # Accumulate metrics
            output, fms, _, _ = model(data, epoch=epoch, experiment=experiment)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            target_total = np.concatenate([target_total, target.cpu().detach().numpy()], axis=None)
            output_total = np.concatenate([output_total, output.argmax(dim=1).cpu().detach().numpy()], axis=None)

            # Store predicted value for each MNIST class
            pred_np = pred.cpu().detach().numpy().flatten()
            for i in range(len(original_target)):
                digit = original_target[i]
                heatmap_data[digit][pred_np[i]] += 1
            
            # Get feature maps for original targets < 3 and > 3
            original_target_tensor = torch.tensor(original_target).to(fms.device)
            indices_0_2 = (original_target_tensor < 3).nonzero(as_tuple=True)[0]
            indices_4_9 = (original_target_tensor > 3).nonzero(as_tuple=True)[0]

            # Store feature map values in list
            if indices_0_2.numel() > 0:
                fms_0_2.append(fms[indices_0_2].cpu().detach().numpy())
            if indices_4_9.numel() > 0:
                fms_4_9.append(fms[indices_4_9].cpu().detach().numpy())

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        if args.use_comet:
            experiment.log_confusion_matrix(
                y_true=target_total,
                y_predicted=output_total,
                labels=["Not Three", "Three"],
                title="Test Set",
                file_name="test.json"
            )
            plot_heatmap(heatmap_data, experiment, epoch)
            plot_fm_histogram_test(fms_0_2, fms_4_9, experiment, epoch)

        stop_test = time()
        print('==== Test Cycle Time ====\n', str(stop_test - start_test))

    # Number of correct test outputs
    return correct

# Training loop
accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch] = test(epoch)
    if args.use_comet:
        experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)
        weights = log_weights(model)
        experiment.log_metrics(weights)

if args.model_filename is not None:
    torch.save(model, args.model_filename)

accuracy /= 100
print(accuracy.max(0))
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

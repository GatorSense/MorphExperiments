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
from torch.utils.data import DataLoader, Subset, Dataset
from utils.plot import *
from utils.logger import log_weights
from utils.custom_dataset import TripletThreesAndNotThreeWithLabel
from pprint import pprint
from dotenv import load_dotenv
from models.models import *
from sklearn.cluster import KMeans

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

class CentersDataset(Dataset):
    def __init__(self, centers):
        self.centers = torch.tensor(centers).float()

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], -1

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
idx_3 = (targets == 5).nonzero(as_tuple=True)[0]
train_subset_3 = Subset(train_dataset, idx_3)

clustered = KMeans(n_clusters=10)
images = torch.stack([x[0] for x in train_subset_3])
images_flat = images.view(len(train_subset_3), -1)
clustered.fit(images_flat)
centers = clustered.cluster_centers_.reshape((10, 28, 28))
centers_dataset = CentersDataset(centers)

criterion_cls   = nn.NLLLoss()
criterion_trip  = nn.TripletMarginLoss(margin=1.0)

idx_48 = ((targets == 3) | (targets == 8)).nonzero(as_tuple=True)[0]
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
        # rand_index = (np.random.rand(10) * len(train_subset_3)).astype(int)
        # selected_3 = Subset(train_subset_3, rand_index)
        # filter_list = [selected_3]
        model = FullMorphModel(filter_list=[centers_dataset])
    else:
        model = FullMorphModel() 
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
    total_multi_class = 0.0
    total_context_class = 0.0
    total_triplet = 0.0

    model.train()
    model.training = True
    start_train = time()
    total_loss = 0.0

    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    fm_dict = {"0": [], "1": []}

    for batch_idx, (A, P, N, y) in enumerate(train_loader):
        device = torch.device("cuda" if args.cuda else "cpu")

        A, P, N = A.to(device), P.to(device), N.to(device)
        optimizer.zero_grad()

        # Only log filters for the first batch
        if (batch_idx == 0):
            model.log_filters(True)
            pred_multi, pred_context, pred_triplet = model(A, P, N, epoch, experiment)
            emb_a, emb_p, emb_n = pred_triplet
            model.log_filters(False)
        else:
            pred_multi, pred_context, pred_triplet = model(A, P, N, epoch, experiment)
            emb_a, emb_p, emb_n = pred_triplet

        loss_trip = criterion_trip(emb_a, emb_p, emb_n)

        loss_cls_multi = criterion_cls(pred_multi.to(device), y.to(device))
        loss_cls_context = criterion_cls(pred_context.to(device), y.to(device))
        loss_cls = 0.5 * (loss_cls_multi + loss_cls_context)

        loss = loss_cls_multi

        total_multi_class = total_multi_class + loss_cls_multi
        total_context_class = total_context_class + loss_cls_context
        total_triplet = total_triplet + loss_trip

        # Compute loss and set model parameters
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        mean_pred = 0.5 * (pred_multi + pred_context)

        # Store predictions for confusion matrix
        target_total = np.concatenate([target_total, y.cpu().detach().numpy()], axis=None)
        # output_total = np.concatenate([output_total, (mean_pred).argmax(dim=1).cpu().detach().numpy()], axis=None)
        output_total = np.concatenate([output_total, (pred_multi).argmax(dim=1).cpu().detach().numpy()], axis=None)

        # Store feature map values for histogram
        indices_0 = (y == 0).nonzero()[0]
        indices_1 = (y == 1).nonzero()[0]
        fm_val = emb_a.detach().cpu().numpy()
        # print(emb_a.shape, emb_p.shape, emb_n.shape)

        fm_dict["0"].append(fm_val[indices_0])
        fm_dict["1"].append(fm_val[indices_1])

    if args.use_comet:
        plot_fm_histogram(fm_dict, experiment, epoch)

        experiment.log_confusion_matrix(
                y_true=target_total,
                y_predicted=output_total,
                labels=["Not Three", "Three"],
                title="Train Set",
                file_name="train.json"
            )
        
        experiment.log_metric('Loss', value=(total_loss / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Triplet Loss', value=(total_triplet / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Context Classifier Loss', value=(total_context_class / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Multi-Layer Classifier Loss', value=(total_multi_class / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
    
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
            target = torch.where(target == 5, 
                                torch.tensor(1, device=target.device), 
                                torch.tensor(0, device=target.device))

            # Accumulate metrics
            pred_multi, pred_context, _ = model(data, None, None, epoch, experiment)
            output = 0.5 * (pred_multi + pred_context)
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
            
            # # Get feature maps for original targets < 3 and > 3
            # original_target_tensor = torch.tensor(original_target).to(fms.device)
            # indices_0_2 = (original_target_tensor < 3).nonzero(as_tuple=True)[0]
            # indices_4_9 = (original_target_tensor > 3).nonzero(as_tuple=True)[0]

            # # Store feature map values in list
            # if indices_0_2.numel() > 0:
            #     fms_0_2.append(fms[indices_0_2].cpu().detach().numpy())
            # if indices_4_9.numel() > 0:
            #     fms_4_9.append(fms[indices_4_9].cpu().detach().numpy())

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
            # plot_fm_histogram_test(fms_0_2, fms_4_9, experiment, epoch)

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

"""
This code is a heavily modified version of code originally written by Xu (2023), 
revised and adapted by Joanne Kim and Sam Gallic.

Original source:
Xu, W. (2023). Deep Morph-Convolutional Neural Network: Combining Morphological Transform and Convolution in Deep Neural Networks
(Doctoral dissertation, University of Florida). UF Digital Collections. https://ufdc.ufl.edu/UFE0059487/00001/pd
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from torchvision.utils import save_image
from utils.plot import *
from utils.logger import log_weights
from utils.plant_dataset import BinaryLeafDataset
from pprint import pprint
from dotenv import load_dotenv
from models.models import *
from sklearn.cluster import KMeans
import pandas as pd
import random
from PIL import Image
import io
from collections import defaultdict

load_dotenv()

start_whole = time()
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST with MNNV2')
parser.add_argument('--batch-size', type=int, default=4,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=8,
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

POS_LABEL = 10
NEG_LABELS = {1, 7}

def read_metadata_expand_ranges(csv_path):
    df = pd.read_csv(csv_path)
    filename_to_meta = {}
    for _, row in df.iterrows():
        start, end = map(int, str(row['filename']).split('-'))
        for i in range(start, end + 1):
            # fname = f"{i}.jpg"
            fname = f"{i}.pt"
            filename_to_meta[fname] = {
                'label': int(row['label']),
                'scientific_name': row['Scientific Name'],
                'common_name': row['Common Name(s)'],
                'url': row['URL'],
            }
    all_filenames = sorted(filename_to_meta.keys())
    return all_filenames, filename_to_meta

# Start loading data
transform=transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

all_filenames, filename_to_meta = read_metadata_expand_ranges('data/leaves/labels.csv')

def diversity_loss_morph(morph):
    H = morph.K_hit.data  
    M = morph.K_miss.data 

    V = H.view(H.size(0), -1) # Flatten each filter
    V = F.normalize(V, dim=1) # Normalize each one
    G = V @ V.t() # Compute the Gram matrix
    off_diag_H = G.cuda() - torch.eye(G.size(0)).cuda() # Don't penalize being similar to self

    V = M.view(M.size(0), -1) 
    V = F.normalize(V, dim=1) 
    G = V @ V.t()   
    off_diag_M = G.cuda() - torch.eye(G.size(0)).cuda()

    return off_diag_H.pow(2).mean() + off_diag_M.pow(2).mean()     

# single shuffle/split for the whole set
random.seed(42)
random.shuffle(all_filenames)
split_idx = int(0.8 * len(all_filenames))
train_filenames = all_filenames[:split_idx]
test_filenames  = all_filenames[split_idx:]

train_dataset = BinaryLeafDataset(train_filenames, filename_to_meta, 'data/leaves/trained_resnet',
                                  transform=transform, mode='train')
test_dataset  = BinaryLeafDataset(test_filenames,  filename_to_meta, 'data/leaves/trained_resnet',
                                  transform=transform, mode='test')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=args.cuda)
test_loader  = DataLoader(test_dataset,  batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=args.cuda)

pos_imgs = []
for imgs, labels, orig_labels, fnames in train_loader:
    mask = (orig_labels == POS_LABEL)
    if mask.any():
        pos_imgs.append(imgs[mask])

if len(pos_imgs) > 0:
    pos_tensor = torch.cat(pos_imgs, dim=0)
else:
    pos_tensor = torch.empty(0, *next(iter(train_loader))[0].shape[1:])  # empty tensor

cluster_dir = "data/leaves/10_clusters/features"
files = sorted(os.listdir(cluster_dir))

avg_images = []
for fname in files:
    if fname.endswith(".pt") or fname.endswith(".pth"):
        path = os.path.join(cluster_dir, fname)
        arr = torch.load(path, weights_only=False)  # could be np.ndarray or torch.Tensor
        tensor = torch.as_tensor(arr)  # ensures it’s a Tensor
        avg_images.append(tensor)

# -> Tensor [10, 2048, 16, 16]
avg_images = torch.stack(avg_images).float()
print(avg_images.shape)

criterion_cls = nn.NLLLoss()

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
        model = FullMorphModel(filter_list=avg_images) 
    else:
        model = FullMorphModel() 
elif args.model_type == 'conv':
    model = CNNModel()
else:
    model = MCNNModel()

if args.cuda:
    model.cuda()

print(model)

# Map numeric label -> a readable class name (Common Name (Scientific Name)) 
label_to_name = {} 
for _, meta in filename_to_meta.items(): 
    lbl = int(meta['label']) 
    if lbl not in label_to_name: 
        cn = meta.get('common_name') or '' 
        pretty = cn if cn else f"Label {lbl}" 
        label_to_name[lbl] = pretty

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
    total_class_loss = 0.0
    total_diversity_loss = 0.0

    target_total = np.array([], dtype=int)
    output_total = np.array([], dtype=int)
    fm_dict = {"0": [], "1": []}

    for batch_idx, (img, label, _, _) in enumerate(train_loader):
        device = torch.device("cuda" if args.cuda else "cpu")

        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        # print(img.shape, label)
        # Only log filters for the first batch
        if (batch_idx == 0):
            model.log_filters(True)
            pred_multi, emb = model(img, epoch, experiment)
        else:
            model.log_filters(False)
            pred_multi, emb = model(img, epoch, experiment)

        # Build a mixed embedding: POS = live graph, NEG = detached (no encoder grad)
        pos_mask = (label == 1).unsqueeze(1)          # [B, 1] bool
        emb_neg_free = torch.where(pos_mask, emb, emb.detach())  # [B, D]

        # Recompute logits only through the head so FC updates on all samples
        logits = model.full_context.head(emb_neg_free)

        loss = criterion_cls(logits, label)

        # Compute loss and set model parameters
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # total_class_loss += class_loss
        total_class_loss += loss
        # total_diversity_loss += diversity_loss

        mean_pred = 0.5 * (pred_multi + pred_context)

        # Store predictions for confusion matrix
        target_total = np.concatenate([target_total, label.cpu().detach().numpy()], axis=None)
        output_total = np.concatenate([output_total, (pred_multi).argmax(dim=1).cpu().detach().numpy()], axis=None)

        # Store feature map values for histogram
        # indices_0 = (label == 0).nonzero()[0]
        # indices_1 = (label == 1).nonzero()[0]
        # fm_val = emb_a.detach().cpu().numpy()
        # print(emb_a.shape, emb_p.shape, emb_n.shape)

        # fm_dict["0"].append(fm_val[indices_0])
        # fm_dict["1"].append(fm_val[indices_1])

    if args.use_comet:
        # plot_fm_histogram(fm_dict, experiment, epoch)

        experiment.log_confusion_matrix(
                y_true=target_total,
                y_predicted=output_total,
                labels=["Not JAW", "JAW"],
                title="Train Set",
                file_name="train.json"
            )
        
        experiment.log_metric('Loss', value=(total_loss / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Diversity Loss', value=(total_diversity_loss / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Classification Loss', value=(total_class_loss / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Triplet Loss', value=(total_triplet / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Context Classifier Loss', value=(total_context_class / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
        experiment.log_metric('Multi-Layer Classifier Loss', value=(total_multi_class / len(train_loader.dataset)) * args.batch_size, epoch=epoch)
    
    stop_train = time()
    log_embedding_histograms(epoch, train_loader, split='train')
    log_embedding_histograms(epoch, test_loader, split='test')
    print('==== Training Time ====', str(stop_train-start_train))

# After making test_filenames
test_label_ids = sorted({filename_to_meta[f]['label'] for f in test_filenames})
label_to_row = {lbl: i for i, lbl in enumerate(test_label_ids)}

def test(epoch):
    start_test = time()
    model.eval()
    device = torch.device("cuda" if args.cuda else "cpu")

    total, correct = 0, 0
    total_loss = 0.0

    # rows = original numeric labels present in TEST; cols = [0=Not JAW, 1=JAW]
    heatmap_data = np.zeros((len(test_label_ids), 2), dtype=int)

    # For Comet confusion matrix (binary)
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for data, bin_target, orig_labels, _ in test_loader:
            data = data.to(device)
            bin_target = bin_target.to(device)  # already 0/1 from dataset

            # Forward
            pred_multi, _ = model(data, epoch, experiment)  # must output shape [N, 2]

            loss = F.nll_loss(pred_multi, bin_target, reduction='sum')

            total_loss += loss.item()

            preds = pred_multi.argmax(dim=1)

            correct += (preds == bin_target).sum().item()
            total   += bin_target.size(0)

            # For Comet's binary confusion matrix
            y_true_all.extend(bin_target.detach().cpu().tolist())
            y_pred_all.extend(preds.detach().cpu().tolist())

            if epoch % 10 == 0:
                # Update heatmap by ORIGINAL numeric label (row) vs predicted binary (col)
                if isinstance(orig_labels, torch.Tensor):
                    ol_np = orig_labels.detach().cpu().numpy()
                else:
                    # list/tuple -> numpy
                    ol_np = np.asarray(orig_labels)
                pr_np = preds.detach().cpu().numpy()

                for ol, pr in zip(ol_np, pr_np):
                    r = label_to_row.get(int(ol))
                    if r is not None and pr in (0, 1):
                        heatmap_data[r, pr] += 1

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    print(f'\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc*100:.0f}%)\n')

    if args.use_comet:
        experiment.log_metric('test_loss', avg_loss, epoch=epoch)
        experiment.log_metric('test_acc', acc, epoch=epoch)
        experiment.log_confusion_matrix(
            y_true=y_true_all,
            y_predicted=y_pred_all,
            labels=["Not Target", "Target"],
            title="Test Set (Binary)",
            file_name="test.json"
        )

        if epoch % 10 == 0:
            # Numeric labels on Y-axis
            plot_heatmap(heatmap_data, experiment, epoch)

    print('==== Test Cycle Time ====\n', str(time() - start_test))
    return correct

# Training loop
accuracy = torch.zeros(args.epochs+1)
for epoch in range(1,args.epochs+1):
    train(epoch)
    accuracy[epoch] = test(epoch)
    if args.use_comet:
        experiment.log_metric("Accuracy", accuracy[epoch] / 100, epoch)
        experiment.log_metrics(log_weights(model))

if args.model_filename is not None:
    torch.save(model, args.model_filename)

accuracy /= 100
print(accuracy.max(0))
stop_whole = time()
print('==== Whole Time ====', str(stop_whole-start_whole))

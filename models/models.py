import torch
import torch.nn as nn
from models.MNN import *
from utils.logger import *
from utils.plot import *
from typing import Optional
import torch.nn.functional as F

class MorphEncoder(nn.Module):
    """Produces a 10-D feature vector for triplet loss."""
    def __init__(self, filter_list: Optional[list] = None):
        super().__init__()
        self.backbone = MorphNet(filter_list)
        self.flatten = nn.Flatten()          # keeps the feature size at (B, 10)

    def forward(self, x, epoch: int | None = None, experiment=None):
        feat_map, hit, miss = self.backbone(x, epoch, experiment)
        embedding = self.flatten(feat_map)   # shape (B, 10)
        return embedding, hit, miss


class MorphHead(nn.Module):
    """Maps a 10-D embedding to class logits."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(1, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, embedding):
        return F.log_softmax(self.fc(self.act(embedding)), dim=1)


class MorphTripletModel(nn.Module):
    """
    If only `anchor` is given: returns class logits.
    If `positive` and `negative` are also passed: returns three embeddings
    (no classifier applied) so you can compute TripletMarginLoss.
    """
    def __init__(self, num_classes: int = 2, filter_list: Optional[list] = None):
        super().__init__()
        self.encoder = MorphEncoder(filter_list)
        self.head     = MorphHead(num_classes)

    def forward(
        self,
        anchor,
        positive=None,
        negative=None,
        epoch: int | None = None,
        experiment=None
    ):
        if positive is None or negative is None:
            emb, hit, miss = self.encoder(anchor, epoch, experiment)
            logits = self.head(emb)
            return logits, emb, hit, miss

        # Triplet path
        anc_emb, *_ = self.encoder(anchor,   epoch, experiment)
        pos_emb, *_ = self.encoder(positive, epoch, experiment)
        neg_emb, *_ = self.encoder(negative, epoch, experiment)
        return anc_emb, pos_emb, neg_emb
    
    def log_filters(self, t_or_f):
        self.encoder.backbone.log_filters(t_or_f)

class MorphNet(nn.Module):
    def __init__(self, filter_list=None):
        super(MorphNet,self).__init__()
        if (filter_list):
            self.MNN1 = MNN(1,10,28, filter_list)
        else:
            self.MNN1 = MNN(1,10,28)
        self.training = True
        self.passes = 0
        self._log_filters = False

    def log_filters(self, t_or_f):
        self._log_filters = t_or_f
    
    def forward(self, x, epoch, experiment):
        output = x
        output, hit, miss = self.MNN1(output)

        # Plot filters
        if self._log_filters and experiment:
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
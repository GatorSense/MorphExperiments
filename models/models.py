import torch
import torch.nn as nn
from models.MNN import *
from utils.logger import *
from utils.plot import *
from typing import Optional
import torch.nn.functional as F


class FullMorphModel(nn.Module):
    def __init__(self, filter_list=None):
        super().__init__()
        # self.multi_layer = MultiLayerMorph()
        if filter_list == None:
            self.full_context = MorphTripletModel()
        else:
            self.full_context = MorphTripletModel(filter_list=filter_list)

    def forward(self, img, epoch, experiment):
        # pred_multi = self.multi_layer(a)
        pred_context, emb = self.full_context(img, epoch=epoch, experiment=experiment)
        # pred_triplet = self.full_context(a, p, n, epoch=epoch, experiment=experiment)
        return pred_context, emb
    
    def log_filters(self, t_or_f):
        self.full_context.encoder.backbone.log_filters(t_or_f)


# class MultiLayerMorph(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.MNN1 = MNN(1, 10, 3)
#         self.MNN2 = MNN(10, 5, 3)
#         self.fc1 = nn.Linear(2880, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, 2)
#         self.activation = nn.LeakyReLU(negative_slope=0.1)

#     def forward(self, x):
#         x = self.MNN1(x)
#         x = self.MNN2(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.activation(self.fc1(x))
#         x = self.activation(self.fc2(x))
#         x = self.fc3(x)
#         return F.log_softmax(x, dim=1)
    

class MultiLayerMorph(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN1 = nn.Conv2d(3, 10, 3)
        self.CNN2 = nn.Conv2d(10, 5, 3)
        self.fc1 = nn.Linear(3920, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = x.reshape(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class MorphEncoder(nn.Module):
    """Produces a 10-D feature vector for triplet loss."""
    def __init__(self, filter_list: Optional[list] = None):
        super().__init__()
        self.backbone = MorphNet(filter_list)
        self.flatten = nn.Flatten()          # keeps the feature size at (B, 10)

    def forward(self, x, epoch: int | None = None, experiment=None):
        feat_map = self.backbone(x, epoch, experiment)
        embedding = self.flatten(feat_map)   # shape (B, 10)
        return embedding


class MorphHead(nn.Module):
    """Maps a 10-D embedding to class logits."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)
        self.act = nn.LeakyReLU()

    def forward(self, embedding):
        return F.log_softmax(self.fc(embedding), dim=1)


class MorphTripletModel(nn.Module):
    """
    If only `anchor` is given: returns class logits.
    If `positive` and `negative` are also passed: returns three embeddings
    (no classifier applied) so you can compute TripletMarginLoss.
    """
    def __init__(self, num_classes: int = 2, filter_list: Optional[torch.Tensor] = None):
        super().__init__()
        self.encoder = MorphEncoder(filter_list)
        self.head    = MorphHead(num_classes)

    def forward(
        self,
        anchor,
        positive=None,
        negative=None,
        epoch: int | None=None,
        experiment=None
    ):
        if positive is None or negative is None:
            emb = self.encoder(anchor, epoch, experiment)
            logits = self.head(emb)
            return logits, emb

        # Triplet path
        anc_emb = self.encoder(anchor, epoch, experiment)
        pos_emb = self.encoder(positive, epoch, experiment)
        neg_emb = self.encoder(negative, epoch, experiment)

        return anc_emb, pos_emb, neg_emb
    

class MorphNet(nn.Module):
    def __init__(self, filter_list=None):
        super(MorphNet,self).__init__()
        if filter_list != None:
            self.MNN1 = MNN(2048,10,kernel_size=16, filter_list=filter_list)
        else:
            self.MNN1 = MNN(2048,10, kernel_size=16)
        self.training = True
        self.passes = 0
        self._log_filters = False

    def log_filters(self, t_or_f):
        self._log_filters = t_or_f
    
    def forward(self, x, epoch, experiment):
        output = x
        output = self.MNN1(output)

        # if self._log_filters and experiment:
        #     plot_morph_filters_forward(self.MNN1.K_hit, experiment, epoch, "hit")
        #     plot_morph_filters_forward(self.MNN1.K_miss, experiment, epoch, "miss")

        return output

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
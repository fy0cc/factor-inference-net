import torch
import torch.nn as nn

from .mlp import MLP

NDIM_HIDDEN = 64


class SigmoidWrapper(nn.Module):
    def __init__(self, model):
        super(SigmoidWrapper, self).__init__()
        self.model = model
        self.m = nn.Sigmoid()

    def forward(self, *args, **kwargs):
        return self.m(self.model(*args, **kwargs))


class GGNNReadoutLayer(nn.Module):
    def __init__(self, nhid, nout):
        super(GGNNReadoutLayer, self).__init__()
        self.readout1 = MLP(2*nhid, NDIM_HIDDEN, nout, n_hidden=2)
        self.readout2 = MLP(nhid, NDIM_HIDDEN, nout, n_hidden=2)

    def forward(self, h0, ht):
        return torch.sigmoid(self.readout1(torch.cat([ht, h0], dim=-1)))*self.readout2(ht)

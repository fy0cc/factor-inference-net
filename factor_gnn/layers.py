import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import GRUCell
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from .mlp import MLP

NDIM_HIDDEN = 64
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple, Union


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


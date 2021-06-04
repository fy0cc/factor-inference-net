import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(
            self,
            ndim_in: int,
            ndim_hidden: int,
            ndim_out: int,
            n_hidden: int = 1,
            nonlinearity=nn.SELU,
            output_nonlinearity=None,
            dropout: int = False,
            dropout_p: float = 0.3,
            use_bn: bool = False,
    ):
        super(MLP, self).__init__()
        self.ndim_in = ndim_in
        self.ndim_hidden = ndim_hidden
        self.ndim_out = ndim_out

        self.input = nn.Linear(ndim_in, ndim_hidden, bias=True)
        self.hidden = nn.ModuleList()
        self.n_hidden = n_hidden
        self.nonlinearity = nonlinearity()
        self.dropout_p = dropout_p
        self.dropout = dropout

        self.output_nonlinearity = output_nonlinearity
        self.bn0 = nn.BatchNorm1d(ndim_hidden)
        self.bnlist = nn.ModuleList()
        self.use_bn = use_bn
        for i in range(self.n_hidden):
            self.hidden.append(nn.Linear(ndim_hidden, ndim_hidden, bias=True))
            if use_bn:
                self.bnlist.append(nn.BatchNorm1d(ndim_hidden))

        self.output = nn.Linear(ndim_hidden, ndim_out, bias=True)
        self._init_weights_kaiming()
        # self._init_weights_zero()
        # self._init_weights_xavier()

    def _init_weights_kaiming(self):
        nn.init.kaiming_normal_(self.input.weight.data,
                                nonlinearity="leaky_relu")
        nn.init.zeros_(self.input.bias.data)

        nn.init.kaiming_normal_(self.output.weight.data,
                                nonlinearity="leaky_relu")
        nn.init.zeros_(self.output.bias.data)
        for i in range(self.n_hidden):
            nn.init.kaiming_normal_(
                self.hidden[i].weight.data, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(self.hidden[i].bias.data)

    def _init_weights_xavier(self):
        nn.init.xavier_normal_(self.input.weight.data)
        nn.init.zeros_(self.input.bias.data)

        nn.init.xavier_normal_(self.output.weight.data)
        nn.init.zeros_(self.output.bias.data)
        for i in range(self.n_hidden):
            nn.init.xavier_normal_(self.hidden[i].weight.data)
            nn.init.zeros_(self.hidden[i].bias.data)

    def _init_weights_zero(self):
        nn.init.zeros_(self.input.weight.data)
        nn.init.zeros_(self.input.bias.data)

        nn.init.zeros_(self.output.weight.data)
        nn.init.zeros_(self.output.bias.data)
        for i in range(self.n_hidden):
            nn.init.zeros_(self.hidden[i].weight.data)
            nn.init.zeros_(self.hidden[i].bias.data)

    def forward(self, x: Tensor):
        x = self.input(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.nonlinearity(x)

        for i in range(self.n_hidden):
            x = self.hidden[i](x)
            if self.use_bn:
                x = self.bnlist[i](x)
            x = self.nonlinearity(x)

        x = self.output(x)
        if self.output_nonlinearity is not None:
            x = self.output_nonlinearity(x)
        return x

import os
from os.path import join
import glob
import time
import random
import pickle
import argparse
import json
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch_geometric
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_max
import sys
import shutil
import scipy.stats as ss
import incense
import itertools

sys.path.append(os.getcwd())

from factor_gnn.layers import *
from factor_gnn.models import *


def build_model(args, dataset):
    if 'bipartite' in dataset["datatype"]:
        if args["model"]=="multigatconvmemory":
            model = MultiGATConvGRU(
                dataset["nnode1"],
                dataset["nnode2"],
                None,
                dataset["ntarget"],
                dataset["total_factor_types"],
                nhid=args["nhid"],
                nstate=args["nstate"],
                rnn_method=args["rnn_method"],
                init_method=args["init_method"],
                aggregation_method=args["aggregation_method"],
                decode_method=args["decode_method"],
                decode_series_len=args["decode_series_len"],
                damping=args["damping"],
                asrnn_kwargs={
                    "gamma":args["asrnn_gamma"],
                    "epsilon":args["asrnn_epsilon"]
                }
            )
        else:
            raise NotImplementedError("not implemented")
    else:
        raise NotImplementedError("not implemented")
    if args["cuda"]:
        model.cuda()
    return model


def test(model, loader, criterion, forward_kwargs={}, output_dynamics=False):
    model_device = next(model.parameters()).device
    model.eval()
    loss_epoch = 0.0
    counter = 0.0
    ylist = []
    yhatlist = []
    dyn_list = []
    for (ibatch, g) in enumerate(loader):
        g = g.to(model_device)
        ylist.append(g.y.detach().cpu())
        if not output_dynamics:
            yhat = model.forward(g, **forward_kwargs)
        else:
            yhat, dyn = model.forward(g, output_dynamics=output_dynamics, **forward_kwargs)
            # convert tensor to cpu and detach
            for i, ll in enumerate(dyn):
                dyn[i] = [t.detach().cpu() for t in ll]
            for i, ll in enumerate(dyn):
                dyn[i] = torch.stack(ll, dim=0)  # (T,N,N_HIDDEM) for each element of dyn_lis
            dyn_list.append(dyn)

        yhatlist.append(yhat.detach().cpu())
        if criterion:
            loss_batch = criterion(g.y, yhat)*g.y.shape[0]
            counter += g.y.shape[0]
            loss_epoch += loss_batch.item()
    if criterion:
        test_loss = loss_epoch/counter
    else:
        test_loss = None
    if not output_dynamics:
        return test_loss, ylist, yhatlist
    else:
        return test_loss, ylist, yhatlist, dyn_list


def train(model, loader, optimizer, criterion, forward_kwargs={}, output_dynamics=False):
    model_device = next(model.parameters()).device
    model.train()
    loss_epoch = 0.0
    counter = 0
    for (ibatch, g) in enumerate(loader):
        # print('batch data dimensions')
        # print(g)
        g = g.to(model_device)
        yhat = model.forward(g, output_dynamics=output_dynamics, **forward_kwargs)
        loss_batch = criterion(g.y, yhat)*g.y.shape[0]
        counter += g.y.shape[0]
        loss_epoch += loss_batch.item()
        # print("batch loss:{}".format(loss_batch.item()))
        optimizer.zero_grad()
        loss_batch.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(),0.25)
        optimizer.step()
    sample = loader.dataset[0]
    train_loss = loss_epoch/counter

    return train_loss


def load_best_model(ex, args=None, dataset=None):
    if args is None:
        args = ex.config.args
    if dataset is None:
        dataset = ex.config.dataset
    model = build_model(args, dataset)
    keys = list(
        filter(
            lambda k:re.search(r"model.best.pt", k) is not None, ex.artifacts.keys()
        )
    )
    best_model_key = "model.best.pt"
    state_dict = (
        ex.artifacts[best_model_key].as_type(incense.artifact.PickleArtifact).render()
    )
    model.load_state_dict(state_dict)
    return model


def kl_mvn(P1, P2):
    """
    KL divergence between two multi-variate normal distributions
    """
    assert type(P1)==type(P2), "types of two inputs should match"
    if isinstance(P1, torch.Tensor):
        P1 = P1.detach().cpu().numpy()
        P2 = P2.detach().cpu().numpy()
    if (
            isinstance(P1, float)
            or isinstance(P1, np.float64)
            or isinstance(P1, np.float32)
    ):
        P1 = np.reshape(np.asarray([P1]), (1, 1))
        P2 = np.reshape(np.asarray([P2]), (1, 1))
        if P1 <= 0 or P2 <= 0:
            return 0.0
    assert P1.shape==P2.shape, "sizes of two inputs should match"
    assert len(P1.shape)==2
    cov1 = np.linalg.inv(P1)
    cov2 = np.linalg.inv(P2)
    return 0.5*(
            np.log(np.linalg.det(cov2)/np.linalg.det(cov1))
            - cov1.shape[0]
            + np.sum(np.diag(np.matmul(P2, cov1)))
    )


def undirected2factorgraph(d):
    """
    Converting a `torch_geometric.data.Data` object representing and undirected graph to a factor graph with singleton and pairwise factors
    """

    nvar = d.x.shape[0]
    nedge = d.edge_attr.shape[0]
    dedge_index = d.edge_index.t().tolist()
    # factor attributes
    ndim_f = max(d.edge_attr.shape[-1], d.x.shape[-1])
    xf = torch.zeros(nvar + nedge, ndim_f)
    xf[0:nedge, 0: d.edge_attr.shape[-1]] = d.edge_attr
    xf[nedge:, 0: d.x.shape[-1]] = d.x
    # edge_index
    edge_index = []
    for i in range(nedge):
        edge_index.append([i, dedge_index[i][0]])
        edge_index.append([i, dedge_index[i][1]])
    for i in range(nvar):
        edge_index.append([i + nedge, i])
    # factor_type
    factor_type = torch.tensor([2.0]*nedge + [1.0]*nvar, dtype=torch.float)
    xf = torch.cat(
        [factor_type.view(-1, 1), xf], dim=-1
    )  # first dimention of xf is the factor type
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data = Data(edge_index=edge_index)
    data.x1 = torch.clone(xf.detach())
    data.x2 = torch.clone(d.x.detach())
    data.y = torch.clone(d.y.detach())
    data.num_factor_types = 2
    return data


def undirected2factorgraph2(d):
    """
    Converting a `torch_geometric.data.Data` object representing and undirected graph to a factor graph with singleton and pairwise factors
    no singleton factor in this case
    """

    nvar = d.x.shape[0]
    nedge = d.edge_attr.shape[0]
    dedge_index = d.edge_index.t().tolist()
    # factor attributes
    xf = d.edge_attr.clone()

    # edge_index
    edge_index = []
    for i in range(nedge):
        edge_index.append([i, dedge_index[i][0]])
        edge_index.append([i, dedge_index[i][1]])
    # factor_type
    factor_type = torch.tensor([2.0]*nedge, dtype=torch.float)
    xf = torch.cat(
        [factor_type.view(-1, 1), xf], dim=-1
    )  # first dimention of xf is the factor type
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data = Data(edge_index=edge_index)
    data.x1 = torch.clone(xf.detach())
    data.x2 = torch.clone(d.x.detach())
    data.y = torch.clone(d.y.detach())
    data.num_factor_types = 1
    return data


def format_qm9_with_pos(d):
    """
    Converting a `torch_geometric.data.Data` object representing and undirected graph to a factor graph with singleton and pairwise factors
    no singleton factor in this case
    """

    nvar = d.x.shape[0]
    nedge = d.edge_attr.shape[0]
    dedge_index = d.edge_index.t().tolist()
    # factor attributes
    xf = d.edge_attr.clone()

    # edge_index
    edge_index = []
    for i in range(nedge):
        edge_index.append([i, dedge_index[i][0]])
        edge_index.append([i, dedge_index[i][1]])
    # factor_type
    factor_type = torch.tensor([2.0]*nedge, dtype=torch.float)
    xf = torch.cat(
        [factor_type.view(-1, 1), xf], dim=-1
    )  # first dimention of xf is the factor type
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    data = Data(edge_index=edge_index)
    data.x1 = torch.clone(xf.detach())
    x2 = torch.cat([d.x.detach(), d.pos.detach()], dim=-1)
    data.x2 = x2
    data.y = torch.clone(d.y.detach())
    data.num_factor_types = 1
    return data


def plot_graph_mask(g):
    if not isinstance(g, Batch):
        g = Batch.from_data_list([g])
    g = augment_permutations(g, 3)
    # original graph
    B = nx.Graph()
    B.add_nodes_from(range(g.x1indices.item()), bipartite=0)
    # B.add_nodes_from(range(g.x1indices.item()), bipartite=0)
    B.add_nodes_from(map(str, range(g.x2indices.item())), bipartite=1)
    idx = g.edge_index.numpy().T.tolist()
    idx = list(map(lambda x:[x[0], str(x[1])], idx))
    B.add_edges_from(idx)
    G = B
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)

    # 
    idx_mask = g.edge_index2[0, :, :].numpy().T.tolist()
    idx_mask = list(map(lambda x:[x[0], str(x[1])], idx_mask))
    fig, ax = plt.subplots(1)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=G.nodes)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=idx, edge_color='b', alpha=0.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=idx_mask, edge_color='r', alpha=1.0, width=2)
    plt.show(fig)


def bgraph_to_networkx(g):
    """ Convert torch_geometric graph to networkx"""
    if not isinstance(g, Batch):
        g = Batch.from_data_list([g])
    B = nx.Graph()
    B.add_nodes_from(range(g.x1indices.item()), bipartite=0)
    B.add_nodes_from(map(str, range(g.x2indices.item())), bipartite=1)
    idx = g.edge_index.numpy().T.tolist()
    idx = map(lambda x:[x[0], str(x[1])], idx)
    B.add_edges_from(idx)
    return B


def edgeindex_to_mat(index, n1, n2, ordering):
    M = torch.zeros(n1, n2)
    for i in range(index.size(1)):
        a, b = index[:, i]
        M[ordering[a], b] = 1
    return M


def augment_permutations(g, k: int):
    """
    Modify graph `g` for graph AR normalizing flow
    
    Args:
      g has the following attributes
        x1: (num_node1, node_channels1) matrix of variable node attributes 
        x2: (num_node2, node_channels2) matrix of factor node attributes
        edge_index: (2,num_edge)
      k (int): number of layers 
    
    Returns:
      g1 with following attributes
        x1: (num_node1, node_channels1) matrix of variable node attributes 
        x2: (num_node2, node_channels2) matrix of factor node attributes
        edge_index: original edge indices
        edge_index1: (num_layer ,2,num_edge1)
        edge_index2: (num_layer, 2,num_edge2)
    """
    if not isinstance(g, Batch):
        g = Batch([g])

    def _mask_from_order(g, ordering):
        num_factor = torch.max(g.edge_index[1, :]) + 1  # assume numbering of factor start from 0
        edge_index2 = torch.LongTensor(2, num_factor)
        edge_index2[1, :] = torch.arange(num_factor)
        edge_index2[0, :] = torch.zeros(num_factor)
        ordering_rev = len(ordering) - 1 - ordering  # reverse the order
        edge_index2_a = g.edge_index[0, scatter_max(ordering[g.edge_index[0]], g.edge_index[1])[1]]
        edge_index2[0] = edge_index2_a
        return edge_index2, ordering

    def _random_index(g):
        ordering = torch.randperm(g.x1.size(0))  # random permutation
        ordering_rev = len(ordering) - 1 - ordering  # reverse the order
        return [_mask_from_order(g, ordering), _mask_from_order(g, ordering_rev)]

    g_new = g.clone()
    g_new.edge_index1 = g.edge_index.clone().unsqueeze(0).expand(k, -1, -1)
    results = [
        _random_index(g) for _ in range((k + 1)//2)
    ]
    results = list(itertools.chain.from_iterable(results))
    g_new['edge_index2'] = torch.stack([results[i][0] for i in range(k)])
    g_new['ordering'] = torch.stack([results[i][1] for i in range(k)])

    if torch_geometric.is_debug_enabled():
        g_new.debug()
        return g_new.contiguous()
    return g_new

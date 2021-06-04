import glob
import os
import pickle
from typing import Union

import deepdish as dd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

Array = Union[Tensor, np.ndarray]


def format_ggm_data(P: Array, adj: Array, marginalP: Array, target: str = "precision"):
    """
    x: node feature matrix, local precision here, [num_nodes, num_node_feat]
    edge_index: [2,num_edges]
    edge_attr: edge feature matrix, precision here, [num_edges,num_edge_feat]
    y: target to train 
    """
    num_node, _ = P.shape
    x = torch.tensor(np.diag(P), dtype=torch.float32).view(num_node, 1)

    # create edge index list without self-loop
    edge_index = []
    edge_attr = []
    for i in range(num_node):
        for j in range(num_node):
            if adj[i, j] and i!=j:
                edge_index.append([i, j])
                edge_attr.append(P[i, j])
    num_edge = len(edge_attr)
    edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(num_edge, 1)
    assert tuple(edge_index.shape)==(2, num_edge), "size incorrect"
    assert tuple(edge_attr.shape)==(num_edge, 1), "size incorrect"
    if target=="precision":
        y = torch.tensor(marginalP, dtype=torch.float32).view(num_node, 1)
    elif target=="covariance":
        y = torch.tensor(1.0/marginalP, dtype=torch.float32).view(num_node, 1)
    else:
        raise NotImplementedError("targe should be precision/covariance")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


def format_thirdorder_bipartite(data_entry: dict, singleton_factor: bool = False):
    """
    input:
        data_entry (dict): has keys
            nfactor, nnode, edge_index,edge_attr,edte_type

    output:
        data (Data): has attributes:
            x1(factor),x2(variable),edge_index(factor to variable),y(target)
    
    """
    # print(data_entry)
    dic, y = data_entry
    nf = dic["nfactor"]
    edge_index = []
    counter = 0
    for i in range(dic["nfactor"]):
        for j in dic["edge_index"][i]:
            edge_index.append([i, j - 1])  # In Julia, index start from 1
    # add singleton factor to the end of list
    if singleton_factor:
        for i in range(dic["nnode"]):
            edge_index.append([dic["nfactor"] + i, i])
    xv = np.stack(dic["node_attr"], axis=0)
    xv = torch.tensor(xv, dtype=torch.float32)
    ndim_xv = xv.shape[-1]
    if nf > 0:
        xf = np.stack(dic["edge_attr"], axis=0)
        xf = torch.tensor(xf, dtype=torch.float32)
        ndim_xf = xf.shape[-1]
        factor_types = torch.tensor(dic["edge_type"], dtype=torch.long)
    edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
    if singleton_factor:
        if nf > 0:
            factor_types = torch.cat([factor_types, torch.ones(dic["nnode"], dtype=torch.long)])
        else:
            factor_types = torch.ones(dic["nnode"], dtype=torch.long)
    total_types = list(set(factor_types.tolist()))
    factor_types = torch.reshape(factor_types, (-1, 1))
    if singleton_factor:
        # padding potentials
        if nf > 0:
            ndim_common = max([ndim_xv, ndim_xf])
            xf_new = torch.zeros(xf.shape[0] + xv.shape[0], ndim_common)
            xf_new[:xf.shape[0], :ndim_xf] = xf
            xf_new[xf.shape[0]:, :ndim_xv] = xv
            xf = xf_new
        else:
            ndim_common = ndim_xv
            xf = torch.clone(xv)

    # first dimension of x_factor is the factor type
    xf = torch.cat([factor_types.float(), xf], dim=-1)
    y2 = torch.tensor(y, dtype=torch.float32)
    y1 = torch.zeros(xf.shape[0], 1, dtype=torch.float32)  # dummy target
    data = Data(edge_index=edge_index)
    data.x1 = xf
    data.x2 = xv
    data.y = y2  # training target  
    data.num_factor_types = len(total_types)

    return data


def format_ggm_data_bipartite(P: Array, adj: Array, marginalP: Array, target: str = "precision",
                              singleton_factor: bool = True):
    """
    Args:
        P (matrix): precision matrix of the GGM
        adj (matrix): adjacency matrix of the GGM
        marginalP (vector): marginal precision
        target (str): use marginal precision or variance as training target
        singleton_factor (bool): whether to use singleton factor
    Outputs:
        x1: factor node features, natural parameter of pairwise interaction
        x2: variable node features, [bias, precision]
        edge_index: [2,num_edges]
        y1: zero
        y2: marginal precision 
    """
    # calculate log partition function
    U = torch.cholesky(torch.tensor(P))
    b = P.shape[0]
    logZ = 0.5*b*torch.log(torch.tensor(2.0*3.141592653589793)) \
           - torch.log(torch.diagonal(U, dim1=-1, dim2=-2)).sum()
    logZ = logZ.view(1)
    num_node, _ = P.shape
    xv = torch.tensor(np.diag(P), dtype=torch.float32).view(num_node, 1)
    xv = torch.cat([torch.zeros_like(xv), xv], dim=-1)  #

    # create edge index list without self-loop
    edge_index = []
    factor_types = []
    xf = []
    counter = 0  # for factor node
    for i in range(num_node):
        # pairwise factor
        for j in range(i + 1, num_node):
            if adj[i, j]:
                edge_index.append([counter, j])
                edge_index.append([counter, i])
                xf.append([P[i, j]])
                factor_types.append([0])
                counter += 1
        # singleton factor
        if singleton_factor==True:
            edge_index.append([counter, i])
            factor_types.append([1])
            xf.append([P[i, i]])
            counter += 1

    num_edge = len(edge_index)
    edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
    num_fnode = counter
    xf = torch.cat([torch.tensor(factor_types), torch.tensor(xf)], dim=-1)
    assert tuple(edge_index.shape)==(2, num_edge), "size incorrect"
    # assert tuple(xf.shape) == (num_fnode, 1), "size incorrect"
    assert tuple(xv.shape)==(num_node, 2), "size incorrect"
    if target=="precision":
        y2 = torch.tensor(marginalP, dtype=torch.float32).view(num_node, 1)
    elif target=="covariance":
        y2 = torch.tensor(1.0/marginalP, dtype=torch.float32).view(num_node, 1)
    else:
        raise NotImplementedError("targe should be precision/covariance")
    y1 = torch.zeros(xf.shape[0], 1, dtype=torch.float32)
    if singleton_factor:
        num_ftypes = 2
    else:
        num_ftypes = 1
    data = Data(x1=xf.float(), x2=xv.float(), edge_index=edge_index, y=y2.float(), num_factor_types=num_ftypes,
                logZ=logZ)
    return data


class GGMDataset(InMemoryDataset):
    def __init__(self, root: str, transform=None, pre_transform=None, target: str = "precision"):
        super(GGMDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.target = target
        # self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return ["data_undirected.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        P, adj, marginalP = pickle.load(open(data_filepath, "rb"))
        if not P.flags['WRITEABLE']:
            P.flags['WRITEABLE'] = True
            adj.flags['WRITEABLE'] = True
            marginalP.flags['WRITEABLE'] = True
            # P = np.copy(P)
            # adj = np.copy(adj)
            # marginalP = np.copy(marginalP)
        ndata, nnode, _ = P.shape

        data_list = [
            format_ggm_data(
                P[i, :, :], adj[i, :, :], marginalP[i, :], target=self.target
            )
            for i in range(ndata)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


class GGMDatasetBipartite(InMemoryDataset):
    def __init__(
            self, root, transform=None, pre_transform=None,
            target="precision",
            singleton_factor=False,
            refresh=True
    ):
        self.target = target
        self.singleton_factor = singleton_factor
        if refresh:
            old_filename = os.path.join(root, "processed", self.processed_file_names[0])
            if os.path.isfile(old_filename):
                print(f"Removing {old_filename}")
                os.remove(old_filename)

        super(GGMDatasetBipartite, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["data.pt"]

    @property
    def processed_file_names(self):
        return ["data_bipartite.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_rawdata(self):
        data_filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(data_filepath, "rb") as file:
            res = pickle.load(file)
        return res

    def process(self):
        # Read data into huge `Data` list.
        data_filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(data_filepath, "rb") as file:
            P, adj, marginalP = pickle.load(file)
        ndata, nnode, _ = P.shape

        data_list = [
            format_ggm_data_bipartite(
                P[i, :, :], adj[i, :, :], marginalP[i, :],
                target=self.target,
                singleton_factor=self.singleton_factor
            )
            for i in range(ndata)
        ]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


class ThirdOrderDatasetBipartite(InMemoryDataset):
    def __init__(self,
                 root: str,
                 transform=None,
                 pre_transform=None,
                 normalize_target: bool = False,
                 singleton_factor: bool = False
                 ):
        self.normalize_target = normalize_target
        self.singleton_factor = singleton_factor
        if self.singleton_factor:
            print("using singleton factor")
        else:
            print("not using singleton factor")
        super(ThirdOrderDatasetBipartite, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # remove processed data after loading
        # os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['data.dd']
        fullpathlist = glob.glob(os.path.join(self.raw_dir, '*'))
        filenames = [os.path.split(f)[-1] for f in fullpathlist]
        return filenames
        # return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ["data_bipartite.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        dataset = []
        for filename in self.raw_file_names:
            if filename.endswith("dd"):
                data_filepath = os.path.join(self.raw_dir, filename)
                dataset += dd.io.load(data_filepath)
        data_list = [format_thirdorder_bipartite(d, singleton_factor=self.singleton_factor) for d in dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # After processing, save to disk
        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

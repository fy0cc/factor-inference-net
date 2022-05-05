import sys
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np
from scipy import stats, sparse
import pickle
import os
import deepdish as dd
import glob
from itertools import combinations
from copy import Error, deepcopy

from torch import Tensor
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple, Union

try:
    print(f"PYTHONPATH: {sys.path}")
    from fgnn.data.ldpc import gen_data_item
    print(f"module fgnn imported")
except ImportError as e:
    print(f"module fgnn not imported")

Array = Union[Tensor, np.ndarray]

def format_gabp_data(P:Array, adj:Array, marginalP:Array, target:str="precision"):
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
            if adj[i, j] and i != j:
                edge_index.append([i, j])
                edge_attr.append(P[i, j])
    num_edge = len(edge_attr)
    edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32).view(num_edge, 1)
    assert tuple(edge_index.shape) == (2, num_edge), "size incorrect"
    assert tuple(edge_attr.shape) == (num_edge, 1), "size incorrect"
    if target == "precision":
        y = torch.tensor(marginalP, dtype=torch.float32).view(num_node, 1)
    elif target == "covariance":
        y = torch.tensor(1.0 / marginalP, dtype=torch.float32).view(num_node, 1)
    else:
        raise NotImplementedError("targe should be precision/covariance")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data


def format_thirdorder_bipartite(data_entry: dict, singleton_factor:bool=False):
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
    logZ = 0.5 * b * torch.log(torch.tensor(2.0 * 3.141592653589793)) \
        - torch.log(torch.diagonal(U, dim1=-1, dim2=-2)).sum()
    logZ = logZ.view(1)
    num_node, _ = P.shape
    xv = torch.tensor(np.diag(P), dtype=torch.float32).view(num_node, 1)
    xv = torch.cat([torch.zeros_like(xv),xv],dim=-1)  # 

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
            xf.append([P[i,i]])
            counter += 1
        
    num_edge = len(edge_index)
    edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
    num_fnode = counter
    xf = torch.cat([torch.tensor(factor_types), torch.tensor(xf)],dim=-1)
    assert tuple(edge_index.shape) == (2, num_edge), "size incorrect"
    # assert tuple(xf.shape) == (num_fnode, 1), "size incorrect"
    assert tuple(xv.shape) == (num_node, 2), "size incorrect"
    if target == "precision":
        y2 = torch.tensor(marginalP, dtype=torch.float32).view(num_node, 1)
    elif target == "covariance":
        y2 = torch.tensor(1.0 / marginalP, dtype=torch.float32).view(num_node, 1)
    else:
        raise NotImplementedError("targe should be precision/covariance")
    y1 = torch.zeros(xf.shape[0], 1, dtype=torch.float32)
    if singleton_factor:
        num_ftypes=2
    else:
        num_ftypes=1
    data = Data(x1=xf.float(), x2=xv.float(), edge_index=edge_index, y=y2.float(), num_factor_types=num_ftypes, logZ = logZ)
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
            old_filename = os.path.join(root,"processed",self.processed_file_names[0])
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
                 root:str,
                 transform=None,
                 pre_transform=None,
                 normalize_target:bool=False,
                 singleton_factor:bool=False
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


class QM9Bipartite(InMemoryDataset):
    chem_accu = torch.tensor(
        [.1, .1, .00158, .00158, .00158, 1.2, .000044, .00158, .00158, .00158, .00158, .05]
    )

    def __init__(self,
                 root:str,
                 transform=None,
                 pre_transform=None
                 )->None:
        super(QM9Bipartite, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['qm9bipartite.pt']

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # read raw data list
        data_filepath = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_list = torch.load(data_filepath)
        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


class ErdosRenyiDataset(InMemoryDataset):
    def __init__(self,
                 root:str="data/ER1/",
                 N: int = 10,
                 p: float = 0.3,
                 size: int = 100,
                 transform=None,
                 pre_transform=None,
                 )->None:
        self.N = N
        self.p = p
        self.size = size
        # os.remove(self.processed_paths[0])
        super(ErdosRenyiDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # remove processed data after loading
        os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['data.dd']
        fullpathlist = glob.glob(os.path.join(self.raw_dir, '*'))
        filenames = [os.path.split(f)[-1] for f in fullpathlist]
        return filenames
        # return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ["data_erdosrenyi.pt"]

    def _generate_one(self):
        # node_indices = torch.range(0,self.N-1)
        x1 = torch.rand(self.N, 2) * 3 + 1.0
        edge_index = []
        ecount = 0
        combs = combinations(range(self.N), 2)
        for (a, b) in list(combs):
            q = torch.rand([1]).item()
            if q < self.p:
                edge_index.append([a, ecount])
                edge_index.append([b, ecount])
                ecount += 1
        # edge_indices = torch.range(0,ecount-1)
        x2 = torch.randn(ecount, 1) / 5
        x2 = torch.cat((torch.ones(ecount, 1), x2), dim=-1)
        edge_index = np.asarray(edge_index)

        edge_index = torch.transpose(torch.Tensor(edge_index).long(), 0, 1)

        data = Data(x1=x1, x2=x2, edge_index=edge_index)
        return data

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # generate data on the fly 
        data_list = [self._generate_one() for _ in range(self.size)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # After processing, save to disk
        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


def sprandsym(n:int, density:float):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs).toarray()
    L = np.tril(X)
    L[np.diag_indices(L.shape[0])] = np.abs(np.random.randn(n) ) + 0.05
    result = L @ L.T
    result = sparse.coo_matrix(result)
    return result


class SparseGaussianDataset(InMemoryDataset):
    def __init__(self,
                 root:str="data/ER1/",
                 N: int = 10,
                 p: float = 0.3,
                 size: int = 100,
                 transform=None,
                 pre_transform=None,
                 )->None:
        self.N = N
        self.p = p
        self.size = size
        # os.remove(self.processed_paths[0])
        super(SparseGaussianDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # remove processed data after loading
        os.remove(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['data.dd']
        fullpathlist = glob.glob(os.path.join(self.raw_dir, '*'))
        filenames = [os.path.split(f)[-1] for f in fullpathlist]
        return filenames
        # return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ["data_gaussian.pt"]

    def _generate_one(self):
        """
        Generate a random Gaussian graphical model
        """
        prec_sp = sprandsym(self.N, self.p)
        precision = torch.Tensor(prec_sp.toarray()) + torch.eye(self.N) * 0.1
        assert np.all(np.linalg.eigvals(precision)>0), "precision matrix not pos-def "
        # bias = torch.randn(self.N)
        bias = torch.zeros(self.N)
        x2 = torch.zeros(2, self.N) # (bias, variance)
        x2[0] = bias * 0.5
        x2[1] = torch.diag(precision) * 0.5
        x2 = torch.transpose(x2,0,1)


        # off-diagonal elements

        x1list = []
        edge_index = []
        ecount = 0
        # pairwise factor
        for i in range(prec_sp.size):
            a = prec_sp.row[i]
            b = prec_sp.col[i]
            if a >= b:
                continue
            edge_index.append([ecount,a])
            edge_index.append([ecount,b])
            ecount += 1
            x1list.append([0.0,0.0, prec_sp.data[i]])
            # x2[i] = 2 * prec_sp.data[i]
        # singleton factor 
        for i in range(self.N): 
            edge_index.append([ecount,i])
            ecount += 1
            x1list.append([0.5 * bias[i], 0.5 * precision[i,i],0.0])

        x1 = torch.Tensor(x1list)
        ftypes = torch.ones(ecount)
        ftypes[-self.N:] = 0.0 
        x1 = torch.cat((ftypes.view(-1,1), x1), dim=-1)
        edge_index = np.asarray(edge_index)

        edge_index = torch.transpose(torch.Tensor(edge_index).long(), 0, 1)

        data = Data(x1=x1, x2=x2, edge_index=edge_index, bias=torch.Tensor(bias),precision=torch.Tensor(precision))
        return data

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # generate data on the fly
        data_list = [self._generate_one() for _ in range(self.size)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # After processing, save to disk
        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


class LDPCDatasetBasic(Dataset):
    def __init__(self,
        sigma_b=0.0, snr_db=0.0, burst_prob=0.05,
        invrate =2,
        length = 10000,
        schema="96.3.963",
                 ):
        super().__init__()
        self.N = int(schema.strip().split('.')[0]) # number of variables
        self.K = int(schema.strip().split('.')[1])
        self.invrate = invrate
        self.M = int(self.N / invrate)  # number of factors
        self.schema = schema
        self.length = length

        self.sigma_b = sigma_b
        self.snr_db = snr_db
        self.burst_prob = burst_prob

        # fixed graph structure
        self.template = self.construct_graph_structure()
    
    def __len__(self):
        return self.length

        
    def generate_data(self,snr_db, sigma_b, burst_prob):
        # generate a sample with given snr, burst_prob, and snr
        noisy, _ ,signal = gen_data_item(
            snr_db, sigma_b, burst_prob, train=False, schema=self.schema)

        data = deepcopy(self.template)
        x2 = torch.tensor(noisy).view(self.N,1).float()
        # prepend by factor type
        # x1 = torch.cat([torch.zeros(self.M,1), x1], dim=-1)
        data.x2 = x2
        data.x1 = torch.zeros(self.M,2).float()
        data.y = torch.tensor(signal).view(data.y.shape).float()
        return data
    
    def construct_graph_structure(self):
        self.schema_filename = os.path.join(
            os.path.dirname(__file__),
            f"../fgnn/ldpc_codes/{self.schema}/{self.schema}"
        )
        with open(self.schema_filename) as f:
            # skipping
            for i in range(4):
                f.readline()
            for i in range(96):
                f.readline()

            # read node id for factors, index in the file starts from 1
            edge_index = []
            for i in range(48):
                index = map(lambda x: int(x)-1, f.readline().strip().split())
                for j in index:
                    edge_index.append([i,j])
            
            edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t()
            data = Data(edge_index=edge_index)
            # dummy edge and node features
            data.x1 = torch.zeros(self.M,1)   # factor node feature
            data.x2 = torch.zeros(self.N,1)   # variable node feature
            data.num_factor_types = 1
            data.y = torch.ones(self.N,1) # read out the variables
            return data
            
    def __getitem__(self, i):
        return self.generate_data(self.snr_db, self.sigma_b, self.burst_prob)

        
class LDPCFileDataset(LDPCDatasetBasic):
    def __init__(self,
        fname,
        schema="96.3.963",
        invrate =2,
        # length = 10000,
        # snr=None,
        # burst_prob=0.05,
    ):
        super(LDPCFileDataset, self).__init__(
            schema=schema, invrate=invrate
        )
        self.filepath = fname
        data = torch.load(fname) # keys: noizy_sg, gts, snr_dbs, sigma_b
        self.inputs = data['noizy_sg']
        self.labels = data['gts']
        self.snr = data['snr_dbs']
        self.sigma_b = data['sigma_b']
        self.length = len(self.inputs)

    def __getitem__(self, key):
        data = deepcopy(self.template)
        x2 = self.inputs[key,:].view(self.N,1).float()
        # prepend by factor type
        # x2 = torch.cat([torch.zeros(self.N,1), x2], dim=-1)
        data.y = self.labels[key,:].view(data.y.shape).float()
        x1 = torch.zeros(self.M,2).float()
        data.x1 = x1

        # concatenate snr and sigma before x2
        snr_db = self.snr[key,:].view(self.N,1).float()
        sigma_b = self.sigma_b[key].expand(self.N,1).float()
        data.x2 = torch.cat([
            snr_db, sigma_b, x2
        ], dim=-1)
        return data
        
        
        
class LDPCDataset(LDPCDatasetBasic):
    def __init__(self,
        schema="96.3.963",
        invrate =2,
        length = 10000,
        snr=None,
        burst_prob=0.05,
    ):
        super(LDPCDataset, self).__init__(
            schema=schema, invrate=invrate,length=length
        )
        
        self.sigma_b = [0, 1, 2, 3, 4, 5]
        if snr is not None:
            self.snr_db = [snr]
        else:
            self.snr_db = [0, 1, 2, 3, 4]

        self.burst_prob = burst_prob

    def __getitem__(self, key):
        sigma_b = np.random.choice(self.sigma_b) 
        snr_db = np.random.choice(self.snr_db)
        return self.generate_data(snr_db, sigma_b, self.burst_prob)


if __name__ == "__main__":
    rootdir = "data/size10000_n6"
    dset = GGMDatasetBipartite(rootdir)
    print(dset[0])

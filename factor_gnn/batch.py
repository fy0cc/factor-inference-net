"""
Mostly taken from `torch_geometric.data.Batch` with some modifications

"""
import torch
import torch_geometric
from torch_geometric.data import Data

from torch import Tensor
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list: List):
        r"""Constructs a batch object from a python list holding
        bipartite graphs
        """

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys
        special_keys = ["x1", "x2", "y", "edge_index", "x1indices", "x2indices", "logZ"]

        batch = Batch()
        batch.batch = []
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key:[0] for key in keys if key in dir(data_list[0])}
        c1 = 0
        c2 = 0
        batch.data_list = data_list
        # concatenate x1,x2,y1,y2
        # concatenate edge_index and enlarge
        g = data_list[0]
        x1 = g.x1.clone()
        x2 = g.x2.clone()
        if 'y' in dir(g) and isinstance(g.y, torch.Tensor):
            y = g.y.clone()
        if 'logZ' in dir(g) and isinstance(g.logZ, torch.Tensor):
            logZ = torch.cat([g.logZ for g in data_list], dim=0)

        edge_index = g.edge_index.clone()
        c1 += g.x1.shape[0]
        c2 += g.x2.shape[0]
        x1indices = [c1]
        x2indices = [c2]
        for i, g in enumerate(data_list[1:]):
            batch.batch.append(i)
            x1 = torch.cat([x1, g.x1], dim=0)
            x2 = torch.cat([x2, g.x2], dim=0)
            if 'y' in dir(g) and isinstance(g.y, torch.Tensor):
                y = torch.cat([y, g.y], dim=0)
            new_edge_index = g.edge_index.clone()
            new_edge_index[0] = new_edge_index[0] + c1
            new_edge_index[1] = new_edge_index[1] + c2
            edge_index = torch.cat([edge_index, new_edge_index], dim=-1)
            c1 += g.x1.shape[0]
            c2 += g.x2.shape[0]
            x1indices.append(c1)
            x2indices.append(c2)
        x1indices = torch.tensor(x1indices, dtype=torch.long)
        x2indices = torch.tensor(x2indices, dtype=torch.long)

        for key in special_keys:
            if key in locals().keys():
                batch[key] = locals()[key]
        batch.size1 = x1.size(0)
        batch.size2 = x2.size(0)

        # batch['num_edges'] = edge_index.shape[0]

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    @staticmethod
    def split(x: Tensor, indices: Tensor):
        assert x.shape[0]==indices[-1], "size not match"
        xs = []
        indices = [0] + indices.detach().cpu().tolist()
        for i in range(len(indices) - 1):
            xs.append(x[indices[i]:indices[i + 1]])
        return xs

    def to_data_list(self):
        raise NotImplementedError("not implemented yet")

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1] + 1

"""
Mostly taken from `torch_geometric.data.DataLoader` with some modifications

"""

import torch.utils.data
# from torch.utils.data.dataloader import default_collate

from factor_gnn.batch import Batch

from torch import Tensor
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from torch.utils.data import Dataset


class DataLoaderBipartite(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False, **kwargs):
        super(DataLoaderBipartite, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list:Batch.from_data_list(data_list),
            **kwargs
        )


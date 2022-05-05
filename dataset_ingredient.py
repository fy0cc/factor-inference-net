import os
from os.path import join
import glob
import time
import random
import pickle
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import DataLoader
from matplotlib import pyplot as plt
import sys
import shutil
import scipy.stats as ss
from sklearn.metrics import r2_score

from factor_gnn.dataloader import DataLoaderBipartite
from factor_gnn.datasets import ThirdOrderDatasetBipartite,QM9Bipartite,GGMDatasetBipartite, LDPCDataset, LDPCFileDataset


from sacred import Ingredient
data_ingredient = Ingredient("dataset")


@data_ingredient.config
def cfg():
    datatype = "gaussianbipartite"
    dataroot = "data/gaussian/alltopo"
    dataset = "n10_size2000"
    test_dataset = ""
    refresh = False   # re-process the data or not
    datapath = join(str(dataroot), str(dataset))
    if len(str(test_dataset).strip()) == 0:
        test_datapath = datapath
    else:
        test_datapath = join(str(dataroot), str(test_dataset))
    rawdatapath = join(datapath, "raw/data.pt")
    test_percentage = 0.1
    batch_size = 256
    normalize_target = False
    singleton_factor = True
    size = 10000  # for dataset that generates samples on-the-fly
    target = -1
    if datatype == "3rdbipartite":
        dset = ThirdOrderDatasetBipartite(datapath, normalize_target=normalize_target,singleton_factor=singleton_factor)
    elif datatype == "gaussianbipartite":
        dset = GGMDatasetBipartite(datapath,singleton_factor=singleton_factor, refresh=refresh, target="precision")
    elif datatype=="qm9bipartite":
        dset = QM9Bipartite(datapath)
    elif datatype=="ldpc":
        dset = LDPCDataset()
    else:
        raise NotImplementedError(f"graph type {datatype} not supported")
    sample = dset[0]
    total_factor_types = list(set(sample.x1[:, 0].long().tolist()))
    num_factor_types = len(total_factor_types)
    record_dynamics_batch_size = 1
    record_dynamics_num_batch = 10

    n1 = sample.x1.shape[-1]  # 1st dimension is factor type
    n2 = sample.x2.shape[-1]
    ntarget = sample.y.shape[-1] if target <0 else 1 
    del dset
    del sample


@data_ingredient.capture
def load_data(datapath, test_datapath, test_percentage, target, batch_size, datatype, refresh,
                normalize_target,singleton_factor,size):
    """
    return `Dataset`, data loader should be constructed later as needed
    """
    dataloaderclass = {
        "3rdbipartite": DataLoaderBipartite,
        "gaussianbipartite": DataLoaderBipartite, 
        "qm9bipartite": DataLoaderBipartite, 
    }
    if datatype == "3rdbipartite":
        dset = ThirdOrderDatasetBipartite(datapath, normalize_target=normalize_target,singleton_factor=singleton_factor)
        if test_datapath is not None:
            testset = ThirdOrderDatasetBipartite(test_datapath, normalize_target=normalize_target,singleton_factor=singleton_factor)
    elif datatype == "gaussianbipartite":
        dset = GGMDatasetBipartite(datapath,singleton_factor=singleton_factor,refresh=refresh)
        if test_datapath is not None:
            testset = GGMDatasetBipartite(test_datapath,singleton_factor=singleton_factor,refresh=refresh)
    elif datatype=="qm9bipartite":
        dset = QM9Bipartite(datapath)
    elif datatype=="ldpc":
        dset = LDPCDataset(length=size)
    else:
        raise NotImplementedError(f"graph type {datatype} not supported")

    if datatype in ['3rdbipartite', 'gaussianbipartite', 'qm9bipartite']:
        ## Transform the dataset
        # choose single target
        if target >=0:
            dset.data.y = dset.data.y[:,[target]]
        # normalize target
        ymat = dset.data.y
        ymean = ymat.mean(dim=0)
        ystd = ymat.std(dim=0)
        if not normalize_target:
            ymean = torch.zeros(ymat.shape[-1])
            ystd = torch.ones(ymat.shape[-1])
        dset.data.y = (dset.data.y - ymean) /ystd 

        # out of distribution test set

        ## Split dataset for training, evaluation, and testing
        N = len(dset)
        # print("number of total data points:{}".format(N))
        percent = int(N*test_percentage)
        lengths = [percent,percent,N-2*percent]
        if test_datapath is not None:
            _,evalset,trainset = torch.utils.data.random_split(dset, lengths,generator=torch.Generator().manual_seed(42))
        else:
            testset,evalset,trainset = torch.utils.data.random_split(dset, lengths,generator=torch.Generator().manual_seed(42))
        # nsample_dyn = record_dynamics_batch_size*record_dynamics_num_batch
        # dynset,_  = torch.utils.data.random_split(testset,[nsample_dyn, percent-nsample_dyn])
        testloader = dataloaderclass[datatype](testset,batch_size=batch_size,shuffle=False,pin_memory=True)  #out of sample
        evalloader = dataloaderclass[datatype](evalset,batch_size=batch_size,shuffle=False,pin_memory=True)
        trainloader = dataloaderclass[datatype](trainset,batch_size=batch_size,shuffle=True,pin_memory=True)
        totalloader = dataloaderclass[datatype](dset, batch_size=batch_size, shuffle=False,pin_memory=True)
        # sample from tests dataset for a subset for recording latant dynamics
        # dynloader = dataloaderclass[datatype](dynset, batch_size=record_dynamics_batch_size,shuffle=False,pin_memory=True)
        # return [trainloader, evalloader, testloader, totalloader], (ymean,ystd)
        return [trainset, evalset,testset, dset], (ymean,ystd)
    elif datatype in ['ldpc']:
        return [
            dset, 
            # dset,
            LDPCDataset(length=size//10), 
            # LDPCDataset(length=size//10), 
            LDPCFileDataset(test_datapath),
        ],
        (None,None)
        
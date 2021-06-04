# -*- coding: utf-8 -*-
"""
@author Yicheng
functions `GABP_t`,`generateBroadH`,`generateData` written by Rajkumar Raju
"""
import os
import shutil
import argparse
import numpy as np

from networkx.utils import py_random_state
import networkx as nx
from networkx.linalg.graphmatrix import adjacency_matrix

from scipy import signal, linalg
from sklearn.datasets import make_sparse_spd_matrix
import pickle
import functools
from functools import partial
import multiprocessing
from numba import jit, njit
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from itertools import product


@jit(nopython=True)
def make_spd_matrix(n_dim):
    """taken from sklearn
    """
    A = np.random.rand(n_dim, n_dim)
    U, _, Vt = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(n_dim))), Vt)
    return X

@jit(nopython=True)
def generate_random_tree(N):
    adj = np.zeros((N, N), dtype=np.int64)
    n = N - 2
    seq = np.random.randint(1, N, size=(N - 2))
    adj = prufer_to_tree(seq)
    return adj

@jit(nopython=True)
def prufer_to_tree(seq):
    n = len(seq)
    nnode = n + 2
    adj = np.zeros((nnode, nnode), dtype=np.int64)
    degree = np.ones(nnode)
    for i in seq:
        degree[i - 1] += 1
    for i in seq:
        for j in range(1, nnode + 1):
            if degree[j - 1] == 1:
                adj[i - 1, j - 1] = 1
                adj[j - 1, i - 1] = 1
                degree[i - 1] -= 1
                degree[j - 1] -= 1
                break
    u = 0
    v = 0
    for i in range(1, nnode + 1):
        if degree[i - 1] == 1:
            if u == 0:
                u = i
            else:
                v = i
                break
    adj[u - 1, v - 1] = 1
    adj[v - 1, u - 1] = 1
    return adj

@jit(nopython=True)
def jacobi_rotate(A,i,j, atol=1e-8):
    if np.abs(A[i,j]) < atol and np.abs(A[j,i]) < atol:
        return
    
    n = A.shape[0]
    beta = (A[j,j] - A[i,i])/(2.0*A[i,j])
    t = np.sign(beta)/(np.abs(beta) + np.sqrt(1.0+beta*beta))
    c = 1./np.sqrt(1+t*t)
    s = c*t
    rho = s/(1. +c)
    aij = A[i,j]
    aii = A[i,i]
    ajj = A[j,j]
    ai = A[i,:].copy()
    aj = A[j,:].copy()
    
    A[i,j] = A[j,i] = 0.0
    A[i,i] = aii - t * aij
    A[j,j] = ajj +  t* aij
    for k in range(n):
        if k!=i and k!=j:
            A[i,k] = A[k,i] = ai[k] - s*(aj[k] + rho * ai[k])
            A[j,k] = A[k,j] = aj[k] + s*(ai[k] - rho * aj[k])
    
@jit(nopython=True)
def rotate_to_adj(A,adj, max_iter=1000, atol=1e-8):
    """
    find a similar matrix of A with adjacency matrix equal to adj
    """
    adj = adj.astype(np.int64)
    off_zero_mask = (adj==0)
    np.fill_diagonal(off_zero_mask, False)
    
    n = A.shape[0]
    A1 = A.copy()
    assert A.shape == adj.shape
    isconverged = False
    counter = 0
    while not isconverged:
        counter += 1
        for i in range(n):
            for j in range(i+1,n):
                if adj[i,j] == 0:
                    jacobi_rotate(A1,i,j)
        if counter > max_iter:
            break   
        isconverged = True
        for i in range(n):
            for j in range(i+1,n):
                if off_zero_mask[i,j]>0 and np.abs(A1[i,j]) > atol:
                    isconverged = False
                if not isconverged:
                    break
            if not isconverged:
                break
        if isconverged:
            for i in range(n):
                for j in range(i+1,n):
                    if off_zero_mask[i,j]:
                        A1[i,j] = A1[j,i] = 0.0
                            
    return A1, isconverged, counter

@jit(nopython=True)
def GABP_t(AMat, BMat, lam, r_init=0.1):
    """
    Function that implements Gaussian belief propagation
    Inputs:
        AMat: precision matrix
        BMat: time series of bias vectors
        lam: relaxation for updates, between 0 and 1
        p(x) = k.exp(-0.5x'Ax + b'x)
   
    Outputs: 
        InferredPrec: vector of marginal precisions
        InferredBias: vector of marginal biases    
        InferredPrecMat: time series of marginal precisions
        InferredBiasMat: time series of marginal biases
    """

    N, T = BMat.shape
    Ad = (AMat != 0) * 1 - np.eye(N)  # adjacency matrix

    # initialize the precision and biases of the messages
    # P[i,j] corresponds to message sent from node i to node j
    P = (
        (2.0 * np.random.rand(N, N) - 1) * r_init * Ad
    )  # initialize the precisions with some noise
    V = np.zeros((N, N))

    Pnew = np.zeros((N, N))
    Vnew = np.zeros((N, N))

    InferredPrec = np.zeros(N)
    InferredBias = np.zeros(N)
    InferredPrecMat = np.zeros((N, T))
    InferredBiasMat = np.zeros((N, T))

    for t in range(T):
        BVec = BMat[:, t]
        for i in range(N):
            for j in range(N):
                if Ad[i, j]:
                    # update message parameters
                    alpha = AMat[i, i] + np.sum(P[:, i]) - P[j, i]
                    beta = BVec[i] + np.sum(V[:, i]) - V[j, i]
                    Pnew[i, j] = (1 - lam) * P[i, j] - lam * AMat[i, j] ** 2 / alpha
                    Vnew[i, j] = (1 - lam) * V[i, j] - lam * AMat[i, j] * beta / alpha

        # now compute the beliefs
        InferredPrec = np.diag(AMat) + np.sum(Pnew, axis=0)
        InferredBias = BVec + np.sum(Vnew, axis=0)
        InferredPrecMat[:, t] = 1.0 * InferredPrec
        InferredBiasMat[:, t] = 1.0 * InferredBias

        P = 1.0 * Pnew
        V = 1.0 * Vnew
        
    isconverged = np.mean(np.abs((InferredPrecMat[:,-1]-  InferredPrecMat[:,-2]))) < 1e-5
    return InferredPrecMat, InferredBiasMat, isconverged


def generateBroadH(Nx, T, Tb, scaling):
    """
    Function to generate b(t), 
    Modeling b(t) such that it stays constant for every Nh time steps.
    """
    # first generate only T/Nh independent values of b
    shape = 2  # gamma shape parameter
    Lb = np.int(T // Tb)
    gsmScale = np.random.gamma(shape, scaling, (Nx, Lb))
    bInd = gsmScale * np.random.randn(Nx, Lb)
    bMat = np.zeros((Nx, T))

    # Then repeat each independent h for Nh time steps
    for t in range(T):
        bMat[:, t] = bInd[:, np.int(t // Tb)]

    return bMat


def generateData(sample_num, N, T, Tb, scale=1, alpha=0.6, r_level=0.2):
    f = signal.hamming(3, sym=True)
    f = f / sum(f)
    # sample a single precision matrix
    AMat = make_sparse_spd_matrix(
        dim=N, alpha=alpha, smallest_coef=r_level * 0.5, largest_coef=r_level * 1.25
    )

    biases = []
    activities = []
    for b in range(sample_num):
        BMat = signal.filtfilt(f, 1, generateBroadH(N, T, Tb, scale))
        biases.append(BMat)
        activities.append(np.stack(GABP_t(AMat, BMat, lam=0.25)))

    inputs = np.transpose(np.stack(biases), (0, 2, 1)).reshape(sample_num, T, N, 1)
    targets = np.transpose(np.stack(activities), (0, 3, 2, 1))
    return AMat, inputs, targets

@jit(nopython=True)
def generate_random_tree_gaussian(n):
    is_valid = False
    while not is_valid:
        # 1. generate a symmetric positive-definite matrix as precision matrix with random eigenvalues from [0.1, 10.0]
        ev = np.random.rand(n) * 9.9 + 0.1
        Pfull = np.diag(ev)
        Q = np.linalg.qr(np.random.randn(n,n))[0].copy()
        P  = Q @ Pfull @ Q.T
        # 2. rotate to tree structure
        adj = generate_random_tree(n)
        P1, converged, counter = rotate_to_adj(P,adj)
        is_valid = converged
    return P1, adj

def generate_data_tree(n):
    """
    generate a data point of tree-structure Gaussian graphical model
    output: tuple of (precision, adj, marginal_precision_vector)
    """
    P, adj = generate_random_tree_gaussian(n)
    C = np.linalg.inv(P)
    marginalP = 1.0 / np.diag(C)
    return (P, adj, marginalP)


def generate_data_fc(n):
    """
    generate a data point of fully connected Gaussian graphical model
    output: tuple of (precision, adj, marginal_precision_vector)
    """
    Cov = make_spd_matrix(n)
    P = np.linalg.inv(Cov)
    adj = np.asarray(P != 0, dtype=np.int)
    for i in range(n):
        adj[i, i] = 0  # set diagonal elements to 0
    marginalP = 1.0 / np.diag(Cov)
    return (P, adj, marginalP)


def generate_data_fc_valid(n):
    isvalid = False
    a = None
    b = None
    c = None
    while not isvalid:
        a, b, c = generate_data_fc(n)
        if np.all(c > 0.01):
            isvalid = True
    return a, b, c

def set_seed(x:int):
    np.random.seed(x)
    return np.random.rand()

def generate_dataset_tree(size, n):
    nprocs = 16
    pool = multiprocessing.Pool(nprocs)
    res = pool.map(set_seed,range(nprocs))    #set different random seeds for different processes
    # print(res)
    P = np.zeros((size, n, n))
    adj = np.zeros((size, n, n))
    marginalP = np.zeros((size, n))
    f = functools.partial(generate_data_tree)
    results = list(tqdm(pool.imap(f,[n]*size), total=size))
    pool.close()
    for i,item in enumerate(results):
        P[i,...] = item[0]
        adj[i,...] = item[1]
        marginalP[i,...] = item[2]
    return P, adj, marginalP


## ----------------- Functions for non-tree dataset --------------- 

def compute_count(channel, group):
    divide = channel//group
    remain = channel%group

    out = np.zeros(group, dtype=int)
    out[:remain]=divide+1
    out[remain:]=divide
    return out

@py_random_state(3)
def ws_graph(n, k, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """
    assert k>=2 and k<=n
    # compute number of edges:
    edge_num = int(round(k*n/2))
    count = compute_count(edge_num, n)
    # print(count)
    G = nx.Graph()
    for i in range(n):
        source = [i]*count[i]
        target = range(i+1,i+count[i]+1)
        target = [node%n for node in target]
        # print(source, target)
        G.add_edges_from(zip(source, target))
    # rewire edges from each node
    nodes = list(G.nodes())
    for i in range(n):
        u = i
        target = range(i+1,i+count[i]+1)
        target = [node%n for node in target]
        for v in target:
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G

@py_random_state(4)
def connected_ws_graph(n, k, p, tries=100, seed=1):
    """Returns a connected ws-flex graph.
    """
    for i in range(tries):
        # seed is an RNG so should change sequence each call
        G = ws_graph(n, k, p, seed)
        if nx.is_connected(G):
            return G
    raise nx.NetworkXError('Maximum number of tries exceeded')

def sprandsym_wsflex(n,k,p, max_iter=1000):
    g = connected_ws_graph(n,k,p)
    adj = adjacency_matrix(g).todense()
    adj = np.asarray(adj, bool)
    np.fill_diagonal(adj,True)
    ev = np.random.rand(n) * 10.0 + 0.1
    isconverged = False
    while not isconverged:
        Pfull = np.diag(ev)
        Q = np.linalg.qr(np.random.randn(n,n))[0]
        Pfull  = Q @ Pfull @ Q.T
        P, isconverged, counter = rotate_to_adj(Pfull, adj, max_iter=max_iter)
    assert np.all(np.linalg.eigvals(P) > 0.0)
    marginalP = 1.0 / np.diag(np.linalg.inv(P))
    return P, adj, marginalP, g
    
def sprandsym_wsflex_wrapper(tup,**kwargs):
    return sprandsym_wsflex(*tup, **kwargs)

def generate_dataset_nontree(ndata,n,nbins=20, max_iter=1000):
    """
    use make_sparse_spd_matrix with uniform alpha
    """
    
    deg_min = 4
    deg_max = n-1
    degrees = np.random.uniform(2,n-1,ndata)
    ps = np.random.uniform(0,1, ndata)
    n_list = [n] * len(ps)
    param_list = list(zip(n_list, degrees, ps))
    
    P=np.zeros((ndata,n,n))
    adj=np.zeros((ndata,n,n))
    marginalP = np.zeros((ndata,n))
    glist = [None for _ in range(ndata)]
    nprocs=8
    pool = multiprocessing.Pool(nprocs)
    res = pool.map(set_seed,range(nprocs))    
    f = partial(sprandsym_wsflex_wrapper, max_iter=max_iter)
    results = list(tqdm(pool.imap(f, param_list), total=len(param_list)))
    for i,res in enumerate(results):
        a,b,d,g = res
        P[i,:,:] = a
        adj[i,:,:] = b
        marginalP[i,:] = d
        glist[i] = g
    pool.close()
    return (P,adj,marginalP, glist)

def show_stats(P):
    # show statistics of singleton precision
    assert len(P.shape) == 3, "size incorrect"
    data = np.reshape(
        np.concatenate([np.diag(P[i, :, :]) for i in range(P.shape[0])], axis=0), (-1,)
    )
    print(
        "Statistics of singleton precision:\n mean:{} median:{} var:{}".format(
            np.mean(data), np.median(data), np.var(data)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structure", type=str, default="tree", help="structure of the graph"
    )
    parser.add_argument("--outdir", type=str, default="data", help="output folder")
    parser.add_argument(
        "--ndata", type=int, default=10000, help="size of the dataset to be generated"
    )
    parser.add_argument("--nnode", type=int, default=10, help="number of nodes per graph")

    args = parser.parse_args()
    
    ## Training set 
    # dirname = "size{}_n{}/raw".format(args.ndata, args.nnode)
    dirname = f"{args.outdir}/raw"
    if not os.path.isdir(dirname):
        os.makedirs(dirname, exist_ok=True)
    outfilename = os.path.join(dirname, "data.pt")
    with open(outfilename, "wb") as file:
        if args.structure == "tree":
            dataset = generate_dataset_tree(args.ndata, args.nnode)
            show_stats(dataset[0])
            pickle.dump(dataset, file)
        elif args.structure == "all":
            dataset = generate_dataset_nontree(args.ndata, args.nnode)
            show_stats(dataset[0])
            pickle.dump(dataset[:3], file)
        else:
            raise ValueError("not implemented")
        print("stored to {}".format(outfilename))

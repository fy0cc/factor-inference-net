"""
functions `ws_graph`, `connected_ws_graph`, `compute_stats` are taken from https://github.com/facebookresearch/graph2nn

"""
import networkx as nx
import numpy as np
import torch
from networkx.utils import py_random_state
from numba import jit


def compute_count(channel, group):
    divide = channel//group
    remain = channel%group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    return out


@py_random_state(3)
def ws_graph(n, k, p, seed=1):
    """Returns a ws-flex graph, k can be real number in [2,n]
    """
    assert k >= 2 and k <= n
    # compute number of edges:
    edge_num = int(round(k*n/2))
    count = compute_count(edge_num, n)
    # print(count)
    G = nx.Graph()
    for i in range(n):
        source = [i]*count[i]
        target = range(i + 1, i + count[i] + 1)
        target = [node%n for node in target]
        # print(source, target)
        G.add_edges_from(zip(source, target))
    # rewire edges from each node
    nodes = list(G.nodes())
    for i in range(n):
        u = i
        target = range(i + 1, i + count[i] + 1)
        target = [node%n for node in target]
        for v in target:
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w==u or G.has_edge(u, w):
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


def compute_stats(G):
    G_cluster = sorted(list(nx.clustering(G).values()))
    cluster = sum(G_cluster)/len(G_cluster)
    # path
    path = nx.average_shortest_path_length(G)
    return cluster, path


def norm(x):
    ## to 0-1
    x = x - x.min()
    if x.max()==0:
        x[:] = 0
    else:
        x = x/x.max()
    return x


def get_graph(n, degree, p, repeat=30):
    # row: repeat, column: n,degree,p,seed,cluster,path
    result = np.zeros((repeat, 6))
    for i in range(repeat):
        graph = connected_ws_graph(n=n, k=degree, p=p, seed=i)
        cluster, path = compute_stats(graph)
        result[i] = [n, degree, p, i, cluster, path]
    return result


def get_incidence_matrix_from_graph(g):
    nfactor = g.edge_index.shape[1]  # (2, num_factor)
    nvar = g.x2.shape[0]
    A = torch.zeros(nfactor, nvar).long()
    for i in range(nfactor):
        ifactor, ivar = g.edge_index[:, i]
        A[ifactor, ivar] = 1
    return A


@jit(nopython=True)
def jacobi_rotate(A, i, j, atol=1e-8):
    if np.abs(A[i, j]) < atol and np.abs(A[j, i]) < atol:
        return

    n = A.shape[0]
    beta = (A[j, j] - A[i, i])/(2.0*A[i, j])
    t = np.sign(beta)/(np.abs(beta) + np.sqrt(1.0 + beta*beta))
    c = 1./np.sqrt(1 + t*t)
    s = c*t
    rho = s/(1. + c)
    aij = A[i, j]
    aii = A[i, i]
    ajj = A[j, j]
    ai = A[i, :].copy()
    aj = A[j, :].copy()

    A[i, j] = A[j, i] = 0.0
    A[i, i] = aii - t*aij
    A[j, j] = ajj + t*aij
    for k in range(n):
        if k!=i and k!=j:
            A[i, k] = A[k, i] = ai[k] - s*(aj[k] + rho*ai[k])
            A[j, k] = A[k, j] = aj[k] + s*(ai[k] - rho*aj[k])


@jit
def rotate_to_adj(A, adj, max_iter=1000, atol=1e-8):
    """
    find a similar matrix of A with adjacency matrix equal to adj
    """
    adj = adj.astype(np.int64)
    off_zero_mask = (adj==0)
    np.fill_diagonal(off_zero_mask, False)

    n = A.shape[0]
    A1 = A.copy()
    assert A.shape==adj.shape
    isconverged = False
    counter = 0
    while not isconverged:
        counter += 1
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]==0:
                    jacobi_rotate(A1, i, j)
        if counter > max_iter:
            break
        isconverged = True
        for i in range(n):
            for j in range(i + 1, n):
                if off_zero_mask[i, j] > 0 and np.abs(A1[i, j]) > atol:
                    isconverged = False
                if not isconverged:
                    break
            if not isconverged:
                break
        if isconverged:
            for i in range(n):
                for j in range(i + 1, n):
                    if off_zero_mask[i, j]:
                        A1[i, j] = A1[j, i] = 0.0

    return A1, isconverged, counter


def is_connected(adj):
    n = adj.shape[0]
    c = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                c[j] = c[i]
    if len(set(c)) > 1:
        return False
    else:
        return True


def generate_random_tree(N):
    adj = np.zeros((N, N))
    n = N - 2
    seq = np.random.randint(1, N, size=(N - 2))
    adj = prufer_to_tree(seq)
    return adj


def prufer_to_tree(seq):
    n = len(seq)
    nnode = n + 2
    adj = np.zeros((nnode, nnode), dtype=np.int)
    degree = np.ones(nnode)
    for i in seq:
        degree[i - 1] += 1
    for i in seq:
        for j in range(1, nnode + 1):
            if degree[j - 1]==1:
                adj[i - 1, j - 1] = 1
                adj[j - 1, i - 1] = 1
                degree[i - 1] -= 1
                degree[j - 1] -= 1
                break
    u = 0
    v = 0
    for i in range(1, nnode + 1):
        if degree[i - 1]==1:
            if u==0:
                u = i
            else:
                v = i
                break
    adj[u - 1, v - 1] = 1
    adj[v - 1, u - 1] = 1
    return adj

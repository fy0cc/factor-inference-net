using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
@everywhere using MyMCMCSampler
@everywhere using StatsBase
@everywhere using Combinatorics
@everywhere using Random
@everywhere using PyCall
using Distributions
using DataFrames
using ArgParse
using JLD2, FileIO
using ProgressMeter
using Base.Iterators: enumerate
using UUIDs
using StatsPlots
using BenchmarkTools
@everywhere using Base: @nloops, @ntuple

py"""
import deepdish as dd
def savefile(filepath,data):
    dd.io.save(filepath,data)
"""
rng = MersenneTwister(1234);

@everywhere extensions = quote
    py"""
    import numpy as np
    import networkx as nx
    from networkx.utils import py_random_state
    def compute_count(channel, group):
        divide = channel//group
        remain = channel%group
    
        out = np.zeros(group, dtype=int)
        out[:remain]=divide+1
        out[remain:]=divide
        return out
    
    @py_random_state(3)
    def ws_graph(n, k, p, seed=1):
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
        for i in range(tries):
            # seed is an RNG so should change sequence each call
            G = ws_graph(n, k, p, seed)
            if nx.is_connected(G):
                return G
        raise nx.NetworkXError('Maximum number of tries exceeded')
    
    
    def compute_stats(G):
        G_cluster = sorted(list(nx.clustering(G).values()))
        cluster = sum(G_cluster) / len(G_cluster)
        # path
        path = nx.average_shortest_path_length(G)
        return cluster, path
    
    def norm(x):
        ## to 0-1
        x = x-x.min()
        if x.max()==0:
            x[:]=0
        else:
            x = x/x.max()
        return x
    
    
    
    def get_graph(n,degree,p,repeat=30):
        # row: repeat, column: n,degree,p,seed,cluster,path
        result = np.zeros((repeat, 6))
        for i in range(repeat):
            graph = connected_ws_graph(n=n, k=degree, p=p, seed=i)
            cluster, path = compute_stats(graph)
            result[i] = [n, degree, p, i, cluster, path]
        return result
    
    
    def is_connected(adj):
        n = adj.shape[0]
        c = np.arange(n)
        for i in range(n):
            for j in range(i+1,n):
                if adj[i,j]:
                    c[j] = c[i]
        if len(set(c)) > 1:
            return False
        else:
            return True
    """
    end
    
@everywhere eval(extensions)



### Generate training dataset
s = ArgParseSettings()
@add_arg_table s begin
    "--outdir"
        help = "folder to store dataset file"
        arg_type = String
        default = "data/tmp2"
    "--gamma"
        help = "scaling factor for the potential"
        arg_type = Float64 
        default = 1.0
    "--ndata"
        help = "number of data points"
        arg_type = Int
        default = 100
    "--n"
        help = "size of the graph"
        arg_type = Int
        default = 10
end
args = parse_args(ARGS, s)
println(args)
n = args["n"]
ndata = args["ndata"]
γ = args["gamma"]

spin_mode=:spinglass
ks= range(2.0, stop=floor((n-1)*(n-2)/6),length=30)
ps= range(0.0, stop=1.0, length=30)
spin_mode=:spinglass

results = @showprogress pmap(1:ndata) do i
    k = rand(Uniform(2.0, floor((n-1)*(n-2)/6)))
    p = rand(Uniform(0.0, 1.0))
    isconnected = false
    d = nothing
    while !isconnected
        d = random_ws_factor_graph(n,0,k,p,γ=γ,mode=spin_mode,degrees=[3], singleton_dist=:randn)
        A = incident_matrix(d)
        A = (A' * A .>0)
        isconnected = py"is_connected"(A)
    end
    mP = computemarginal(d)[1]
    mP = reshape(mP,:,1)
    return (d,dist2dict(d),mP)
end

# save to file 
data2file = map(results) do res
    res[2:end]  # (dict, marginal)
end
outdir = joinpath(args["outdir"], "raw")
mkpath(outdir)
outfilename = joinpath(outdir, "data.dd")
println("Output to: $(outfilename)")
py"savefile"(outfilename,data2file)
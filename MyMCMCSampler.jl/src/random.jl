"""
utility functions to randomly generate random graphical models with desired properties
"""

using Base.Iterators: enumerate
using Random: randperm

function rand_regular_graph(
    n::Integer,
    k::Integer,
    d::Integer;
    β::Real = 1.0,
    α::Real = 0.2,
    glassy::Bool = false,
)
    """
    random regular graph where the degree of factor node is fixed to be k, and the degree
    of variable node is `d` on average
    """

    # Singleton potential coefficients
    node_attr =
        [vcat(rand(Normal(0.0, 0.25), 1), rand(Normal(0.0, 1.0), k - 1)) for _ = 1:n]
    nfactor = Integer(ceil(n * d / k))
    all_indexk = collect(combinations(1:n, k))
    if nfactor > length(all_indexk)
        nfactor = length(all_indexk)
    end
    edge_index = all_indexk[randperm(length(all_indexk))[1:nfactor]]    # variables involved for each k-factor

    #  k-factor potential coefficients
    edge_attr = [rand(Uniform(0.02, 3.0), 1) * β for _ = 1:length(edge_index)]
    if glassy
        edge_attr .*= rand([-1.0, 1.0], length(edge_attr))
    end

    edge_type = [k for _ = 1:length(edge_index)]

    return DiscreteSpinGlass(edge_index, edge_attr .* β, edge_type, node_attr .* β, α)
end

## regular graphs 

function randomfcspinglass(n::Integer;beta::Real=1.0)
    # generate a fully connected 2,3-spin glass distribution
    node_attr = [rand(Normal(0.0,1.0),1)*beta for _ in 1:n]
    edge_attr = [rand(Normal(0.0,1.0),1)*beta for _ in 1:(Int(n*(n-1)/2+n*(n-1)*(n-2)/6))]
    all_index2 = collect(combinations(1:n,2))
    all_index3 = collect(combinations(1:n,3))
    edge_index = vcat(all_index2,all_index3)
    edge_type = vcat([2 for _ in 1:length(all_index2)],[3 for _ in 1:length(all_index3)])

    return DiscreteSpinGlass(edge_index,edge_attr,edge_type,node_attr,0.0)
end
    
function randomfcferro(n::Integer;beta::Real=1.0)
    # generate a fully connected 2,3-spin glass distribution
    node_attr = [[0.0].*beta for _ in 1:n]
    edge_attr = [rand(Uniform(0.0,1.0),1)*beta for _ in 1:(Int(n*(n-1)/2+n*(n-1)*(n-2)/6))]
    all_index2 = collect(combinations(1:n,2))
    all_index3 = collect(combinations(1:n,3))
    edge_index = vcat(all_index2,all_index3)
    edge_type = vcat([2 for _ in 1:length(all_index2)],[3 for _ in 1:length(all_index3)])

    return DiscreteSpinGlass(edge_index,edge_attr,edge_type,node_attr,0.0)
end

function randomkfactorgraph(n::Integer,k::Integer,ρ::Real;β::Real=1.0)
    # generate a random k-factor graph without singleton potentials
    # ρ is the density of factors
    node_attr = [
        vcat(rand(Normal(0.0,0.25),1),rand(Normal(0.0,1.0),k-1)) for _ in 1:n
    ]
    all_indexk = collect(combinations(1:n,k))
    Cnk = length(all_indexk)
    Nfactor = Integer(ceil(ρ*length(all_indexk)))
    idx = randperm(length(all_indexk))[1:Nfactor]
    edge_index = all_indexk[idx]
    edge_attr = [rand(Normal(0.0,1.0),1)*β for _ in 1:length(edge_index)]
    edge_type = [k for _ in 1:length(edge_index)]

    return DiscreteSpinGlass(edge_index,edge_attr,edge_type,node_attr,0.0)
end

function randomregulargraph(n::Integer,k::Integer,d::Integer;β::Real = 1.0)
    # random regular graph where the degree of factor node is fixed to be k, and the degree
    # of variable node is also fixed to be d

    node_attr = [
        vcat(rand(Normal(0.0,0.25),1),rand(Normal(0.0,1.0),k-1)) for _ in 1:n
    ]
    nfactor = Integer(ceil(n*d/k))
    
    all_indexk = collect(combinations(1:n,k))
    edge_index = all_indexk[randperm(length(all_indexk))[1:nfactor]]
        
    edge_attr = [rand(Normal(0.0,1.0),1)*β for _ in 1:length(edge_index)]
    edge_type = [k for _ in 1:length(edge_index)]

    return DiscreteSpinGlass(edge_index,edge_attr,edge_type,node_attr,0.0)
end


function compute_count(channel::Integer, group::Integer)
    divide, remain = divrem(channel, group)
    out = zeros(Int,group)
    out[1:remain] .= divide +1
    out[remain+1:end] .= divide
    return out
end

function random_ws_graph(n::Integer,k::Real,p::Real)
    @assert k>=2 && k<=n
    edge_num = Int(round(k*n/2))
    count = compute_count(edge_num,n)
    idx = []
    # ring graph
    for i in 0:n-1
        source = repeat([i],count[i+1])
        target = collect(i+1:i+count[i+1]+1)
        target = [mod(t,n) for t in target]
        append!(idx, collect(zip(source,target)))
    end
    # rewire
    for i in 0:n-1
        u = i
        target = collect(i+1:i+count[i+1]+1)
        target = [mod(t,n) for t in target]
        for v in target
            if rand() < p
                w = rand(0:n-1)
                while w==u || (u,w) in idx 
                    w = rand(0:n-1)
                    if length(findall(x->x[1]==u, idx)) >= n-1
                        break
                    end
                end
                deleteat!(idx, findall(x->x==1, (u,v)))
                push!(idx, (u,w))
            end
        end
    end
    # reshape into (nfactor, 2)
    nfactor = length(idx)
    factors = zeros(Int, nfactor,2)
    for i in 1:nfactor
        factors[i,:] = [idx[i][1], idx[i][2]]
    end
    return factors .+ 1
end


#### WS-style random graph and utilities
matchrow(a,B) = findfirst(i->all(j->a[j] == B[i,j],1:size(B,2)),1:size(B,1))

function random_ws_factor_structure(n::Integer, k::Real, p::Real)
    """
    k: in [2.0, floor((n-1)*(n-2)/6)]
    p: in [0,1]
    """
    f = floor(n*k/3) |> Int
    k̂ = floor(f/n) |> Int
    factors = zeros(Int, f,3)
    offsets = []
    for l in 1:n-1
        for i in 1:div(l,2)
            if l==2*i
                push!(offsets,[i,i])
            else
                if mod(n,2)==0 & l-2*i==1
                    push!(offsets,[i,l-i])
                else
                    push!(offsets,[i,l-i])
                    push!(offsets,[l-i,i])
                end
            end
        end
    end

    # 1. create "ring" factor graph
    counter = 0
    for i in 1:n, j in 1:k̂
        counter += 1
        off = offsets[j]
        factors[counter,:] = [mod(i-off[1], n), mod(i,n), mod(i+off[2],n)]
    end
    
    # 2. randomly connect remaining factors
    max_factor_num = length(combinations(1:n-1,2))
    for _ in 1:rem(f,n)
        counter += 1
        # find available center node
        found = false
        icenter = 0
        while !found 
            i = rand(Categorical(n))
            if sum(factors[:,2]==i) < max_factor_num
                found = true
                icenter = i
            end
        end
        # find closest neibouring nodes to form a 3-factor
        for j in 1:length(offsets)
            off = offsets[j]
            candidate = [mod(icenter-off[1], n), mod(icenter,n), mod(icenter+off[2],n)]
            ioff = matchrow(candidate, factors)
            if ioff === nothing
                factors[counter,:] = candidate
                break
            end
        end
    end
    # keep each factor neighbor list sorted
    factors = mapslices(sort, factors, dims=2)
    
    # 3. randomly rewire each factor
    all_comb = collect(combinations(0:n-1,3))
    for i in 1:f
        if rand()< p
            all_remain = setdiff(all_comb, [factors[i,:] for i in 1:size(factors,1)])
            if length(all_remain) > 0
                candidate = rand(all_remain)
                factors[i,:] = candidate
            end
        end
    end
            
    return factors .+ 1
end
        

function random_ws_factor_graph(n::Integer, k2::Real, k3::Real, p::Real;γ::Real = 1.0,γ1=nothing, mode=:spinglass, degrees::Vector{<:Integer}=[3], singleton_dist=:uniform)
    if γ1 === nothing
        γ1 = γ
    end

    if singleton_dist == :uniform
        node_attr = [
                rand(Uniform(0.0, 1.0),3).*γ1  for _ in 1:n 
            ]
    elseif singleton_dist == :randn
        node_attr = [
                rand(Normal(0.0,1.0),3).*γ1 for _ in 1:n 
            ]
    elseif singleton_dist == :zero
        node_attr = [
                zeros(3).*γ1 for _ in 1:n 
            ]
    end
    edge_index_tot = []
    edge_attr_tot = []
    edge_type_tot = []

    if 3 in degrees
        factors = random_ws_factor_structure(n,k3,p)
        edge_index = [factors[i,:] for i in 1:size(factors,1)]
        nfactor = length(edge_index)
        if mode==:spinglass
            edge_attr = [rand(Normal(0.0,1.0),1)*γ for _ in 1:length(edge_index)]
        elseif mode==:ferro
            edge_attr = [rand(Gamma(2.0),1)*γ for _ in 1:length(edge_index)]
        end
        edge_type = [3 for _ in 1:length(edge_index)]
        append!(edge_type_tot, edge_type)
        append!(edge_attr_tot, edge_attr)
        append!(edge_index_tot, edge_index)
    end
    if 2 in degrees
        factors = random_ws_graph(n,k2,p)
        edge_index = [factors[i,:] for i in 1:size(factors,1)]
        nfactor = length(edge_index)
        if mode==:spinglass
            edge_attr = [rand(Normal(0.0,1.0),1)*γ for _ in 1:length(edge_index)]
        elseif mode==:ferro
            edge_attr = [rand(Gamma(2.0),1)*γ for _ in 1:length(edge_index)]
        end
        edge_type = [2 for _ in 1:length(edge_index)]
        append!(edge_type_tot, edge_type)
        append!(edge_attr_tot, edge_attr)
        append!(edge_index_tot, edge_index)
    end

    return DiscreteSpinGlass(edge_index_tot,edge_attr_tot,edge_type_tot,node_attr,1.0)
end
function incident_matrix(d::Union{ThirdOrderExpFamily,DiscreteSpinGlass})
    A = zeros(Bool, d.nfactor, d.nnode)
    for i in 1:d.nfactor
        A[i, d.edge_index[i]] .= 1
    end
    return A
end

#### functions on `ThirdOrderExpFamily`
function Base.length(d::ThirdOrderExpFamily)::Int
    return d.nnode
end

function Distributions._logpdf(d::ThirdOrderExpFamily, x::AbstractVector)
    # unnormalized log probability (negative free energy)
    # and L2 norm square is added to ensure the functions is normalizable
    return -potential(d, x) - d.alpha * sum(x.^2)^2
end

function Distributions.insupport(d::ThirdOrderExpFamily, x::AbstractVector)
    return true
end

function potential(d::ThirdOrderExpFamily, x::AbstractVector)
    # free energy
    res = 0.0
    # singleton
    for i in 1:d.nnode
        res += ef1(x[i], d.node_attr[i])
    end
    # higher order
    for i in 1:d.nfactor
        if d.edge_type[i] == 2
            res += ef2(x[d.edge_index[i]], d.edge_attr[i])
        elseif d.edge_type[i] == 3
            res += ef3(x[d.edge_index[i]], d.edge_attr[i])
        else
            error("edge type note supported")
        end
    end
    
    return res
end

function ef1(x::Real, attr::AbstractVector)
    ans = 0.0
    for (i, a) in enumerate(attr)
        ans += a * x^i
    end
    return ans
end

function ef2(x::AbstractVector, attr::AbstractVector)
    @assert length(x) == 2 "len error, should be 2"
    @assert length(attr) == 1 "len error, should be 1"
    return attr[1] * prod(x)
end

function ef3(x::AbstractVector, attr::AbstractVector)
    @assert length(x) == 3 "len error, should be 3"
    @assert length(attr) == 1 "len error, should be 1"
    return attr[1] * prod(x)
end

function dist2dict(d::ThirdOrderExpFamily)
    # convert a distribution to a dictionary for serialization
    dic = Dict()
    dic["node_attr"] = d.node_attr
    dic["edge_attr"] = d.edge_attr
    dic["edge_type"] = d.edge_type
    dic["edge_index"] = d.edge_index
    dic["nnode"] = d.nnode 
    dic["nfactor"] = d.nfactor
    dic["alpha"] = d.alpha 
    return dic
end

function dist2dict(d::DiscreteSpinGlass)
    dic = Dict()
    dic["node_attr"] = d.node_attr
    dic["edge_attr"] = d.edge_attr
    dic["edge_type"] = d.edge_type
    dic["edge_index"] = d.edge_index
    dic["nnode"] = d.nnode 
    dic["nfactor"] = d.nfactor
    dic["beta"] = d.beta
    return dic
end


## -----------------------------   SingleThirdOrderExpFamily
function Base.length(d::SingleThirdOrderExpFamily)::Int
    return 1
end

function Distributions._logpdf(d::SingleThirdOrderExpFamily, x::AbstractVector)
    @assert length(x) == 1 "size incorrect"
    return -potential(d, x) - d.alpha * x[1]^4
end

function Distributions.insupport(d::SingleThirdOrderExpFamily, x::AbstractVector)
    @assert length(x) == 1 "size incorrect"
    return true
end

function potential(d::SingleThirdOrderExpFamily, x::AbstractVector)
    @assert length(x) == 1 "size incorrect"
    ans = 0.0 
    for i in 1:length(d.node_attr)
        ans += d.node_attr[i] * x[1]^i 
    end
    return ans
end

## -----------------------------   DiscreteSpinGlass
function Base.length(d::DiscreteSpinGlass)::Int 
    return d.nnode 
end

function Distributions._logpdf(d::DiscreteSpinGlass, x::AbstractVector)
    return -potential(d, x) * d.beta
end

function potential(d::DiscreteSpinGlass, x::AbstractVector)
    ans = 0.0
    if !has_singleton_factor(d)      # if no singleton factor
        ans += sum([ef1(x[i],attr) for (i,attr) in enumerate(d.node_attr)])
        # ans += sum(d.node_attr .* x)  # bias term calculated from node_attr
    end
    for i in 1:d.nfactor 
        ans += d.edge_attr[i][1] * prod(x[d.edge_index[i]])
    end
    return ans
end

function Distributions.insupport(d::DiscreteSpinGlass, x::AbstractVector)
    return true 
end



## ------------------------------ MCMC sampling function -----
function sampleSingleChain(nstep::Integer, d::ContinuousDistribution, burnin::Integer = 3000, 
        target::Real = 0.8)
    # Sample a single chain in one process
    # Return: Array of shape(nsample,nvariable)
    nvar = length(d)
    logp = x->logpdf(d, x)
    gradlogp = x->ForwardDiff.gradient(logp, x)
    logfgrad = x->(logp(x), gradlogp(x))
    theta = NUTSVariate(rand(Uniform(-100.0, 100.0), nvar), logfgrad; target = target)
    ans = zeros(nstep - burnin, nvar)
    
    for i in 1:nstep
        sample!(theta, adapt = (i <= burnin))
        if i > burnin
            ans[i - burnin,:] = theta
        end
    end
    return ans
end

function sampleMultiChain(nchain::Integer, nstep::Integer, d::ContinuousDistribution, burnin::Integer = 3000, 
        target::Real = 0.8; distributed = true)
    sim = Chains(nstep, length(d), start = (burnin + 1), chains = nchain)
    args = []
    results = pmap(sampleSingleChain, repeat([nstep], nchain), repeat([d], nchain), repeat([burnin], nchain), repeat([target], nchain);distributed = distributed)
    for i in 1:length(results)
        sim[:,:,i] = results[i]
    end
    return sim
end


function sampleMultiChain2(model::Model, nvar::Integer, nstep::Integer, burnin::Integer = 5000, nchain::Integer = 6)
    inits = [
        Dict(:x => rand(Uniform(-10, 10), nvar)) for _ in 1:nchain
    ]
    scheme = [NUTS([:x])]
    setsamplers!(model, scheme)
    sim = mcmc(model, Dict{Symbol,Any}(), inits, nstep, burnin = burnin, chains = nchain;verbose = false)
    return sim
end


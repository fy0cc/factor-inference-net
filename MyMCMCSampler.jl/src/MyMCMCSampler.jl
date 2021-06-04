
module MyMCMCSampler
# export simpleRandomGraph,randomGraph, singlePotential, multiPotential, makeDataDict, sampleSingleChain, sampleAllChains, ThirdOrderExpFamily

using Reexport
@reexport using Mamba
using Distributions
using DataFrames
using ForwardDiff
using Distributed
using Parameters
using Combinatorics: combinations

###################### Types #############################
struct ThirdOrderExpFamily <: ContinuousMultivariateDistribution
    edge_index::AbstractArray # an array of node index in each edge
    edge_attr::AbstractArray # edge potential
    edge_type::AbstractArray # array of edge type
    node_attr::AbstractArray # singleton potential
    nnode::Integer # number of variable node
    nfactor::Integer # number of factor node
    alpha::Float64  #  coefficient before the |x|^4_2 base measure
end

ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, nnode::Integer, nfactor::Integer) = ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, nnode, nfactor, 0.2) 
ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr) = ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, length(node_attr), length(edge_type))
ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, alpha::Real) = ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, length(node_attr), length(edge_type), alpha)


struct SingleThirdOrderExpFamily <: ContinuousMultivariateDistribution
    node_attr::AbstractArray # singleton potential
    alpha::Float64 
end

SingleThirdOrderExpFamily(node_attr) = SingleThirdOrderExpFamily(node_attr, 0.2)

################### Spin Glass Distributions ######################3
struct DiscreteSpinGlass <: DiscreteMultivariateDistribution
    edge_index::AbstractArray # an array of node index in each edge
    edge_attr::AbstractArray # edge potential
    edge_type::AbstractArray # edge type
    node_attr::AbstractArray # singleton potential
    nnode::Integer # number of variable node
    nfactor::Integer # number of factor node
    beta::Real # inverse temperature
end

# DiscreteSpinGlass(edge_index,edge_attr,node_attr,nnode::Integer) = DiscreteSpinGlass(edge_index,edge_attr,node_attr,nnode,nfactor) 
DiscreteSpinGlass(edge_index, edge_attr, edge_type, node_attr, beta::Real) = DiscreteSpinGlass(edge_index, edge_attr,edge_type, node_attr, length(node_attr), length(edge_attr), beta)
DiscreteSpinGlass(edge_index, edge_attr, edge_type, node_attr) = DiscreteSpinGlass(edge_index, edge_attr, edge_type, node_attr, 1.0)
# DiscreteSpinGlass(;kwargs) = DiscreteSpinGlass(;kwargs...)
# DiscreteSpinGlass(edge_index,edge_attr,node_attr)=DiscreteSpinGlass(edge_index,edge_attr,node_attr,length(node_attr),length(edge_attr))

function add_singleton_factor(d::DiscreteSpinGlass)
    @unpack edge_index, edge_attr, edge_type, node_attr, beta,nnode,nfactor = d
    nfactor += nnode
    for i in 1:nnode 
        push!(edge_index, [i])
        push!(edge_type,1)
        push!(edge_attr, node_attr[i])
    end
    return DiscreteSpinGlass(edge_index, edge_attr, edge_type, node_attr, nnode, nfactor,beta)
end

function add_singleton_factor(d::ThirdOrderExpFamily)
    @unpack edge_index, edge_attr, edge_type, node_attr, alpha,nnode,nfactor = d
    nfactor += nnode
    for i in 1:nnode 
        push!(edge_index, [i])
        push!(edge_type,1)
        push!(edge_attr, node_attr[i])
    end
    return ThirdOrderExpFamily(edge_index, edge_attr, edge_type, node_attr, nnode, nfactor,alpha)
end

function has_singleton_factor(d::Union{ThirdOrderExpFamily, DiscreteSpinGlass})
    return (1 in unique(d.edge_type))
end



###################### Include #############################
include("thirdexpfam.jl")
include("mcmcutils.jl")
include("random.jl")
include("bp.jl")
# include("gaussian.jl")

###################### Exports #########################
export 
    ThirdOrderExpFamily,
    SingleThirdOrderExpFamily,
    DiscreteSpinGlass,
    sampleSingleChain,
    sampleMultiChain,
    sampleMultiChain2,
    potential,
    dist2dict,
    getstats,
    isvalid,
    rand_regular_graph,
    add_singleton_factor, 
    random_ws_factor_graph,
    random_ws_factor_structure,
    incident_matrix,
    computemarginal,
    get_neighbours, 
    bp_3spin,
    bp

end # module MyMCMCSampler

# types and util funcitons used by the gaussian BP
# included by MessageGNN.jl

export GMM,GMMBP, randomSPMatrix, randomGaussian, randPrecision, randomGaussianBPConvergence, messagePassing, messagePassingOneStep, generateRandomBPConvgenceGraph,generateRandomBPConvgenceTree,getMarginalPrecision

import Distributions
import RandomMatrices

mutable struct GMM
    μ::Array{Float32,1}
    V::Array{Float32,2}
    P::Array{Float32,2}
    idx::Array{Int32,2}
    adj::Array{Int32,2}
    coeffs::Array{Float32,2}
    nvar::Int32
    nfactor::Int32
    GMM(μ, V;isPrecision = false) = constructGMM(new(), μ, V;isPrecision = isPrecision)
end

mutable struct GMMBP
    g::GMM
    marginalMeanList::Array{Float32,2} # list of intermediate marginal means
    marginalPList::Array{Float32,2} # list of intermediate marginal variances
    GMMBP(g, meanList, PList) = constructGMMBP(new(), g, meanList, PList)
end

function constructGMM(g::GMM, μ::AbstractVector, M::AbstractArray;isPrecision = false)
    if isPrecision
        g.P = M
        g.V = inv(M)
    else
        g.V = M
        g.P = inv(P)
    end
    g.μ = μ
    P = g.P
    g.nvar = size(μ, 1)
    g.nfactor = g.nvar * (g.nvar + 1) / 2 # will be updated later
    g.idx = zeros(Int32, g.nfactor, 2)
    g.adj = zeros(Int32, g.nfactor, g.nvar)
    g.coeffs = zeros(Float32, g.nfactor, 3)
    # singleton factors
    for i in 1:g.nvar
        g.idx[i,:] = [i,-1]
        g.adj[i,i] = 1
        g.coeffs[i,:] = [P[i,i], -2 * g.μ[i] * P[i,i], 0]
    end
    # pairwise factors
    fid = g.nvar + 1
    for i in 1:g.nvar - 1
        for j in i + 1:g.nvar
            if (P[i,j] != 0)
                g.idx[fid,:] = [i,j]
                g.adj[fid,i] = 1
                g.adj[fid,j] = 1
                g.coeffs[fid,:] =  [2 * P[i,j], -2 * g.μ[i] * P[i,j], -2 * g.μ[j] * P[i,j] ]
                fid += 1
            end
        end
    end
    g.nfactor = fid - 1
    g.idx = g.idx[1:g.nfactor,:]
    g.adj = g.adj[1:g.nfactor,:]
    g.coeffs = g.coeffs[1:g.nfactor,:]
    return g
end

function constructGMMBP(gbp::GMMBP, g::GMM, meanList::AbstractArray, PList::AbstractArray)
    gbp.g = g
    gbp.marginalMeanList = meanList
    gbp.marginalPList = PList
    return gbp
end

function randomSPMatrix(N::Integer)
    # A = Q'ΛQ
    # 2. random orthonormal used to transform the diagonal matrix
    # Harr matrix samples from beta-circular ensemble. beta: 1-orthogonal 2-unitary 4-symplectic
    Λ = diagm(rand(Distributions.Uniform(0, 10), N))
    Q = rand(RandomMatrices.Haar(1), N)
    return Q' * Λ * Q
end

function randomGaussian(N::Integer)
    # generate a dense random gaussian graphical model
    # 1. random mean vector
    μ =  rand(Uniform(-10, 10), N)
    # 2. random positive definite covariance matrix
    V = randomSPMatrix(N)
    # 3. return tuple of (mean,variance) as parametrization
    return (μ, V)
end

function randPrecision(ndim::Integer, nsample::Integer)
    ## generate random precision matrix that BP can converge to the true variance
    count = 0
    res = Array{Array{Float64,2},1}(nsample)
    total = 0
    while count < nsample
        total += 1
        _, Σ = randomGaussian(ndim)
        P = inv(Σ)
        P2 = P.^2
        crossSum = sum(P2, 2)
        crossMean = mean(crossSum)
        P[diagind(P)] += crossSum + rand(Uniform(0, crossMean), ndim)
        @assert all(eigvals(P) .> 0) "precision matrix not positive definite"
        # println("spectral radius: $ρ")
        count += 1
        res[count] = P
    end
    return res
end

function randPrecision(ndim::Integer)
    _, Σ = randomGaussian(ndim)
    P = inv(Σ)
    P2 = P.^2
    crossSum = sum(P2, 2)
    crossMean = mean(crossSum)
    P[diagind(P)] += crossSum + rand(Uniform(0, crossMean), ndim)
    @assert all(eigvals(P) .> 0) "precision matrix not positive definite"
    return P
end

function randomGaussianBPConvergence(ndim::Integer)
    # generate a random gaussian graphical model that can be exactly solved by BP
    # return: (mean, precision)
    # print("start generating bp graph")
    μ, Σ = randomGaussian(ndim)
    P = inv(Σ)
    P2 = P.^2
    crossSum = sum(P2, 2)
    crossMean = mean(crossSum)
    P[diagind(P)] += crossSum + rand(Uniform(0, crossMean), ndim)
    @assert all(eigvals(P) .> 0) "precision matrix not positive definite"
    # print(μ)
    # print("generated bp graph")
    return μ, P
end

function messagePassing(C::AbstractArray, niter::Integer;verbose = false)
    # message passing on a GGM with precision matrix C
    # C: original precision matrix
    # propagation rule:
    #   Pv_ij = C_ij^2/4Pf_ij
    #   Pf_jk = ∑_{i!=k} Pv_ij   # including Pv_jj
    niter -= 1
    n = size(C, 1)
    # initialize
    Pv = zeros(n, n)
    Pf = zeros(n, n)
    PvList = zeros(niter + 1, n, n)
    PfList = zeros(niter + 1, n, n)
    PrecisionList = zeros(niter + 1, n)

    for i in 1:n
        Pv[i,i] = C[i,i]
    end
    PvList[1,:,:] = Pv
    PfList[1,:,:] = Pf
    PrecisionList[1,:] = getMarginalPrecision(Pv)

    for q in 1:niter
        # 0.make a copy for convergence test
        oldPv = copy(Pv)
        oldPf = copy(Pf)
        # 1.update Pf
        for j in 1:n
            for k in 1:n
                if j != k
                    Pf[j,k] = sum(Pv[:,j]) - Pv[k,j]
                end
            end
        end
        # 2.update Pv
        for i in 1:n
            for j in 1:n
                if ( i != j ) && (C[i,j] != 0)
                    Pv[i,j] = -C[i,j]^2 / (4 * oldPf[i,j])
                end
            end
        end
        PvList[q + 1,:,:] = Pv
        PfList[q + 1,:,:] = Pf
        PrecisionList[q + 1,:] = getMarginalPrecision(Pv)
    end
    if verbose
        return Pv, Pf, PrecisionList
    else
        return Pv, Pf
    end
end

function messagePassingOneStep(Pv::AbstractArray, Pf::AbstractArray, C::AbstractArray)
    @assert size(Pv) == size(Pf) "size not match"
    @assert size(Pv, 1) == size(Pv, 2) "input should be square matrix"
    n = size(Pv, 1)
    oldPf = copy(Pf)
    for j in 1:n
        for k in 1:n
            if j != k
                Pf[j,k] = sum(Pv[:,j]) - Pv[k,j]
            end
        end
    end
    # 2.update Pv
    for i in 1:n
        for j in 1:n
            if (i != j) && (C[i,j] != 0)
                Pv[i,j] = -C[i,j]^2 / (oldPf[i,j])
            end
        end
    end
    return Pv, Pf
end

function getMarginalPrecision(Pv::AbstractMatrix)
    @assert size(Pv, 1) == size(Pv, 2) "input should be square matrix"
    return sum(Pv, 1)
end


function pruferToTree(seq)
    n = size(seq, 1)
    nnode = n + 2
    adj = zeros(nnode, nnode)
    degree = ones(nnode)
    for i in seq
        degree[i] += 1
    end
    for i in seq
        for jnode in 1:nnode
            if degree[jnode] == 1
                adj[i,jnode] = 1
                adj[jnode,i] = 1
                degree[i] -= 1
                degree[jnode] -= 1
                break
            end
        end
    end
    u = 0
    v = 0
    for inode in 1:nnode
        if degree[inode] == 1
            if u == 0
                u = inode
            else
                v = inode
                break
            end
        end
    end
    adj[u,v] = 1
    adj[v,u] = 1
    return adj
end

function randomTreeMatrix(ndim::Integer)
    n = ndim - 2
    seq = rand(1:ndim, n)
    adj = pruferToTree(seq)
    return adj
end


function generateRandomBPConvgenceGraph(ndim::Integer, bpIteration::Integer)
    # 1. generate a random precison matrix that will converge when applying BP on it
    μ, C = randomGaussianBPConvergence(ndim)
    # 2. run message passing algorithm and get extracted intermediate marginal precision vector
    Pv, Pf, PList = messagePassing(C, bpIteration;verbose = true)
    # 3. construct the data structure
    meanList = zeros(size(PList)...)
    g = GMM(μ, C;isPrecision = true)
    gbp = GMMBP(g, meanList, PList)
    return gbp
end


function generateRandomBPConvgenceTree(ndim::Integer, bpIteration::Integer)
    # 1. generate a random precison matrix that will converge when applying BP on it
    μ, C = randomGaussian(ndim)
    C .*= randomTreeMatrix(ndim) + eye(ndim)
    # 2. run message passing algorithm and get extracted intermediate marginal precision vector
    Pv = zeros(ndim, ndim)
    for k in 1:ndim
        Pv[k,k] = C[k,k]
    end
    Pf = zeros(ndim, ndim)
    for j in 1:bpIteration
        Pv, Pf = messagePassingOneStep(Pv, Pf, C)
    end
    # 3. construct the data structure
    meanList = zeros(size(PList)...)
    g = GMM(μ, C;isPrecision = true)
    gbp = GMMBP(g, meanList, PList)
    return gbp
end

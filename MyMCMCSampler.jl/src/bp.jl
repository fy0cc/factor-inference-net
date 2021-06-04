
using Base: @nloops, @ntuple
using Einsum
using LinearAlgebra
using TensorOperations

function computemarginal(d::DiscreteSpinGlass)
    # compute the marginal probiliby for each variable to take +1
    n = length(d)
    exp = quote 
        Z = 0.0
        mP = zeros($n) # marginal probability
        @nloops $n i (d)->[-1.0,1.0] begin
            x = @ntuple $n i
            x = collect(x)
            local p = exp(Distributions.logpdf($d,x))
            term = p.*x
            term[term.<0] .= 0.0
            global mP .+= term
            global Z += p
        end
        return mP./Z, Z
    end
    eval(exp)
end



function get_neighbours(d::Union{DiscreteSpinGlass, ThirdOrderExpFamily})
    """
    Get list of neighbor ids for both factor nodes and variable nodes
    """
    fidx = d.edge_index
    vidx = Array{Array{Int64,1},1}([[] for _ in 1:d.nnode])
    for i in 1:d.nfactor
        for j in fidx[i]
            push!(vidx[j], i)
        end
    end
    return fidx, vidx
end


function get_factor_tensor(d::DiscreteSpinGlass)
    """
    get factor tensor on support [-1,1]^r where r is the rank of the tensor
    """
    res = []
    for i in 1:d.nfactor
        idx = d.edge_index[i]
        supp = [1.; -1.]
        alpha = d.edge_attr[i][1] # assume scalar
        if length(idx) == 1
            @tensor begin
                T[i] := supp[i]
            end
        elseif length(idx) == 2
            @tensor begin 
                T[i,j] := supp[i] * supp[j]
            end
        elseif length(idx) == 3
            @tensor begin 
                T[i,j,k] := supp[i] * supp[j] * supp[k]
            end
        end
        T = @. exp(-alpha * T)
        T ./= maximum(T)
        push!(res,T)
    end
    return res
end

function bp_3spin(fidx::AbstractVector, vidx::AbstractVector, ftensor::AbstractVector; niter::Integer=1, atol=1e-5)
    nf = length(fidx)
    nv = length(vidx)
    ## Initialzie messages
    mv2f = Dict{Tuple,Array}()
    mf2v = Dict{Tuple,Array}()
    for i in 1:nf
        for j in fidx[i]
            mf2v[(i,j)] = ones(2) / 2.  # 2 state spin
            mv2f[(j,i)] = ones(2) / 2.
        end
    end
    
    ## main loop
    isconverged=false
    for iiter in 1:niter
        mv2f_new = deepcopy(mv2f)
        mf2v_new = deepcopy(mf2v)
        # update variable-to-factor messages
        for j in 1:nv
            source_idx = combinations(1:length(vidx[j]),length(vidx[j])-1) |> collect |> reverse
            for (neighbor_id, i) in enumerate(vidx[j])
                if length(source_idx) > 1
                    mv2f_new[(j,i)] = foldl((x,y)->x.*y,[mf2v[(vidx[j][k],j)] for k in source_idx[neighbor_id]])
                end
            end
        end
        # update factor-to-variable messages
        for i in 1:nf
            factor_size = length(fidx[i])
            source_idx = combinations(1:factor_size,factor_size-1) |> collect |> reverse
            for (neighbor_id, j) in enumerate(fidx[i])
                if factor_size == 1
                    mf2v_new[(i,j)] = ftensor[i]
                else
                    perm = collect(1:factor_size )
                    perm[neighbor_id] = 1
                    perm[1] = neighbor_id
                    ft_perm = permutedims(ftensor[i],perm)
                    if factor_size == 2
                        T = ft_perm
                        k1 = source_idx[neighbor_id][1]
                        m1 = mv2f[(fidx[i][k1],i)]
                        @einsum tmp[a] := T[a,b] * m1[b]
                        mf2v_new[(i,j)] .= tmp
                    elseif factor_size == 3
                        T = ft_perm
                        k1 = source_idx[neighbor_id][1]
                        k2 = source_idx[neighbor_id][2]
                        m1 = mv2f[(fidx[i][k1],i)]
                        m2 = mv2f[(fidx[i][k2],i)]
                        @einsum tmp[a] := T[a,b,c] * m1[b] * m2[c]
                        mf2v_new[(i,j)] .= tmp
                    end
                    @assert all(tmp .>= 0) "$tmp\n$(T) \n$(m1)\n$(m2)"
                end
            end
        end 
        # normalize message
        for k in keys(mv2f)   
            mv2f_new[k] = mv2f_new[k] ./ sum(mv2f_new[k])
        end
        for k in keys(mf2v)
            mf2v_new[k] = mf2v_new[k] ./ sum(mf2v_new[k])
        end
        # check convergence
        # assert convergent if all message change < atol 
        isconverged = true 
        for k in keys(mv2f)
            if abs(mv2f[k][1]-mv2f_new[k][1]) > atol
                isconverged = false
                break 
            end
        end
        for k in keys(mf2v)
            if abs(mf2v[k][1]-mf2v_new[k][1]) > atol
                isconverged = false
                break 
            end
        end
        mv2f = mv2f_new 
        mf2v = mf2v_new
    end
    
    ## Calculate marginal distributions
    # singleton marginals
    marginals = []
    for j in 1:nv
        incoming_messages = [mf2v[(i,j)] for i in vidx[j]]
        if length(incoming_messages)>1
            mP = foldl((x,y)->x.*y,incoming_messages) 
        else
            mP = ones(2) / .2
        end
        mP ./= sum(mP)
        push!(marginals,mP[1])
    end
    
    return marginals, isconverged, mf2v, mv2f
end

function bp(d::DiscreteSpinGlass; niter::Integer=10, atol::Real=1e-5)
    if !has_singleton_factor(d)
        d = add_singleton_factor(d)
    end
    fidx, vidx = get_neighbours(d)
    ftensor = get_factor_tensor(d) 
    mP_bp, isconverged, mf2v, mv2f = bp_3spin(fidx,vidx,ftensor, niter=niter, atol=atol)
    return mP_bp, isconverged
end





    
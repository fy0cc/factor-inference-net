function isvalid(sim::Chains)
    if !all(gelmandiag(sim).value[:,1,1] .< 1.01)
        return false 
    end
    for i in 1:size(sim.value)[2]
        data = mean(sim.value[:,i,:], dims = [1])[:]
        # mu = mean(data)
        sig = std(data)
        if sig > 0.05
            return false
        end
    end
    return true
end


function getstats(value::AbstractArray;standardize::Bool = false)
    # get statistics or natural parameter of given MCMC chains
    # in this function, return the first four moments
    m1 = mean(value, dims = [1,3])[:]
    m2 = var(value, dims = [1,3])[:]
    if standardize
        m3 = map(1:size(value)[2]) do i
            return skewness(value[:,i,:][:])
        end
        m3 = m3[:]
        m4 = map(1:size(value)[2]) do i 
            return kurtosis(value[:,i,:][:])
        end
        m4 = m4[:]
    else 
        m3 = mean(value.^3, dims = [1,3])[:]
        m4 = mean(value.^4, dims = [1,3])[:]
    end
    res = [m1 m2 m3 m4]
    @assert size(res) == (size(value)[2], 4) "size incorrect"
    return res
end
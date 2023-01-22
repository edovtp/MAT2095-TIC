function Dp(M, G0::Distribution, ε=1e-6)
    """
    Dirichlet Process simulation using the stick-breaking construction given by Sethuraman 
    (1994) and the ε-DP approximation (Mulliere, Tardella 1998). It is necessary to be
    able to draw at least one sample from G0.

    M  : precision parameter
    G0 : centering measure
    """
    weights = ElasticArrays.ElasticVector{Float64}(undef, 0)
    sum_l_ups = 0
    while sum(weights) < 1 - ε
        ups_sample = rand(Distributions.Beta(1, M))
        append!(weights, exp(log(ups_sample) + sum_l_ups))
        sum_l_ups += log(1 - ups_sample)
    end
    append!(weights, 1 - sum(weights))
    locations = [rand(G0) for _ in 1:length(weights)]

    return (locations=locations, weights=weights)
end

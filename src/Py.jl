function Py(M, b, G0::Distribution, ε=1e-6)
    """
    Pitman-Yor process simulation using the stick-breaking construction and a ϵ-PY approxi-
    mation. It is necessary to be able to draw at least one sample from G0.

    M  : precision parameter
    b  : discount parameter
    G0 : centering measure
    """
    weights = ElasticArrays.ElasticVector{Float64}(undef, 0)
    sum_l_ups = 0
    counter = 1
    while sum(weights) < 1 - ε
        ups_sample = rand(Distributions.Beta(1 - b, M + b * counter))
        append!(weights, exp(log(ups_sample) + sum_l_ups))
        sum_l_ups += log(1 - ups_sample)
        counter += 1
    end
    append!(weights, 1 - sum(weights))
    locations = [rand(G0) for _ in 1:length(weights)]

    return (locations=locations, weights=weights)
end

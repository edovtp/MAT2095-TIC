include("00_extras.jl")


# Dirichlet process simulation -----------------------------------------------
function tic_rdp(M, G0::Distribution, tol=1e-6)
    """
    Random sample of a (Finite) Dirichlet process using the stick-breaking construction

    M  : precision parameter
    G0 : centering measure

    Note: It's necessary that we can draw a sample of size >1 from G0 with rand()
    """
    
    probs = ElasticArrays.ElasticVector{Float64}(undef, 0)

    ups_dist = Distributions.Beta(1, M)
    sum_l_ups = 0
    while sum(probs) < 1 - tol
        ups_sample = rand(ups_dist)
        append!(probs, exp(log(ups_sample) + sum_l_ups))
        sum_l_ups += log(1 - ups_sample)
    end

    append!(probs, 1 - sum(probs))
    locations = rand(G0, length(probs))

    return (locations=locations, probs=probs)
end


# Data simulation from a Dirichlet Process ------------------------------------------
function tic_rdp_marginal(n, M, G0::Distribution)
    """
    Random sample from a Dirichlet process using the marginal distribution

    n  : sample length
    M  : precision parameter
    G0 : centering mesasure
    """

    marginal_sample = ElasticArray{Float64}(undef, length(G0), 0)
    counter = Dict{Any, Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(G0)
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        append!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample'
end

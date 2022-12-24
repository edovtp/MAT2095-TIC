using Random
using StatsBase
using Distributions
using ElasticArrays
using Plots
using StatsPlots

function RdpNigMarginal(n, M, par)
    """
    Data simulation from a Dirichlet process with NIG base measure using the Pólya Urn 
    representation of Blackwell and McQueen (1973).

    n   : sample length
    M   : precision parameter
    par : NIG parameters (m, λ, s, S)
    """
    m, λ, s, S = par
    marginal_sample = ElasticArray{Float64}(undef, 2, 0)
    counter = Dict{Any,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        V_sample = rand(InverseGamma(s, S))
        μ_sample = rand(Normal(m, sqrt(V_sample / λ)))
        candidate = (μ_sample, V_sample)
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

function RdpmNormal(n::Int64, M, par)
    """
    Random sample from a DPM of normals with NIG base measure. Returns the component
    parameters, the number of unique components (k) and the simulated data

    n   : sample length
    M   : precision parameter
    par : NIG parameters (m, λ, s, S)
    """
    θ = RdpNigMarginal(n, M, par)
    k = size(unique(θ, dims=1))[1]
    y = map(params -> rand(Normal(params[1], sqrt(params[2]))), eachrow(θ))

    return (y=y, θ=θ, k=k)
end

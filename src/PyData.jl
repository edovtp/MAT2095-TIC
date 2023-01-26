function DataPy(n::Int64, M, b, par, dist)
    """
    Data simulation from a Pitman-Yor process using the Pólya Urn representation.

    It is straightforward to change the dist parameter to an arbitrary function G0, but I
    prefer to be explicit for now.

    n   : sample length
    M   : precision parameter
    b   : discount parameter
    par : distribution parameters
    """
    if dist == "Normal"
        return DataPyNormal(n, M, b, par)
    elseif dist == "Gamma" 
        return DataPyGamma(n, M, b, par)
    end
end

# Specific functions
function DataPyNormal(n::Int64, M, b, par)
    """
    Normal base measure
    """
    μ, σ = par
    marginal_sample = ElasticVector{Float64}(undef, 0)
    counter = Dict{Float64, Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        k = length(all_values)
        candidate = rand(Normal(μ, σ))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [(old_freq .- b) .* norm_term, (M + b * k) * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataPyGamma(n::Int64, M, b, par)
    """
    Gamma(α, θ) base measure, θ the scale parameter
    """
    α, θ = par
    marginal_sample = ElasticVector{Float64}(undef, 0)
    counter = Dict{Float64, Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        k = length(all_values)
        candidate = rand(Gamma(α, θ))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [(old_freq .- b) .* norm_term; (M + b * k) * norm_term]
        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

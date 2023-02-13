function DataDp(n::Int64, M, par, dist)
    """
    Data simulation from a Dirichlet process using the Pólya Urn representation of
    Blackwell and McQueen (1973), i.e. the marginal distribution y_1, ..., y_n

    It is straightforward to change the dist parameter to an arbitrary function G0, but
    I prefer to be explicit for now.
    """
    if dist == "Normal"
        return DataDpNormal(n, M, par)
    elseif dist == "Gamma"
        return DataDpGamma(n, M, par)
    elseif dist == "Nig"
        return DataDpNig(n, M, par)
    elseif dist == "Niw"
        return DataDpNiw(n, M, par)
    end
end

# Specific functions
function DataDpNormal(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Normal base measure using the
    Pólya Urn representation of Blackwell and McQueen (1973).

    n   : sample length
    M   : precision parameter
    par : Normal parameters (μ, σ)
    """
    μ, σ = par
    marginal_sample = ElasticVector{Float64}(undef, 0)
    counter = Dict{Float64,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(Normal(μ, σ))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataDpGamma(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Gamma base measure using the
    Pólya Urn representation of Blackwell and McQueen (1973).

    n   : sample length
    M   : precision parameter
    par : Gamma parameters (α, θ) where θ is the scale parameter
    """
    α, θ = par
    marginal_sample = ElasticVector{Float64}(undef, 0)
    counter = Dict{Float64,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(Gamma(α, θ))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataDpNig(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Normal-Inv-Gamma base measure using the
    Pólya Urn representation of Blackwell and McQueen (1973).
    
    The parameterization of the base measure is such that γ scales σ. That is,
    G0 = Normal(μ|m, γσ^2)⋅Inv-Gamma(σ^2|s, S)

    n   : sample length
    M   : precision parameter
    par : NIG parameters (m, γ, s, S)
    """
    m, γ, s, S = par
    marginal_sample = ElasticVector{Tuple}(undef, 0)
    counter = Dict{Tuple,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate = rand(NormalInverseGamma(m, γ, s, S))
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end

function DataDpNiw(n::Int64, M, par)
    """
    Data simulation from a Dirichlet process with Normal-Inv-Wishart base measure using 
    the Pólya Urn representation of Blackwell and McQueen (1973).

    The parameterization of the base measure is such that γ scales Σ. That is,
    G0 = MvNormal(μ|m, γΣ)⋅Inv-Wishart(Σ|Ψ, ν), and E(Σ) = Ψ/(ν - p + 1) with p the
    dimension of the data.

    n   : sample length
    M   : precision parameter
    par : NIW parameters (m, γ, Ψ, ν)
    """
    m, γ, Ψ, ν = par
    marginal_sample = ElasticVector{Tuple}(undef, 0)
    counter = Dict{Tuple,Int64}()

    for i in 1:n
        all_values = collect(keys(counter))
        candidate_Σ = rand(InverseWishart(ν, Ψ))
        candidate_μ = rand(MvNormal(m, γ * candidate_Σ))
        candidate = (candidate_μ, candidate_Σ)
        push!(all_values, candidate)

        norm_term = 1 / (M + i - 1)
        old_freq = collect(values(counter))
        probs = [old_freq .* norm_term; M * norm_term]

        new_value = StatsBase.sample(all_values, Weights(probs))
        push!(marginal_sample, new_value)

        freq = get(counter, new_value, 0)
        counter[new_value] = freq + 1
    end

    return marginal_sample
end
